// Voronoi-neighborhood Gaussian fields for NeRF-style volume rendering of a fixed point cloud.
//
// Every site p_i (fixed) carries
//   * a density Gaussian:  rho_i = exp(raw_i),  Sigma_i = R(q_i) S(s_i)^2 R(q_i)^T,
//   * a color Gaussian (its own scale / rotation, same center p_i) and SH coefficients (degree <= 3).
// For a sample x whose nearest site is i, the neighborhood is N(i) = {i} U Delaunay neighbors of i and
//   sigma(x) = sum_{k in N(i)} rho_k G_k(x),                     G_k(x) = exp(-(x - p_k)^T Sigma_k^-1 (x - p_k) / 2)
//   C(x, d)  = sum_{k in N(i)} H_k(x) c_k(d) / sum_k H_k(x),     c_k(d) = max(SH_k(d) + 1/2, 0)
// with H_k the color Gaussian.
//
// Nearest sites: the samples of a ray are sorted, so each one starts a greedy walk on the Delaunay graph
// at the site of the previous sample (the first one at the site containing the camera) and moves to the
// closest neighbor while that neighbor is closer to x. A Voronoi cell is the intersection of the half
// spaces of its Delaunay neighbors, so the walk stops exactly at the nearest site.
//
// Backward passes recompute the neighborhoods from the saved sites and scatter gradients with atomics.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr float kInf = 3.402823466e38f;
constexpr int kBlock = 128;
constexpr int kWalkCap = 4096;     // safety bound on walk steps per sample
constexpr float kMinWeight = 1e-7f;  // smaller (normalized) contributions get no gradient

#define CHECK_INPUT(x, dtype)                                                                              \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                                               \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");                                            \
    TORCH_CHECK((x).scalar_type() == (dtype), #x " has the wrong dtype")

inline int blocks_for(int64_t n) { return static_cast<int>((n + kBlock - 1) / kBlock); }

struct Graph {
    const float* __restrict__ points;  // (N, 3)
    const int* __restrict__ offsets;   // (N + 1)
    const int* __restrict__ adj;       // (M)
};

__device__ inline float dist2(const float* __restrict__ p, int i, float x, float y, float z) {
    const float dx = x - __ldg(p + 3 * i), dy = y - __ldg(p + 3 * i + 1), dz = z - __ldg(p + 3 * i + 2);
    return dx * dx + dy * dy + dz * dz;
}

// ============================================================================================ walk

__global__ void __launch_bounds__(kBlock)
    walk_kernel(Graph g, const float* __restrict__ o, const float* __restrict__ d, const float* __restrict__ z,
                const int* __restrict__ start, int64_t B, int S, int* __restrict__ site, int* __restrict__ steps) {
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= B) return;
    const float ox = o[3 * r], oy = o[3 * r + 1], oz = o[3 * r + 2];
    const float dx = d[3 * r], dy = d[3 * r + 1], dz = d[3 * r + 2];
    int cur = start[r];
    int total = 0;
    for (int s = 0; s < S; ++s) {
        const float t = z[r * S + s];
        const float x = ox + t * dx, y = oy + t * dy, w = oz + t * dz;
        float best_d = dist2(g.points, cur, x, y, w);
        for (int it = 0; it < kWalkCap; ++it) {
            int best = cur;
            const int e1 = __ldg(g.offsets + cur + 1);
            for (int e = __ldg(g.offsets + cur); e < e1; ++e) {
                const int q = __ldg(g.adj + e);
                const float dq = dist2(g.points, q, x, y, w);
                if (dq < best_d) best_d = dq, best = q;
            }
            if (best == cur) break;
            cur = best;
            ++total;
        }
        site[r * S + s] = cur;
    }
    if (steps != nullptr) steps[r] = total;
}

// ======================================================================================= Gaussians

struct GaussParams {
    const float* __restrict__ log_scale;  // (N, 3)
    const float* __restrict__ quat;       // (N, 4) w x y z, not normalized
};

struct GaussLocal {
    float v[3];    // x - p
    float y[3];    // R^T (x - p)
    float is2[3];  // 1 / s^2
    float qn[4];   // normalized quaternion
    float qnorm;
    float R[9];  // row-major, columns are the Gaussian axes
    float m;     // Mahalanobis distance squared
};

__device__ inline void gauss_local(const float* __restrict__ points, const GaussParams& G, int i, float x,
                                   float y, float z, float inv_mult2, GaussLocal& L) {
    const int64_t i3 = 3 * static_cast<int64_t>(i);
    L.v[0] = x - __ldg(points + i3);
    L.v[1] = y - __ldg(points + i3 + 1);
    L.v[2] = z - __ldg(points + i3 + 2);
    const float4 q = __ldg(reinterpret_cast<const float4*>(G.quat) + i);
    L.qnorm = fmaxf(sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w), 1e-12f);
    const float w = q.x / L.qnorm, a = q.y / L.qnorm, b = q.z / L.qnorm, c = q.w / L.qnorm;
    L.qn[0] = w, L.qn[1] = a, L.qn[2] = b, L.qn[3] = c;
    L.R[0] = 1.f - 2.f * (b * b + c * c), L.R[1] = 2.f * (a * b - w * c), L.R[2] = 2.f * (a * c + w * b);
    L.R[3] = 2.f * (a * b + w * c), L.R[4] = 1.f - 2.f * (a * a + c * c), L.R[5] = 2.f * (b * c - w * a);
    L.R[6] = 2.f * (a * c - w * b), L.R[7] = 2.f * (b * c + w * a), L.R[8] = 1.f - 2.f * (a * a + b * b);
    L.m = 0.f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        L.y[k] = L.R[k] * L.v[0] + L.R[3 + k] * L.v[1] + L.R[6 + k] * L.v[2];
        const float s = __expf(fminf(fmaxf(__ldg(G.log_scale + i3 + k), -20.f), 20.f));
        L.is2[k] = inv_mult2 / (s * s);
        L.m += L.y[k] * L.y[k] * L.is2[k];
    }
}

// Accumulates dL/dlog_scale and dL/dquat of one Gaussian given gm = dL/dm.
__device__ inline void scatter_gauss_grad(const GaussLocal& L, float gm, int i, float* __restrict__ g_ls,
                                          float* __restrict__ g_quat) {
    const int64_t i3 = 3 * static_cast<int64_t>(i);
    float yk[3];  // dL/dy_k, m = sum_k y_k^2 / s_k^2, y = R^T v
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        yk[k] = 2.f * gm * L.y[k] * L.is2[k];
        atomicAdd(g_ls + i3 + k, -2.f * gm * L.y[k] * L.y[k] * L.is2[k]);
    }
    float GR[9];  // dL/dR_ab = v_a dL/dy_b
#pragma unroll
    for (int a2 = 0; a2 < 3; ++a2)
#pragma unroll
        for (int b2 = 0; b2 < 3; ++b2) GR[3 * a2 + b2] = L.v[a2] * yk[b2];
    const float w = L.qn[0], x = L.qn[1], y = L.qn[2], z = L.qn[3];
    float dq[4];
    dq[0] = 2.f * (-z * GR[1] + y * GR[2] + z * GR[3] - x * GR[5] - y * GR[6] + x * GR[7]);
    dq[1] = 2.f * (y * GR[1] + z * GR[2] + y * GR[3] - 2.f * x * GR[4] - w * GR[5] + z * GR[6] + w * GR[7] -
                   2.f * x * GR[8]);
    dq[2] = 2.f * (-2.f * y * GR[0] + x * GR[1] + w * GR[2] + x * GR[3] + z * GR[5] - w * GR[6] + z * GR[7] -
                   2.f * y * GR[8]);
    dq[3] = 2.f * (-2.f * z * GR[0] - w * GR[1] + x * GR[2] + w * GR[3] - 2.f * z * GR[4] + y * GR[5] +
                   x * GR[6] + y * GR[7]);
    const float proj = dq[0] * w + dq[1] * x + dq[2] * y + dq[3] * z;
    float* gq = g_quat + 4 * static_cast<int64_t>(i);
#pragma unroll
    for (int k = 0; k < 4; ++k) atomicAdd(gq + k, (dq[k] - proj * L.qn[k]) / L.qnorm);
}

// Iterates k over N(i) = {i} U adj(i).
#define FOR_NEIGHBORHOOD(g, i, k, ...)                                                    \
    {                                                                                     \
        const int e0_ = __ldg((g).offsets + (i)), e1_ = __ldg((g).offsets + (i) + 1);     \
        for (int e_ = e0_ - 1; e_ < e1_; ++e_) {                                          \
            const int k = e_ < e0_ ? (i) : __ldg((g).adj + e_);                           \
            __VA_ARGS__                                                                   \
        }                                                                                 \
    }

// ========================================================================================= density

__global__ void __launch_bounds__(kBlock)
    density_forward_kernel(Graph g, const float* __restrict__ raw, GaussParams G, const float* __restrict__ x,
                           const int* __restrict__ site, int64_t m, float inv_mult2, float* __restrict__ sigma) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= m) return;
    const float qx = x[3 * t], qy = x[3 * t + 1], qz = x[3 * t + 2];
    const int i = site[t];
    float s = 0.f;
    FOR_NEIGHBORHOOD(g, i, k, {
        GaussLocal L;
        gauss_local(g.points, G, k, qx, qy, qz, inv_mult2, L);
        s += __expf(fminf(__ldg(raw + k), 30.f) - 0.5f * L.m);
    })
    sigma[t] = s;
}

__global__ void __launch_bounds__(kBlock)
    density_backward_kernel(Graph g, const float* __restrict__ raw, GaussParams G, const float* __restrict__ x,
                            const int* __restrict__ site, int64_t m, const float* __restrict__ grad_sigma,
                            float* __restrict__ g_raw, float* __restrict__ g_ls, float* __restrict__ g_quat) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= m) return;
    const float gs = grad_sigma[t];
    if (gs == 0.f) return;
    const float qx = x[3 * t], qy = x[3 * t + 1], qz = x[3 * t + 2];
    const int i = site[t];
    FOR_NEIGHBORHOOD(g, i, k, {
        GaussLocal L;
        gauss_local(g.points, G, k, qx, qy, qz, 1.f, L);
        const float c = __expf(fminf(__ldg(raw + k), 30.f) - 0.5f * L.m);  // rho_k G_k
        if (c * fabsf(gs) > 1e-12f) {
            atomicAdd(g_raw + k, gs * c);
            scatter_gauss_grad(L, -0.5f * gs * c, k, g_ls, g_quat);
        }
    })
}

// =========================================================================================== color

__device__ inline int sh_count(int deg) { return (deg + 1) * (deg + 1); }

// Real spherical harmonics basis (same constants and signs as 3D Gaussian splatting).
__device__ inline void sh_basis(float x, float y, float z, int deg, float (&b)[16]) {
    b[0] = 0.28209479177387814f;
#pragma unroll
    for (int i = 1; i < 16; ++i) b[i] = 0.f;
    if (deg < 1) return;
    b[1] = -0.4886025119029199f * y;
    b[2] = 0.4886025119029199f * z;
    b[3] = -0.4886025119029199f * x;
    if (deg < 2) return;
    const float xx = x * x, yy = y * y, zz = z * z;
    b[4] = 1.0925484305920792f * x * y;
    b[5] = -1.0925484305920792f * y * z;
    b[6] = 0.31539156525252005f * (2.f * zz - xx - yy);
    b[7] = -1.0925484305920792f * x * z;
    b[8] = 0.5462742152960396f * (xx - yy);
    if (deg < 3) return;
    b[9] = -0.5900435899266435f * y * (3.f * xx - yy);
    b[10] = 2.890611442640554f * x * y * z;
    b[11] = -0.4570457994644658f * y * (4.f * zz - xx - yy);
    b[12] = 0.3731763325901154f * z * (2.f * zz - 3.f * xx - 3.f * yy);
    b[13] = -0.4570457994644658f * x * (4.f * zz - xx - yy);
    b[14] = 1.445305721320277f * z * (xx - yy);
    b[15] = -0.5900435899266435f * x * (xx - 3.f * yy);
}

__device__ inline void sh_eval(const float* __restrict__ sh, int k, const float (&b)[16], int nb, float (&c)[3]) {
    const float* p = sh + 48 * static_cast<int64_t>(k);
    c[0] = c[1] = c[2] = 0.5f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (j >= nb) break;
        c[0] += b[j] * __ldg(p + 3 * j);
        c[1] += b[j] * __ldg(p + 3 * j + 1);
        c[2] += b[j] * __ldg(p + 3 * j + 2);
    }
}

__device__ inline void unit_dir(const float* __restrict__ dirs, int64_t t, float& dx, float& dy, float& dz) {
    dx = dirs[3 * t], dy = dirs[3 * t + 1], dz = dirs[3 * t + 2];
    const float inv = rsqrtf(fmaxf(dx * dx + dy * dy + dz * dz, 1e-20f));
    dx *= inv, dy *= inv, dz *= inv;
}

__global__ void __launch_bounds__(kBlock)
    color_forward_kernel(Graph g, GaussParams H, const float* __restrict__ sh, int sh_deg,
                         const float* __restrict__ x, const float* __restrict__ dirs, const int* __restrict__ site,
                         int64_t m, float* __restrict__ rgb) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= m) return;
    const float qx = x[3 * t], qy = x[3 * t + 1], qz = x[3 * t + 2];
    float dx, dy, dz;
    unit_dir(dirs, t, dx, dy, dz);
    float b[16];
    sh_basis(dx, dy, dz, sh_deg, b);
    const int nb = sh_count(sh_deg);
    const int i = site[t];
    // online normalized sum: weights exp(-(m_k - m_min) / 2)
    float mmin = kInf, A = 0.f, cr = 0.f, cg = 0.f, cb = 0.f;
    FOR_NEIGHBORHOOD(g, i, k, {
        GaussLocal L;
        gauss_local(g.points, H, k, qx, qy, qz, 1.f, L);
        if (L.m < mmin) {
            const float f = mmin == kInf ? 0.f : __expf(-0.5f * (mmin - L.m));
            A *= f, cr *= f, cg *= f, cb *= f;
            mmin = L.m;
        }
        const float a = __expf(-0.5f * (L.m - mmin));
        float c[3];
        sh_eval(sh, k, b, nb, c);
        A += a;
        cr += a * fmaxf(c[0], 0.f);
        cg += a * fmaxf(c[1], 0.f);
        cb += a * fmaxf(c[2], 0.f);
    })
    rgb[3 * t] = cr / A;
    rgb[3 * t + 1] = cg / A;
    rgb[3 * t + 2] = cb / A;
}

__global__ void __launch_bounds__(kBlock)
    color_backward_kernel(Graph g, GaussParams H, const float* __restrict__ sh, int sh_deg,
                          const float* __restrict__ x, const float* __restrict__ dirs, const int* __restrict__ site,
                          int64_t m, const float* __restrict__ rgb, const float* __restrict__ grad_rgb,
                          float* __restrict__ g_ls, float* __restrict__ g_quat, float* __restrict__ g_sh) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= m) return;
    const float gr = grad_rgb[3 * t], gg = grad_rgb[3 * t + 1], gb = grad_rgb[3 * t + 2];
    if (gr == 0.f && gg == 0.f && gb == 0.f) return;
    const float qx = x[3 * t], qy = x[3 * t + 1], qz = x[3 * t + 2];
    const int i = site[t];
    float mmin = kInf;
    FOR_NEIGHBORHOOD(g, i, k, {
        GaussLocal L;
        gauss_local(g.points, H, k, qx, qy, qz, 1.f, L);
        mmin = fminf(mmin, L.m);
    })
    float A = 0.f;
    FOR_NEIGHBORHOOD(g, i, k, {
        GaussLocal L;
        gauss_local(g.points, H, k, qx, qy, qz, 1.f, L);
        A += __expf(-0.5f * (L.m - mmin));
    })
    const float inv_a = 1.f / A;
    const float gCC = gr * rgb[3 * t] + gg * rgb[3 * t + 1] + gb * rgb[3 * t + 2];
    float dx, dy, dz;
    unit_dir(dirs, t, dx, dy, dz);
    float b[16];
    sh_basis(dx, dy, dz, sh_deg, b);
    const int nb = sh_count(sh_deg);
    FOR_NEIGHBORHOOD(g, i, k, {
        GaussLocal L;
        gauss_local(g.points, H, k, qx, qy, qz, 1.f, L);
        const float a = __expf(-0.5f * (L.m - mmin)) * inv_a;
        if (a >= kMinWeight) {
            float c[3];
            sh_eval(sh, k, b, nb, c);
            const float dc0 = c[0] > 0.f ? a * gr : 0.f, dc1 = c[1] > 0.f ? a * gg : 0.f,
                        dc2 = c[2] > 0.f ? a * gb : 0.f;
            float* gs = g_sh + 48 * static_cast<int64_t>(k);
            for (int j = 0; j < nb; ++j) {
                if (dc0 != 0.f) atomicAdd(gs + 3 * j, b[j] * dc0);
                if (dc1 != 0.f) atomicAdd(gs + 3 * j + 1, b[j] * dc1);
                if (dc2 != 0.f) atomicAdd(gs + 3 * j + 2, b[j] * dc2);
            }
            // dL/dm_k = -a_k / 2 (<dL/dC, c_k> - <dL/dC, C>)
            const float ga = gr * fmaxf(c[0], 0.f) + gg * fmaxf(c[1], 0.f) + gb * fmaxf(c[2], 0.f);
            const float gm = -0.5f * a * (ga - gCC);
            if (gm != 0.f) scatter_gauss_grad(L, gm, k, g_ls, g_quat);
        }
    })
}

// ============================================================================================ conics

// Rotation (row-major, columns = Gaussian axes) of a normalized quaternion (w, x, y, z).
__device__ inline void quat_to_rot(float w, float a, float b, float c, float (&R)[9]) {
    R[0] = 1.f - 2.f * (b * b + c * c), R[1] = 2.f * (a * b - w * c), R[2] = 2.f * (a * c + w * b);
    R[3] = 2.f * (a * b + w * c), R[4] = 1.f - 2.f * (a * a + c * c), R[5] = 2.f * (b * c - w * a);
    R[6] = 2.f * (a * c - w * b), R[7] = 2.f * (b * c + w * a), R[8] = 1.f - 2.f * (a * a + b * b);
}

struct ConicLocal {
    float qn[4], qnorm, R[9], iv[3], s2[3];
};

__device__ inline void conic_local(const float* __restrict__ log_scale, const float* __restrict__ quat, int64_t i,
                                   float floor2, ConicLocal& L) {
    const float4 q = reinterpret_cast<const float4*>(quat)[i];
    L.qnorm = fmaxf(sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w), 1e-12f);
    L.qn[0] = q.x / L.qnorm, L.qn[1] = q.y / L.qnorm, L.qn[2] = q.z / L.qnorm, L.qn[3] = q.w / L.qnorm;
    quat_to_rot(L.qn[0], L.qn[1], L.qn[2], L.qn[3], L.R);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float s = __expf(2.f * fminf(fmaxf(log_scale[3 * i + k], -20.f), 20.f));
        L.s2[k] = s;
        L.iv[k] = 1.f / (s + floor2);
    }
}

// A = R diag(1 / (s^2 + floor^2)) R^T as (xx, xy, xz, yy, yz, zz).
__global__ void __launch_bounds__(kBlock)
    conic_forward_kernel(const float* __restrict__ log_scale, const float* __restrict__ quat, int64_t n, float floor2,
                         float* __restrict__ A) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ConicLocal L;
    conic_local(log_scale, quat, i, floor2, L);
    const float* R = L.R;
    constexpr int ab[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
#pragma unroll
    for (int e = 0; e < 6; ++e) {
        const int a = ab[e][0], b = ab[e][1];
        A[6 * i + e] = R[3 * a] * R[3 * b] * L.iv[0] + R[3 * a + 1] * R[3 * b + 1] * L.iv[1] +
                       R[3 * a + 2] * R[3 * b + 2] * L.iv[2];
    }
}

__global__ void __launch_bounds__(kBlock)
    conic_backward_kernel(const float* __restrict__ log_scale, const float* __restrict__ quat, int64_t n,
                          float floor2, const float* __restrict__ gA, float* __restrict__ g_ls,
                          float* __restrict__ g_quat) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ConicLocal L;
    conic_local(log_scale, quat, i, floor2, L);
    const float* R = L.R;
    const float* g = gA + 6 * i;
    // symmetric dL/dA (off-diagonal storage entries stand for both A_ab and A_ba)
    const float G[9] = {g[0], 0.5f * g[1], 0.5f * g[2], 0.5f * g[1], g[3], 0.5f * g[4], 0.5f * g[2], 0.5f * g[4], g[5]};
    float GR_[9];  // G R
#pragma unroll
    for (int a = 0; a < 3; ++a)
#pragma unroll
        for (int k = 0; k < 3; ++k) GR_[3 * a + k] = G[3 * a] * R[k] + G[3 * a + 1] * R[3 + k] + G[3 * a + 2] * R[6 + k];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float g_iv = R[k] * GR_[k] + R[3 + k] * GR_[3 + k] + R[6 + k] * GR_[6 + k];  // (R^T G R)_kk
        g_ls[3 * i + k] = g_iv * (-2.f * L.s2[k] * L.iv[k] * L.iv[k]);
    }
    float dR[9];  // dL/dR = 2 G R diag(iv)
#pragma unroll
    for (int a = 0; a < 3; ++a)
#pragma unroll
        for (int k = 0; k < 3; ++k) dR[3 * a + k] = 2.f * GR_[3 * a + k] * L.iv[k];
    const float w = L.qn[0], x = L.qn[1], y = L.qn[2], z = L.qn[3];
    float dq[4];
    dq[0] = 2.f * (-z * dR[1] + y * dR[2] + z * dR[3] - x * dR[5] - y * dR[6] + x * dR[7]);
    dq[1] = 2.f * (y * dR[1] + z * dR[2] + y * dR[3] - 2.f * x * dR[4] - w * dR[5] + z * dR[6] + w * dR[7] -
                   2.f * x * dR[8]);
    dq[2] = 2.f * (-2.f * y * dR[0] + x * dR[1] + w * dR[2] + x * dR[3] + z * dR[5] - w * dR[6] + z * dR[7] -
                   2.f * y * dR[8]);
    dq[3] = 2.f * (-2.f * z * dR[0] - w * dR[1] + x * dR[2] + w * dR[3] - 2.f * z * dR[4] + y * dR[5] +
                   x * dR[6] + y * dR[7]);
    const float proj = dq[0] * w + dq[1] * x + dq[2] * y + dq[3] * z;
#pragma unroll
    for (int k = 0; k < 4; ++k) g_quat[4 * i + k] = (dq[k] - proj * L.qn[k]) / L.qnorm;
}

Graph make_graph(const torch::Tensor& points, const torch::Tensor& offsets, const torch::Tensor& adj) {
    CHECK_INPUT(points, torch::kFloat32);
    CHECK_INPUT(offsets, torch::kInt32);
    CHECK_INPUT(adj, torch::kInt32);
    return {points.data_ptr<float>(), offsets.data_ptr<int>(), adj.data_ptr<int>()};
}

GaussParams make_gauss(const torch::Tensor& log_scale, const torch::Tensor& quat) {
    CHECK_INPUT(log_scale, torch::kFloat32);
    CHECK_INPUT(quat, torch::kFloat32);
    return {log_scale.data_ptr<float>(), quat.data_ptr<float>()};
}

}  // namespace

std::vector<torch::Tensor> walk(torch::Tensor points, torch::Tensor offsets, torch::Tensor adj, torch::Tensor o,
                                torch::Tensor d, torch::Tensor z, torch::Tensor start) {
    const Graph g = make_graph(points, offsets, adj);
    CHECK_INPUT(o, torch::kFloat32);
    CHECK_INPUT(d, torch::kFloat32);
    CHECK_INPUT(z, torch::kFloat32);
    CHECK_INPUT(start, torch::kInt32);
    const at::cuda::CUDAGuard guard(z.device());
    const int64_t B = z.size(0);
    const int S = static_cast<int>(z.size(1));
    auto site = torch::empty({B, S}, z.options().dtype(torch::kInt32));
    auto steps = torch::empty({B}, z.options().dtype(torch::kInt32));
    if (B == 0) return {site, steps};
    walk_kernel<<<blocks_for(B), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        g, o.data_ptr<float>(), d.data_ptr<float>(), z.data_ptr<float>(), start.data_ptr<int>(), B, S,
        site.data_ptr<int>(), steps.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {site, steps};
}

torch::Tensor density_forward(torch::Tensor points, torch::Tensor offsets, torch::Tensor adj, torch::Tensor raw,
                              torch::Tensor log_scale, torch::Tensor quat, torch::Tensor x, torch::Tensor site,
                              double scale_mult) {
    const Graph g = make_graph(points, offsets, adj);
    const GaussParams G = make_gauss(log_scale, quat);
    CHECK_INPUT(raw, torch::kFloat32);
    CHECK_INPUT(x, torch::kFloat32);
    CHECK_INPUT(site, torch::kInt32);
    const at::cuda::CUDAGuard guard(x.device());
    const int64_t m = x.size(0);
    auto sigma = torch::empty({m}, x.options());
    if (m == 0) return sigma;
    density_forward_kernel<<<blocks_for(m), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        g, raw.data_ptr<float>(), G, x.data_ptr<float>(), site.data_ptr<int>(), m,
        static_cast<float>(1.0 / (scale_mult * scale_mult)), sigma.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sigma;
}

std::vector<torch::Tensor> density_backward(torch::Tensor points, torch::Tensor offsets, torch::Tensor adj,
                                            torch::Tensor raw, torch::Tensor log_scale, torch::Tensor quat,
                                            torch::Tensor x, torch::Tensor site, torch::Tensor grad_sigma) {
    const Graph g = make_graph(points, offsets, adj);
    const GaussParams G = make_gauss(log_scale, quat);
    CHECK_INPUT(grad_sigma, torch::kFloat32);
    const at::cuda::CUDAGuard guard(x.device());
    const int64_t m = x.size(0);
    auto g_raw = torch::zeros_like(raw), g_ls = torch::zeros_like(log_scale), g_quat = torch::zeros_like(quat);
    if (m == 0) return {g_raw, g_ls, g_quat};
    density_backward_kernel<<<blocks_for(m), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        g, raw.data_ptr<float>(), G, x.data_ptr<float>(), site.data_ptr<int>(), m, grad_sigma.data_ptr<float>(),
        g_raw.data_ptr<float>(), g_ls.data_ptr<float>(), g_quat.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_raw, g_ls, g_quat};
}

torch::Tensor color_forward(torch::Tensor points, torch::Tensor offsets, torch::Tensor adj, torch::Tensor log_scale,
                            torch::Tensor quat, torch::Tensor sh, int64_t sh_deg, torch::Tensor x, torch::Tensor dirs,
                            torch::Tensor site) {
    const Graph g = make_graph(points, offsets, adj);
    const GaussParams H = make_gauss(log_scale, quat);
    CHECK_INPUT(sh, torch::kFloat32);
    CHECK_INPUT(x, torch::kFloat32);
    CHECK_INPUT(dirs, torch::kFloat32);
    CHECK_INPUT(site, torch::kInt32);
    TORCH_CHECK(sh.dim() == 3 && sh.size(1) == 16 && sh.size(2) == 3, "sh must be (N, 16, 3)");
    const at::cuda::CUDAGuard guard(x.device());
    const int64_t m = x.size(0);
    auto rgb = torch::empty({m, 3}, x.options());
    if (m == 0) return rgb;
    color_forward_kernel<<<blocks_for(m), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        g, H, sh.data_ptr<float>(), static_cast<int>(sh_deg), x.data_ptr<float>(), dirs.data_ptr<float>(),
        site.data_ptr<int>(), m, rgb.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return rgb;
}

std::vector<torch::Tensor> color_backward(torch::Tensor points, torch::Tensor offsets, torch::Tensor adj,
                                          torch::Tensor log_scale, torch::Tensor quat, torch::Tensor sh,
                                          int64_t sh_deg, torch::Tensor x, torch::Tensor dirs, torch::Tensor site,
                                          torch::Tensor rgb, torch::Tensor grad_rgb) {
    const Graph g = make_graph(points, offsets, adj);
    const GaussParams H = make_gauss(log_scale, quat);
    CHECK_INPUT(rgb, torch::kFloat32);
    CHECK_INPUT(grad_rgb, torch::kFloat32);
    const at::cuda::CUDAGuard guard(x.device());
    const int64_t m = x.size(0);
    auto g_ls = torch::zeros_like(log_scale), g_quat = torch::zeros_like(quat), g_sh = torch::zeros_like(sh);
    if (m == 0) return {g_ls, g_quat, g_sh};
    color_backward_kernel<<<blocks_for(m), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        g, H, sh.data_ptr<float>(), static_cast<int>(sh_deg), x.data_ptr<float>(), dirs.data_ptr<float>(),
        site.data_ptr<int>(), m, rgb.data_ptr<float>(), grad_rgb.data_ptr<float>(), g_ls.data_ptr<float>(),
        g_quat.data_ptr<float>(), g_sh.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_ls, g_quat, g_sh};
}

torch::Tensor conic_forward(torch::Tensor log_scale, torch::Tensor quat, double floor) {
    CHECK_INPUT(log_scale, torch::kFloat32);
    CHECK_INPUT(quat, torch::kFloat32);
    const at::cuda::CUDAGuard guard(quat.device());
    const int64_t n = quat.size(0);
    auto A = torch::empty({n, 6}, quat.options());
    if (n == 0) return A;
    conic_forward_kernel<<<blocks_for(n), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        log_scale.data_ptr<float>(), quat.data_ptr<float>(), n, static_cast<float>(floor * floor), A.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return A;
}

std::vector<torch::Tensor> conic_backward(torch::Tensor log_scale, torch::Tensor quat, double floor, torch::Tensor gA) {
    CHECK_INPUT(log_scale, torch::kFloat32);
    CHECK_INPUT(quat, torch::kFloat32);
    CHECK_INPUT(gA, torch::kFloat32);
    const at::cuda::CUDAGuard guard(quat.device());
    const int64_t n = quat.size(0);
    auto g_ls = torch::empty_like(log_scale), g_quat = torch::empty_like(quat);
    if (n == 0) return {g_ls, g_quat};
    conic_backward_kernel<<<blocks_for(n), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        log_scale.data_ptr<float>(), quat.data_ptr<float>(), n, static_cast<float>(floor * floor), gA.data_ptr<float>(),
        g_ls.data_ptr<float>(), g_quat.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_ls, g_quat};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conic_forward", &conic_forward, "Inverse covariances R diag(1 / (s^2 + floor^2)) R^T, (N, 6)");
    m.def("conic_backward", &conic_backward, "Gradients of the inverse covariances");
    m.def("walk", &walk, "Nearest site of every (sorted) ray sample by a greedy walk on the Delaunay graph");
    m.def("density_forward", &density_forward, "sum of the density Gaussians of the sample's Voronoi neighborhood");
    m.def("density_backward", &density_backward, "Gradients of the density field");
    m.def("color_forward", &color_forward, "Normalized blend of the color Gaussians with SH colors");
    m.def("color_backward", &color_backward, "Gradients of the color field");
}
