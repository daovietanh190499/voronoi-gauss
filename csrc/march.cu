// Fused uniform-step ray marching of the Voronoi-Gaussian field (Plenoxels-style sampling).
//
// One warp per ray; the 32 lanes split the Voronoi neighborhood {i} U adj(i) of the current site. Samples
// are the midpoints of fixed steps dt inside the unit ball, then n_bg steps uniform in disparity out to the
// far sphere. At each sample the warp walks to the nearest site, sums the density Gaussians, and (when the
// sample's weight exceeds w_thresh) blends the color Gaussians; the ray stops once T < t_thresh.
//
// Gaussians use precomputed conics A = Sigma^-1 stored as (xx, xy, xz, yy, yz, zz), so
// m = v^T A v = A_xx v_x^2 + A_yy v_y^2 + A_zz v_z^2 + 2 (A_xy v_x v_y + A_xz v_x v_z + A_yz v_y v_z).
//
// The backward pass re-marches every ray with identical arithmetic and uses
//   dC_fg/dsigma_k = delta_k (T_{k+1} c_k - (C_fg - C_fg,<=k)),   dT_final/dsigma_k = -delta_k T_final.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>

namespace {

constexpr float kInf = 3.402823466e38f;
constexpr int kBlock = 128;
constexpr int kWalkCap = 4096;
constexpr float kMinWeight = 1e-7f;
constexpr unsigned kFull = 0xffffffffu;

#define CHECK_INPUT(x, dtype)                                                                              \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                                               \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");                                            \
    TORCH_CHECK((x).scalar_type() == (dtype), #x " has the wrong dtype")

struct Graph {
    const float* __restrict__ points;
    const int* __restrict__ offsets;
    const int* __restrict__ adj;
};

struct Field {
    const float* __restrict__ raw;  // (N)
    const float* __restrict__ dA;   // (N, 6) density conics
    const float* __restrict__ cA;   // (N, 6) color conics
    const float* __restrict__ sh;   // (N, 16, 3)
    int sh_deg;
};

struct MarchCfg {
    float dt;
    int n_bg;
    float t_thresh;
    float w_thresh;
};

struct Ray {
    float o[3], d[3];
    float near, mid, far;
    int n_in, n_total;
};

__device__ inline float warp_sum(float v) {
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(kFull, v, s);
    return v;
}

__device__ inline float warp_min(float v) {
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) v = fminf(v, __shfl_xor_sync(kFull, v, s));
    return v;
}

__device__ inline int neighbor(const Graph& g, int i, int e0, int e) { return e == 0 ? i : __ldg(g.adj + e0 + e - 1); }

__device__ inline void offset_to(const float* __restrict__ p, int k, const float (&x)[3], float (&v)[3]) {
    const int64_t k3 = 3 * static_cast<int64_t>(k);
    v[0] = x[0] - __ldg(p + k3), v[1] = x[1] - __ldg(p + k3 + 1), v[2] = x[2] - __ldg(p + k3 + 2);
}

__device__ inline float mahalanobis(const float* __restrict__ A, int k, const float (&v)[3]) {
    const float* a = A + 6 * static_cast<int64_t>(k);
    return __ldg(a) * v[0] * v[0] + __ldg(a + 3) * v[1] * v[1] + __ldg(a + 5) * v[2] * v[2] +
           2.f * (__ldg(a + 1) * v[0] * v[1] + __ldg(a + 2) * v[0] * v[2] + __ldg(a + 4) * v[1] * v[2]);
}

__device__ inline void scatter_conic(float* __restrict__ gA, int k, float gm, const float (&v)[3]) {
    float* a = gA + 6 * static_cast<int64_t>(k);
    atomicAdd(a, gm * v[0] * v[0]);
    atomicAdd(a + 1, 2.f * gm * v[0] * v[1]);
    atomicAdd(a + 2, 2.f * gm * v[0] * v[2]);
    atomicAdd(a + 3, gm * v[1] * v[1]);
    atomicAdd(a + 4, 2.f * gm * v[1] * v[2]);
    atomicAdd(a + 5, gm * v[2] * v[2]);
}

// Real SH basis, same constants and signs as 3D Gaussian splatting.
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

__device__ inline void load_ray(int64_t r, const float* o, const float* d, const float* near, const float* mid,
                                const float* far, const MarchCfg& M, Ray& R) {
#pragma unroll
    for (int a = 0; a < 3; ++a) R.o[a] = o[3 * r + a], R.d[a] = d[3 * r + a];
    R.near = near[r], R.mid = mid[r], R.far = far[r];
    R.n_in = R.mid > R.near ? static_cast<int>(ceilf((R.mid - R.near) / M.dt)) : 0;
    R.n_total = R.n_in + M.n_bg;
}

__device__ inline void sample_at(const Ray& R, const MarchCfg& M, int k, float& t, float& delta) {
    if (k < R.n_in) {
        const float t0 = R.near + k * M.dt, t1 = fminf(t0 + M.dt, R.mid);
        t = 0.5f * (t0 + t1), delta = t1 - t0;
    } else {
        const float i0 = 1.f / R.mid, i1 = 1.f / R.far;
        const float u0 = static_cast<float>(k - R.n_in) / M.n_bg, u1 = static_cast<float>(k - R.n_in + 1) / M.n_bg;
        t = 1.f / (i0 + 0.5f * (u0 + u1) * (i1 - i0));
        delta = 1.f / (i0 + u1 * (i1 - i0)) - 1.f / (i0 + u0 * (i1 - i0));
    }
}

// Warp-cooperative greedy walk to the nearest site of x (ties broken by the smaller index, so every move
// strictly decreases (distance, index) and the walk terminates).
__device__ inline int walk_to(const Graph& g, int cur, const float (&x)[3], int lane, int& steps) {
    for (int it = 0; it < kWalkCap; ++it) {
        const int e0 = __ldg(g.offsets + cur), n = __ldg(g.offsets + cur + 1) - e0 + 1;
        float bd = kInf;
        int bk = INT_MAX;
        for (int e = lane; e < n; e += 32) {
            const int k = neighbor(g, cur, e0, e);
            float v[3];
            offset_to(g.points, k, x, v);
            const float dk = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if (dk < bd || (dk == bd && k < bk)) bd = dk, bk = k;
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            const float od = __shfl_xor_sync(kFull, bd, s);
            const int ok = __shfl_xor_sync(kFull, bk, s);
            if (od < bd || (od == bd && ok < bk)) bd = od, bk = ok;
        }
        if (bk == cur) break;
        cur = bk;
        ++steps;
    }
    return cur;
}

__device__ inline float density_at(const Graph& g, const Field& F, int i, const float (&x)[3], int lane) {
    const int e0 = __ldg(g.offsets + i), n = __ldg(g.offsets + i + 1) - e0 + 1;
    float s = 0.f;
    for (int e = lane; e < n; e += 32) {
        const int k = neighbor(g, i, e0, e);
        float v[3];
        offset_to(g.points, k, x, v);
        s += __expf(fminf(__ldg(F.raw + k), 30.f) - 0.5f * mahalanobis(F.dA, k, v));
    }
    return warp_sum(s);
}

// Normalized blend of the neighborhood colors; also returns m_min and 1 / sum of the shifted weights.
__device__ inline void color_at(const Graph& g, const Field& F, int i, const float (&x)[3], const float (&b)[16],
                                int lane, float (&C)[3], float& mmin, float& inv_a) {
    const int e0 = __ldg(g.offsets + i), n = __ldg(g.offsets + i + 1) - e0 + 1;
    const int nb = (F.sh_deg + 1) * (F.sh_deg + 1);
    float ml = kInf;
    for (int e = lane; e < n; e += 32) {
        const int k = neighbor(g, i, e0, e);
        float v[3];
        offset_to(g.points, k, x, v);
        ml = fminf(ml, mahalanobis(F.cA, k, v));
    }
    mmin = warp_min(ml);
    float A = 0.f, cr = 0.f, cg = 0.f, cb = 0.f;
    for (int e = lane; e < n; e += 32) {
        const int k = neighbor(g, i, e0, e);
        float v[3];
        offset_to(g.points, k, x, v);
        const float a = __expf(-0.5f * (mahalanobis(F.cA, k, v) - mmin));
        float c[3];
        sh_eval(F.sh, k, b, nb, c);
        A += a, cr += a * fmaxf(c[0], 0.f), cg += a * fmaxf(c[1], 0.f), cb += a * fmaxf(c[2], 0.f);
    }
    inv_a = 1.f / warp_sum(A);
    C[0] = warp_sum(cr) * inv_a, C[1] = warp_sum(cg) * inv_a, C[2] = warp_sum(cb) * inv_a;
}

__global__ void __launch_bounds__(kBlock)
    march_forward_kernel(Graph g, Field F, MarchCfg M, const float* __restrict__ o, const float* __restrict__ d,
                         const int* __restrict__ start, const float* __restrict__ near, const float* __restrict__ mid,
                         const float* __restrict__ far, int64_t B, float* __restrict__ rgb, float* __restrict__ T_out,
                         float* __restrict__ depth, int* __restrict__ stats) {
    const int64_t r = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (r >= B) return;
    Ray R;
    load_ray(r, o, d, near, mid, far, M, R);
    float b[16];
    sh_basis(R.d[0], R.d[1], R.d[2], F.sh_deg, b);
    int cur = start[r], n_samples = 0, n_color = 0, walk_steps = 0;
    float T = 1.f, acc[3] = {0.f, 0.f, 0.f}, dep = 0.f;
    for (int k = 0; k < R.n_total; ++k) {
        float t, delta, x[3];
        sample_at(R, M, k, t, delta);
#pragma unroll
        for (int a = 0; a < 3; ++a) x[a] = R.o[a] + t * R.d[a];
        cur = walk_to(g, cur, x, lane, walk_steps);
        const float alpha = 1.f - __expf(-density_at(g, F, cur, x, lane) * delta);
        const float w = T * alpha;
        ++n_samples;
        if (w > M.w_thresh) {
            float C[3], mmin, inv_a;
            color_at(g, F, cur, x, b, lane, C, mmin, inv_a);
#pragma unroll
            for (int a = 0; a < 3; ++a) acc[a] += w * C[a];
            dep += w * t;
            ++n_color;
        }
        T *= 1.f - alpha;
        if (T < M.t_thresh) break;
    }
    if (lane == 0) {
#pragma unroll
        for (int a = 0; a < 3; ++a) rgb[3 * r + a] = acc[a];
        T_out[r] = T;
        depth[r] = dep;
        stats[3 * r] = n_samples, stats[3 * r + 1] = n_color, stats[3 * r + 2] = walk_steps;
    }
}

__global__ void __launch_bounds__(kBlock)
    march_backward_kernel(Graph g, Field F, MarchCfg M, const float* __restrict__ o, const float* __restrict__ d,
                          const int* __restrict__ start, const float* __restrict__ near,
                          const float* __restrict__ mid, const float* __restrict__ far, int64_t B,
                          const float* __restrict__ rgb, const float* __restrict__ T_out,
                          const float* __restrict__ g_rgb, const float* __restrict__ g_T, float* __restrict__ g_raw,
                          float* __restrict__ g_dA, float* __restrict__ g_cA, float* __restrict__ g_sh) {
    const int64_t r = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (r >= B) return;
    Ray R;
    load_ray(r, o, d, near, mid, far, M, R);
    float b[16];
    sh_basis(R.d[0], R.d[1], R.d[2], F.sh_deg, b);
    const int nb = (F.sh_deg + 1) * (F.sh_deg + 1);
    const float gR[3] = {g_rgb[3 * r], g_rgb[3 * r + 1], g_rgb[3 * r + 2]};
    const float fg[3] = {rgb[3 * r], rgb[3 * r + 1], rgb[3 * r + 2]};
    const float gT_final = g_T[r] * T_out[r];
    int cur = start[r], walk_steps = 0;
    float T = 1.f, acc[3] = {0.f, 0.f, 0.f};
    for (int k = 0; k < R.n_total; ++k) {
        float t, delta, x[3];
        sample_at(R, M, k, t, delta);
#pragma unroll
        for (int a = 0; a < 3; ++a) x[a] = R.o[a] + t * R.d[a];
        cur = walk_to(g, cur, x, lane, walk_steps);
        const float alpha = 1.f - __expf(-density_at(g, F, cur, x, lane) * delta);
        const float w = T * alpha;
        const float T_next = T * (1.f - alpha);
        float C[3] = {0.f, 0.f, 0.f}, mmin = 0.f, inv_a = 0.f;
        const bool active = w > M.w_thresh;
        if (active) {
            color_at(g, F, cur, x, b, lane, C, mmin, inv_a);
#pragma unroll
            for (int a = 0; a < 3; ++a) acc[a] += w * C[a];
        }
        float gsig = 0.f;
#pragma unroll
        for (int a = 0; a < 3; ++a) gsig += gR[a] * (T_next * C[a] - (fg[a] - acc[a]));
        gsig = delta * (gsig - gT_final);

        const int e0 = __ldg(g.offsets + cur), n = __ldg(g.offsets + cur + 1) - e0 + 1;
        if (gsig != 0.f) {
            for (int e = lane; e < n; e += 32) {
                const int q = neighbor(g, cur, e0, e);
                float v[3];
                offset_to(g.points, q, x, v);
                const float gc = gsig * __expf(fminf(__ldg(F.raw + q), 30.f) - 0.5f * mahalanobis(F.dA, q, v));
                if (fabsf(gc) > 1e-12f) {
                    atomicAdd(g_raw + q, gc);
                    scatter_conic(g_dA, q, -0.5f * gc, v);
                }
            }
        }
        if (active) {
            const float gC[3] = {w * gR[0], w * gR[1], w * gR[2]};
            const float gCC = gC[0] * C[0] + gC[1] * C[1] + gC[2] * C[2];
            for (int e = lane; e < n; e += 32) {
                const int q = neighbor(g, cur, e0, e);
                float v[3];
                offset_to(g.points, q, x, v);
                const float a = __expf(-0.5f * (mahalanobis(F.cA, q, v) - mmin)) * inv_a;
                if (a < kMinWeight) continue;
                float c[3];
                sh_eval(F.sh, q, b, nb, c);
                const float dc0 = c[0] > 0.f ? a * gC[0] : 0.f, dc1 = c[1] > 0.f ? a * gC[1] : 0.f,
                            dc2 = c[2] > 0.f ? a * gC[2] : 0.f;
                float* gs = g_sh + 48 * static_cast<int64_t>(q);
                for (int j = 0; j < nb; ++j) {
                    if (dc0 != 0.f) atomicAdd(gs + 3 * j, b[j] * dc0);
                    if (dc1 != 0.f) atomicAdd(gs + 3 * j + 1, b[j] * dc1);
                    if (dc2 != 0.f) atomicAdd(gs + 3 * j + 2, b[j] * dc2);
                }
                const float ga = gC[0] * fmaxf(c[0], 0.f) + gC[1] * fmaxf(c[1], 0.f) + gC[2] * fmaxf(c[2], 0.f);
                const float gm = -0.5f * a * (ga - gCC);
                if (gm != 0.f) scatter_conic(g_cA, q, gm, v);
            }
        }
        T = T_next;
        if (T < M.t_thresh) break;
    }
}

struct Inputs {
    Graph g;
    Field F;
    MarchCfg M;
};

Inputs make_inputs(const torch::Tensor& points, const torch::Tensor& offsets, const torch::Tensor& adj,
                   const torch::Tensor& raw, const torch::Tensor& dA, const torch::Tensor& cA, const torch::Tensor& sh,
                   int64_t sh_deg, const torch::Tensor& o, const torch::Tensor& d, const torch::Tensor& start,
                   const torch::Tensor& near, const torch::Tensor& mid, const torch::Tensor& far, double dt,
                   int64_t n_bg, double t_thresh, double w_thresh) {
    CHECK_INPUT(points, torch::kFloat32);
    CHECK_INPUT(offsets, torch::kInt32);
    CHECK_INPUT(adj, torch::kInt32);
    CHECK_INPUT(raw, torch::kFloat32);
    CHECK_INPUT(dA, torch::kFloat32);
    CHECK_INPUT(cA, torch::kFloat32);
    CHECK_INPUT(sh, torch::kFloat32);
    CHECK_INPUT(o, torch::kFloat32);
    CHECK_INPUT(d, torch::kFloat32);
    CHECK_INPUT(start, torch::kInt32);
    CHECK_INPUT(near, torch::kFloat32);
    CHECK_INPUT(mid, torch::kFloat32);
    CHECK_INPUT(far, torch::kFloat32);
    TORCH_CHECK(sh.dim() == 3 && sh.size(1) == 16 && sh.size(2) == 3, "sh must be (N, 16, 3)");
    TORCH_CHECK(dA.size(1) == 6 && cA.size(1) == 6, "conics must be (N, 6)");
    return {{points.data_ptr<float>(), offsets.data_ptr<int>(), adj.data_ptr<int>()},
            {raw.data_ptr<float>(), dA.data_ptr<float>(), cA.data_ptr<float>(), sh.data_ptr<float>(),
             static_cast<int>(sh_deg)},
            {static_cast<float>(dt), static_cast<int>(n_bg), static_cast<float>(t_thresh), static_cast<float>(w_thresh)}};
}

inline int warp_blocks(int64_t B) { return static_cast<int>((32 * B + kBlock - 1) / kBlock); }

}  // namespace

std::vector<torch::Tensor> march_forward(torch::Tensor points, torch::Tensor offsets, torch::Tensor adj,
                                         torch::Tensor raw, torch::Tensor dA, torch::Tensor cA, torch::Tensor sh,
                                         int64_t sh_deg, torch::Tensor o, torch::Tensor d, torch::Tensor start,
                                         torch::Tensor near, torch::Tensor mid, torch::Tensor far, double dt,
                                         int64_t n_bg, double t_thresh, double w_thresh) {
    const Inputs in = make_inputs(points, offsets, adj, raw, dA, cA, sh, sh_deg, o, d, start, near, mid, far, dt, n_bg,
                                  t_thresh, w_thresh);
    const at::cuda::CUDAGuard guard(o.device());
    const int64_t B = o.size(0);
    auto rgb = torch::empty({B, 3}, o.options());
    auto T = torch::empty({B}, o.options());
    auto depth = torch::empty({B}, o.options());
    auto stats = torch::empty({B, 3}, o.options().dtype(torch::kInt32));
    if (B == 0) return {rgb, T, depth, stats};
    march_forward_kernel<<<warp_blocks(B), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        in.g, in.F, in.M, o.data_ptr<float>(), d.data_ptr<float>(), start.data_ptr<int>(), near.data_ptr<float>(),
        mid.data_ptr<float>(), far.data_ptr<float>(), B, rgb.data_ptr<float>(), T.data_ptr<float>(),
        depth.data_ptr<float>(), stats.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rgb, T, depth, stats};
}

std::vector<torch::Tensor> march_backward(torch::Tensor points, torch::Tensor offsets, torch::Tensor adj,
                                          torch::Tensor raw, torch::Tensor dA, torch::Tensor cA, torch::Tensor sh,
                                          int64_t sh_deg, torch::Tensor o, torch::Tensor d, torch::Tensor start,
                                          torch::Tensor near, torch::Tensor mid, torch::Tensor far, double dt,
                                          int64_t n_bg, double t_thresh, double w_thresh, torch::Tensor rgb,
                                          torch::Tensor T, torch::Tensor g_rgb, torch::Tensor g_T) {
    const Inputs in = make_inputs(points, offsets, adj, raw, dA, cA, sh, sh_deg, o, d, start, near, mid, far, dt, n_bg,
                                  t_thresh, w_thresh);
    CHECK_INPUT(rgb, torch::kFloat32);
    CHECK_INPUT(T, torch::kFloat32);
    CHECK_INPUT(g_rgb, torch::kFloat32);
    CHECK_INPUT(g_T, torch::kFloat32);
    const at::cuda::CUDAGuard guard(o.device());
    const int64_t B = o.size(0);
    auto g_raw = torch::zeros_like(raw), g_dA = torch::zeros_like(dA), g_cA = torch::zeros_like(cA),
         g_sh = torch::zeros_like(sh);
    if (B == 0) return {g_raw, g_dA, g_cA, g_sh};
    march_backward_kernel<<<warp_blocks(B), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        in.g, in.F, in.M, o.data_ptr<float>(), d.data_ptr<float>(), start.data_ptr<int>(), near.data_ptr<float>(),
        mid.data_ptr<float>(), far.data_ptr<float>(), B, rgb.data_ptr<float>(), T.data_ptr<float>(),
        g_rgb.data_ptr<float>(), g_T.data_ptr<float>(), g_raw.data_ptr<float>(), g_dA.data_ptr<float>(),
        g_cA.data_ptr<float>(), g_sh.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_raw, g_dA, g_cA, g_sh};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("march_forward", &march_forward, "Fused uniform-step ray marching (forward)");
    m.def("march_backward", &march_backward, "Fused uniform-step ray marching (backward, re-marches each ray)");
}
