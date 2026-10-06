// Voronoi-Gaussian field rendered at the ray's interaction points with the Voronoi diagram of the sites.
//
// One warp per ray; the 32 lanes split the Voronoi neighborhood {i} U adj(i) of the current cell. The ray
// starts at t = near in the cell of x(near) (greedy walk from the camera's site) and is traced cell by cell:
// the exit of cell i is the nearest bisector plane between p_i and a Delaunay neighbor p_j ahead of the ray,
//   t_j = ((p_i + p_j) / 2 - o) . (p_j - p_i) / (d . (p_j - p_i)),   d . (p_j - p_i) > 0,
// and the ray continues in cell j. The samples are x(near), every face crossing, and x(far) (or the last
// crossing before T < t_thresh). Each sample is evaluated once, with the neighborhood of the cell it enters:
//   sigma(x) = sum_k rho_k G_k(x),   C(x, d) = sum_k H_k(x) c_k(d) / sum_k H_k(x).
// Consecutive samples bound a segment inside one cell, composited with the trapezoid rule:
//   alpha_s = 1 - exp(-(sigma_a + sigma_b) / 2 * delta_s),   c_s = (C_a + C_b) / 2,
// colors only being evaluated for segments with weight > w_thresh.
//
// Gaussians use precomputed conics A = Sigma^-1 stored as (xx, xy, xz, yy, yz, zz). The training forward pass
// records up to K samples per ray; the backward pass replays them (or re-traces longer rays with identical
// arithmetic). A sample's gradient collects half of the segment gradients on both sides,  dC_fg/dsigma_bar_s = delta_s (T_{s+1} c_s - (C_fg - C_fg,<=s)),  dT/dsigma_bar_s = -delta_s T.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>

namespace {

constexpr float kInf = 3.402823466e38f;
constexpr int kBlock = 128;
constexpr int kWalkCap = 4096;
constexpr int kMaxCells = 8192;  // safety bound on cells traversed per ray
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

struct Cfg {
    float t_thresh;
    float w_thresh;
};

struct Ray {
    float o[3], d[3];
    float near, far;
};

// One sample: position, the cell whose neighborhood evaluates it, density and (lazily) color.
struct Sample {
    float t, x[3];
    int site;
    float sigma;
    bool has_c;
    float C[3], mmin, inv_a;
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

// Warp argmin of (value, index), ties broken by the smaller index.
__device__ inline void warp_argmin(float& v, int& k) {
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        const float ov = __shfl_xor_sync(kFull, v, s);
        const int ok = __shfl_xor_sync(kFull, k, s);
        if (ov < v || (ov == v && ok < k)) v = ov, k = ok;
    }
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

__device__ inline void load_ray(int64_t r, const float* o, const float* d, const float* near, const float* far,
                                Ray& R) {
#pragma unroll
    for (int a = 0; a < 3; ++a) R.o[a] = o[3 * r + a], R.d[a] = d[3 * r + a];
    R.near = near[r], R.far = far[r];
}

// Warp-cooperative greedy walk to the nearest site of x (ties broken by the smaller index).
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
        warp_argmin(bd, bk);
        if (bk == cur) break;
        cur = bk;
        ++steps;
    }
    return cur;
}

// Exit of cell i along the ray: smallest bisector-plane parameter ahead of the ray, and the cell beyond it
// (INT_MAX when the cell is unbounded in the ray direction).
__device__ inline void exit_face(const Graph& g, int i, const Ray& R, int lane, float& t_exit, int& next) {
    const int64_t i3 = 3 * static_cast<int64_t>(i);
    const float pi[3] = {__ldg(g.points + i3), __ldg(g.points + i3 + 1), __ldg(g.points + i3 + 2)};
    const int e0 = __ldg(g.offsets + i), e1 = __ldg(g.offsets + i + 1);
    float bt = kInf;
    int bk = INT_MAX;
    for (int e = e0 + lane; e < e1; e += 32) {
        const int j = __ldg(g.adj + e);
        const int64_t j3 = 3 * static_cast<int64_t>(j);
        float n[3], m[3];
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            const float pj = __ldg(g.points + j3 + a);
            n[a] = pj - pi[a];
            m[a] = 0.5f * (pj + pi[a]) - R.o[a];
        }
        const float dn = n[0] * R.d[0] + n[1] * R.d[1] + n[2] * R.d[2];
        if (dn > 0.f) {
            const float t = (m[0] * n[0] + m[1] * n[1] + m[2] * n[2]) / dn;
            if (t < bt || (t == bt && j < bk)) bt = t, bk = j;
        }
    }
    warp_argmin(bt, bk);
    t_exit = bt, next = bk;
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

// Normalized blend of the neighborhood colors at the sample (also keeps m_min and 1 / sum of the weights).
__device__ inline void color_at(const Graph& g, const Field& F, Sample& P, const float (&b)[16], int lane) {
    const int i = P.site;
    const int e0 = __ldg(g.offsets + i), n = __ldg(g.offsets + i + 1) - e0 + 1;
    const int nb = (F.sh_deg + 1) * (F.sh_deg + 1);
    float ml = kInf;
    for (int e = lane; e < n; e += 32) {
        const int k = neighbor(g, i, e0, e);
        float v[3];
        offset_to(g.points, k, P.x, v);
        ml = fminf(ml, mahalanobis(F.cA, k, v));
    }
    P.mmin = warp_min(ml);
    float A = 0.f, cr = 0.f, cg = 0.f, cb = 0.f;
    for (int e = lane; e < n; e += 32) {
        const int k = neighbor(g, i, e0, e);
        float v[3];
        offset_to(g.points, k, P.x, v);
        const float a = __expf(-0.5f * (mahalanobis(F.cA, k, v) - P.mmin));
        float c[3];
        sh_eval(F.sh, k, b, nb, c);
        A += a, cr += a * fmaxf(c[0], 0.f), cg += a * fmaxf(c[1], 0.f), cb += a * fmaxf(c[2], 0.f);
    }
    P.inv_a = 1.f / warp_sum(A);
    P.C[0] = warp_sum(cr) * P.inv_a, P.C[1] = warp_sum(cg) * P.inv_a, P.C[2] = warp_sum(cb) * P.inv_a;
    P.has_c = true;
}

__device__ inline void make_sample(const Graph& g, const Field& F, const Ray& R, float t, int site, int lane,
                                   Sample& P) {
    P.t = t;
#pragma unroll
    for (int a = 0; a < 3; ++a) P.x[a] = R.o[a] + t * R.d[a];
    P.site = site;
    P.sigma = density_at(g, F, site, P.x, lane);
    P.has_c = false;
}

// Next sample after P (in cell `cur`): the exit crossing, or x(far) if the cell reaches beyond far.
__device__ inline bool next_sample(const Graph& g, const Field& F, const Ray& R, const Sample& P, int cur, int lane,
                                   Sample& Q) {
    float t_exit;
    int nxt;
    exit_face(g, cur, R, lane, t_exit, nxt);
    const bool last = nxt == INT_MAX || t_exit >= R.far;
    make_sample(g, F, R, last ? R.far : fmaxf(t_exit, P.t), last ? cur : nxt, lane, Q);
    return last;
}

// Per-sample record kept by the training forward pass so that the backward pass does not re-trace the ray:
// (t, site bits, sigma, C rgb, m_min, 1 / sum of the color weights or 0 when the color was not evaluated).
constexpr int kRecord = 8;

__device__ inline void store_sample(float* __restrict__ rec, int64_t r, int K, int s, const Sample& P, int lane) {
    if (rec == nullptr || s >= K || lane >= kRecord) return;
    const float v = lane == 0 ? P.t
                  : lane == 1 ? __int_as_float(P.site)
                  : lane == 2 ? P.sigma
                  : lane == 3 ? P.C[0]
                  : lane == 4 ? P.C[1]
                  : lane == 5 ? P.C[2]
                  : lane == 6 ? P.mmin
                              : (P.has_c ? P.inv_a : 0.f);
    rec[(r * K + s) * kRecord + lane] = v;
}

__device__ inline void load_sample(const float* __restrict__ rec, int64_t r, int K, int s, const Ray& R, Sample& P) {
    const float* p = rec + (r * K + s) * kRecord;
    P.t = p[0];
#pragma unroll
    for (int a = 0; a < 3; ++a) P.x[a] = R.o[a] + P.t * R.d[a];
    P.site = __float_as_int(p[1]);
    P.sigma = p[2];
    P.C[0] = p[3], P.C[1] = p[4], P.C[2] = p[5];
    P.mmin = p[6];
    P.inv_a = p[7];
    P.has_c = P.inv_a > 0.f;
}

// Accumulates the gradients of one sample: gs = dL/dsigma, gC = dL/dC.
__device__ inline void scatter_sample(const Graph& g, const Field& F, const Sample& P, float gs, const float (&gC)[3],
                                      const float (&b)[16], int lane, float* __restrict__ g_raw,
                                      float* __restrict__ g_dA, float* __restrict__ g_cA, float* __restrict__ g_sh) {
    const int i = P.site;
    const int e0 = __ldg(g.offsets + i), n = __ldg(g.offsets + i + 1) - e0 + 1;
    if (gs != 0.f) {
        for (int e = lane; e < n; e += 32) {
            const int q = neighbor(g, i, e0, e);
            float v[3];
            offset_to(g.points, q, P.x, v);
            const float gc = gs * __expf(fminf(__ldg(F.raw + q), 30.f) - 0.5f * mahalanobis(F.dA, q, v));
            if (fabsf(gc) > 1e-12f) {
                atomicAdd(g_raw + q, gc);
                scatter_conic(g_dA, q, -0.5f * gc, v);
            }
        }
    }
    if (!P.has_c || (gC[0] == 0.f && gC[1] == 0.f && gC[2] == 0.f)) return;
    const int nb = (F.sh_deg + 1) * (F.sh_deg + 1);
    const float gCC = gC[0] * P.C[0] + gC[1] * P.C[1] + gC[2] * P.C[2];
    for (int e = lane; e < n; e += 32) {
        const int q = neighbor(g, i, e0, e);
        float v[3];
        offset_to(g.points, q, P.x, v);
        const float a = __expf(-0.5f * (mahalanobis(F.cA, q, v) - P.mmin)) * P.inv_a;
        if (a < kMinWeight) continue;
        float c[3];
        sh_eval(F.sh, q, b, nb, c);
        const float dc0 = c[0] > 0.f ? a * gC[0] : 0.f, dc1 = c[1] > 0.f ? a * gC[1] : 0.f,
                    dc2 = c[2] > 0.f ? a * gC[2] : 0.f;
        float* gsh = g_sh + 48 * static_cast<int64_t>(q);
        for (int j = 0; j < nb; ++j) {
            if (dc0 != 0.f) atomicAdd(gsh + 3 * j, b[j] * dc0);
            if (dc1 != 0.f) atomicAdd(gsh + 3 * j + 1, b[j] * dc1);
            if (dc2 != 0.f) atomicAdd(gsh + 3 * j + 2, b[j] * dc2);
        }
        const float ga = gC[0] * fmaxf(c[0], 0.f) + gC[1] * fmaxf(c[1], 0.f) + gC[2] * fmaxf(c[2], 0.f);
        const float gm = -0.5f * a * (ga - gCC);
        if (gm != 0.f) scatter_conic(g_cA, q, gm, v);
    }
}

__global__ void __launch_bounds__(kBlock)
    foam_forward_kernel(Graph g, Field F, Cfg M, const float* __restrict__ o, const float* __restrict__ d,
                        const int* __restrict__ start, const float* __restrict__ near, const float* __restrict__ far,
                        int64_t B, float* __restrict__ rgb, float* __restrict__ T_out, float* __restrict__ depth,
                        int* __restrict__ stats, float* __restrict__ rec, int K) {
    const int64_t r = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (r >= B) return;
    Ray R;
    load_ray(r, o, d, near, far, R);
    float b[16];
    sh_basis(R.d[0], R.d[1], R.d[2], F.sh_deg, b);
    int walk_steps = 0, n_samples = 1, n_color = 0;
    float x0[3];
#pragma unroll
    for (int a = 0; a < 3; ++a) x0[a] = R.o[a] + R.near * R.d[a];
    int cur = walk_to(g, start[r], x0, lane, walk_steps);
    Sample P, Q;
    make_sample(g, F, R, R.near, cur, lane, P);
    float T = 1.f, acc[3] = {0.f, 0.f, 0.f}, dep = 0.f;
    for (int it = 0; it < kMaxCells; ++it) {
        const bool last = next_sample(g, F, R, P, cur, lane, Q);
        ++n_samples;
        const float delta = Q.t - P.t;
        const float alpha = 1.f - __expf(-0.5f * (P.sigma + Q.sigma) * delta);
        const float w = T * alpha;
        if (w > M.w_thresh) {
            if (!P.has_c) color_at(g, F, P, b, lane);
            color_at(g, F, Q, b, lane);
#pragma unroll
            for (int a = 0; a < 3; ++a) acc[a] += w * 0.5f * (P.C[a] + Q.C[a]);
            dep += w * 0.5f * (P.t + Q.t);
            ++n_color;
        }
        T *= 1.f - alpha;
        store_sample(rec, r, K, n_samples - 2, P, lane);
        P = Q;
        cur = Q.site;
        if (last || T < M.t_thresh) break;
    }
    store_sample(rec, r, K, n_samples - 1, P, lane);
    if (lane == 0) {
#pragma unroll
        for (int a = 0; a < 3; ++a) rgb[3 * r + a] = acc[a];
        T_out[r] = T;
        depth[r] = dep;
        stats[3 * r] = n_samples, stats[3 * r + 1] = n_color, stats[3 * r + 2] = walk_steps;
    }
}

__global__ void __launch_bounds__(kBlock)
    foam_backward_kernel(Graph g, Field F, Cfg M, const float* __restrict__ o, const float* __restrict__ d,
                         const int* __restrict__ start, const float* __restrict__ near, const float* __restrict__ far,
                         int64_t B, const float* __restrict__ rgb, const float* __restrict__ T_out,
                         const float* __restrict__ g_rgb, const float* __restrict__ g_T, float* __restrict__ g_raw,
                         float* __restrict__ g_dA, float* __restrict__ g_cA, float* __restrict__ g_sh,
                         const float* __restrict__ rec, int K, const int* __restrict__ stats) {
    const int64_t r = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (r >= B) return;
    Ray R;
    load_ray(r, o, d, near, far, R);
    float b[16];
    sh_basis(R.d[0], R.d[1], R.d[2], F.sh_deg, b);
    const float gR[3] = {g_rgb[3 * r], g_rgb[3 * r + 1], g_rgb[3 * r + 2]};
    const float fg[3] = {rgb[3 * r], rgb[3 * r + 1], rgb[3 * r + 2]};
    const float gT_final = g_T[r] * T_out[r];
    // replay the samples recorded by the forward pass, or re-trace the ray when they did not fit
    const int n_stored = rec != nullptr && stats[3 * r] <= K ? stats[3 * r] : 0;
    Sample P, Q;
    int cur = 0;
    if (n_stored > 0) {
        load_sample(rec, r, K, 0, R, P);
    } else {
        int walk_steps = 0;
        float x0[3];
#pragma unroll
        for (int a = 0; a < 3; ++a) x0[a] = R.o[a] + R.near * R.d[a];
        cur = walk_to(g, start[r], x0, lane, walk_steps);
        make_sample(g, F, R, R.near, cur, lane, P);
    }
    float T = 1.f, acc[3] = {0.f, 0.f, 0.f};
    float gs_pending = 0.f, gC_pending[3] = {0.f, 0.f, 0.f};  // gradients of P from the segment before it
    for (int it = 0; it < kMaxCells; ++it) {
        bool last;
        if (n_stored > 0) {
            if (it + 1 >= n_stored) break;
            load_sample(rec, r, K, it + 1, R, Q);
            last = it + 2 >= n_stored;
        } else {
            last = next_sample(g, F, R, P, cur, lane, Q);
        }
        const float delta = Q.t - P.t;
        const float alpha = 1.f - __expf(-0.5f * (P.sigma + Q.sigma) * delta);
        const float w = T * alpha;
        const float T_next = T * (1.f - alpha);
        const bool active = w > M.w_thresh;
        float c[3] = {0.f, 0.f, 0.f};
        if (active) {
            if (!P.has_c) color_at(g, F, P, b, lane);
            if (!Q.has_c) color_at(g, F, Q, b, lane);
#pragma unroll
            for (int a = 0; a < 3; ++a) {
                c[a] = 0.5f * (P.C[a] + Q.C[a]);
                acc[a] += w * c[a];
            }
        }
        float g_sbar = 0.f;
#pragma unroll
        for (int a = 0; a < 3; ++a) g_sbar += gR[a] * (T_next * c[a] - (fg[a] - acc[a]));
        g_sbar = delta * (g_sbar - gT_final);
        const float gc_half[3] = {active ? 0.5f * w * gR[0] : 0.f, active ? 0.5f * w * gR[1] : 0.f,
                                  active ? 0.5f * w * gR[2] : 0.f};
        const float gC_P[3] = {gC_pending[0] + gc_half[0], gC_pending[1] + gc_half[1], gC_pending[2] + gc_half[2]};
        scatter_sample(g, F, P, gs_pending + 0.5f * g_sbar, gC_P, b, lane, g_raw, g_dA, g_cA, g_sh);
        gs_pending = 0.5f * g_sbar;
#pragma unroll
        for (int a = 0; a < 3; ++a) gC_pending[a] = gc_half[a];
        T = T_next;
        P = Q;
        cur = Q.site;
        if (last || T < M.t_thresh) break;
    }
    scatter_sample(g, F, P, gs_pending, gC_pending, b, lane, g_raw, g_dA, g_cA, g_sh);
}

struct Inputs {
    Graph g;
    Field F;
    Cfg M;
};

Inputs make_inputs(const torch::Tensor& points, const torch::Tensor& offsets, const torch::Tensor& adj,
                   const torch::Tensor& raw, const torch::Tensor& dA, const torch::Tensor& cA, const torch::Tensor& sh,
                   int64_t sh_deg, const torch::Tensor& o, const torch::Tensor& d, const torch::Tensor& start,
                   const torch::Tensor& near, const torch::Tensor& far, double t_thresh, double w_thresh) {
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
    CHECK_INPUT(far, torch::kFloat32);
    TORCH_CHECK(sh.dim() == 3 && sh.size(1) == 16 && sh.size(2) == 3, "sh must be (N, 16, 3)");
    TORCH_CHECK(dA.size(1) == 6 && cA.size(1) == 6, "conics must be (N, 6)");
    return {{points.data_ptr<float>(), offsets.data_ptr<int>(), adj.data_ptr<int>()},
            {raw.data_ptr<float>(), dA.data_ptr<float>(), cA.data_ptr<float>(), sh.data_ptr<float>(),
             static_cast<int>(sh_deg)},
            {static_cast<float>(t_thresh), static_cast<float>(w_thresh)}};
}

inline int warp_blocks(int64_t B) { return static_cast<int>((32 * B + kBlock - 1) / kBlock); }

}  // namespace

std::vector<torch::Tensor> foam_forward(torch::Tensor points, torch::Tensor offsets, torch::Tensor adj,
                                        torch::Tensor raw, torch::Tensor dA, torch::Tensor cA, torch::Tensor sh,
                                        int64_t sh_deg, torch::Tensor o, torch::Tensor d, torch::Tensor start,
                                        torch::Tensor near, torch::Tensor far, double t_thresh, double w_thresh,
                                        int64_t record_cap) {
    const Inputs in =
        make_inputs(points, offsets, adj, raw, dA, cA, sh, sh_deg, o, d, start, near, far, t_thresh, w_thresh);
    const at::cuda::CUDAGuard guard(o.device());
    const int64_t B = o.size(0);
    auto rgb = torch::empty({B, 3}, o.options());
    auto T = torch::empty({B}, o.options());
    auto depth = torch::empty({B}, o.options());
    auto stats = torch::empty({B, 3}, o.options().dtype(torch::kInt32));
    auto rec = torch::empty({record_cap > 0 ? B : 0, record_cap, kRecord}, o.options());
    if (B == 0) return {rgb, T, depth, stats, rec};
    foam_forward_kernel<<<warp_blocks(B), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        in.g, in.F, in.M, o.data_ptr<float>(), d.data_ptr<float>(), start.data_ptr<int>(), near.data_ptr<float>(),
        far.data_ptr<float>(), B, rgb.data_ptr<float>(), T.data_ptr<float>(), depth.data_ptr<float>(),
        stats.data_ptr<int>(), record_cap > 0 ? rec.data_ptr<float>() : nullptr, static_cast<int>(record_cap));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rgb, T, depth, stats, rec};
}

std::vector<torch::Tensor> foam_backward(torch::Tensor points, torch::Tensor offsets, torch::Tensor adj,
                                         torch::Tensor raw, torch::Tensor dA, torch::Tensor cA, torch::Tensor sh,
                                         int64_t sh_deg, torch::Tensor o, torch::Tensor d, torch::Tensor start,
                                         torch::Tensor near, torch::Tensor far, double t_thresh, double w_thresh,
                                         torch::Tensor rgb, torch::Tensor T, torch::Tensor g_rgb, torch::Tensor g_T,
                                         torch::Tensor rec, torch::Tensor stats) {
    const Inputs in =
        make_inputs(points, offsets, adj, raw, dA, cA, sh, sh_deg, o, d, start, near, far, t_thresh, w_thresh);
    CHECK_INPUT(rgb, torch::kFloat32);
    CHECK_INPUT(T, torch::kFloat32);
    CHECK_INPUT(g_rgb, torch::kFloat32);
    CHECK_INPUT(g_T, torch::kFloat32);
    CHECK_INPUT(rec, torch::kFloat32);
    CHECK_INPUT(stats, torch::kInt32);
    const bool replay = rec.numel() > 0;
    TORCH_CHECK(!replay || (rec.dim() == 3 && rec.size(0) == o.size(0) && rec.size(2) == kRecord),
                "rec must be (B, K, 8)");
    const at::cuda::CUDAGuard guard(o.device());
    const int64_t B = o.size(0);
    auto g_raw = torch::zeros_like(raw), g_dA = torch::zeros_like(dA), g_cA = torch::zeros_like(cA),
         g_sh = torch::zeros_like(sh);
    if (B == 0) return {g_raw, g_dA, g_cA, g_sh};
    foam_backward_kernel<<<warp_blocks(B), kBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        in.g, in.F, in.M, o.data_ptr<float>(), d.data_ptr<float>(), start.data_ptr<int>(), near.data_ptr<float>(),
        far.data_ptr<float>(), B, rgb.data_ptr<float>(), T.data_ptr<float>(), g_rgb.data_ptr<float>(),
        g_T.data_ptr<float>(), g_raw.data_ptr<float>(), g_dA.data_ptr<float>(), g_cA.data_ptr<float>(),
        g_sh.data_ptr<float>(), replay ? rec.data_ptr<float>() : nullptr, replay ? static_cast<int>(rec.size(1)) : 0,
        stats.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {g_raw, g_dA, g_cA, g_sh};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("foam_forward", &foam_forward,
          "Rendering at the ray / Voronoi-face interaction points (forward; keeps up to record_cap samples per ray)");
    m.def("foam_backward", &foam_backward,
          "Rendering at the interaction points (backward; replays the recorded samples or re-traces the ray)");
}
