// Inference-only renderer of the Voronoi-Gaussian field at the ray / Voronoi-face interaction points (same
// samples, quadrature and early termination as foam.cu), organized for memory traffic instead of gradients:
//
//  * one thread per ray: neighboring pixels of a warp walk through the same cells, so their gathers hit the
//    same cache lines (foam.cu spreads one ray over a warp, every lane reading a different site);
//  * the neighborhood {c} U adj(c) of a cell is stored contiguously, self first, one float4 per edge:
//    (p_j, half2(r2_j, lam_j)), with r2 (rounded up) the squared radius beyond which the density Gaussian
//    of site j is below eps_density (inf: never skipped) and lam (rounded down) the smallest eigenvalue of its
//    color conic, so that |v|^2 lam bounds the color Mahalanobis distance from below. Exit faces and these
//    bounds read ~4 consecutive cache lines per cell instead of one random line per neighbor;
//  * one pass over the neighborhood of the cell c a sample enters gives the exit face of c, the density at the
//    sample and its color; the per-site records attr (5 float4, below) and SH are only gathered for the
//    neighbors within sqrt(r2) (density) or not excluded by the bound (color);
//  * colors, normalized blends of max(SH_k(d) + 1/2, 0) over the neighbors whose normalized weight
//    exp(-(m - m_min) / 2) exceeds eps_color (eps_color = 0: all of them), SH in fp32 or fp16:
//      lazy  - the smallest color Mahalanobis distance m_min is found in the density pass, the SH read in a
//              second pass, for the segments with weight > w_thresh only (divergent within a warp);
//      fused - blended in the density pass of every sample, online (running m_min), no second pass;
//      baked - per-view site colors with the direction camera center -> site (as in 3D Gaussian splatting).
//    Exact site colors are kept in a small per-ray cache (a site belongs to the neighborhoods of several
//    consecutive cells of a ray, and its color only depends on the ray direction).
//
// attr (N, 20): dA xx xy xz yy | dA yz zz, min(raw, 30), 0 | cA xx xy xz yy | cA yz zz, 0, 0 | baked r g b, 0.
// STAGE (ablation): 0 = exit faces only, 1 = + density, 2 = full rendering. `cap` (optional, samples per ray)
// keeps the traversal of stages 0 and 1 equal to the one of the full renderer.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <climits>
#include <cstdint>

namespace {

constexpr float kInf = 3.402823466e38f;
constexpr int kBlock = 64;
constexpr int kWalkCap = 4096;
constexpr int kMaxCells = 8192;
constexpr int kAttr = 5;  // float4 per site
constexpr int kLazy = 0, kBaked = 1, kFused = 2;
constexpr int kCache = 8;  // per-ray direct-mapped cache of site colors (shared memory)

#define CHECK_INPUT(x, dtype)                                                                              \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                                               \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");                                            \
    TORCH_CHECK((x).scalar_type() == (dtype), #x " has the wrong dtype")

struct Net {
    const float4* __restrict__ points;  // (N) p, unused
    const int* __restrict__ offsets;    // (N + 1) into the self-first neighborhoods
    const int* __restrict__ nbr;        // site of every edge (the first edge of a cell is the cell itself)
    const float4* __restrict__ edge;    // p_j, half2(r2_j, lam_j)
    const float4* __restrict__ attr;
    const float4* __restrict__ sh;  // (N, 12) fp32 or (N, 6) fp16 (8 halves per float4)
    const int* __restrict__ orig;   // original index of every site (ties are broken as before a renumbering)
};

// k precedes j in the tie-breaking order (the smaller original index).
__device__ __forceinline__ bool before(const Net& N, int k, int j) {
    return N.orig != nullptr ? __ldg(N.orig + k) < __ldg(N.orig + j) : k < j;
}

struct Opt {
    float t_thresh, w_thresh, color_cut;  // color_cut = 2 ln(1 / eps_color) (inf when eps_color = 0)
    int sh_deg;
};

struct Blend {
    float m, a, r, g, b;
};

__device__ __forceinline__ float maha(float xx, float xy, float xz, float yy, float yz, float zz, float vx,
                                      float vy, float vz) {
    return xx * vx * vx + yy * vy * vy + zz * vz * vz + 2.f * (xy * vx * vy + xz * vx * vz + yz * vy * vz);
}

__device__ __forceinline__ void blend_init(Blend& B) { B.m = kInf, B.a = B.r = B.g = B.b = 0.f; }

__device__ __forceinline__ void blend_add(Blend& B, float m, float r, float g, float b) {
    if (m < B.m) {
        const float s = __expf(-0.5f * (B.m - m));
        B.a *= s, B.r *= s, B.g *= s, B.b *= s, B.m = m;
    }
    const float w = __expf(-0.5f * (m - B.m));
    B.a += w, B.r += w * r, B.g += w * g, B.b += w * b;
}

__device__ __forceinline__ float2 cutoffs(const float4& e) {
    const float w = e.w;
    return __half22float2(*reinterpret_cast<const __half2*>(&w));
}

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

// max(SH_k(d) + 1/2, 0) with the basis b of the ray direction.
template <bool HALF>
__device__ __forceinline__ void site_color(const Net& N, int k, const float (&b)[16], float (&c)[3]) {
    c[0] = c[1] = c[2] = 0.5f;
    if (HALF) {
        const float4* s = N.sh + 6 * static_cast<int64_t>(k);
#pragma unroll
        for (int q = 0; q < 6; ++q) {
            const float4 f = __ldg(s + q);
            const __half2* h = reinterpret_cast<const __half2*>(&f);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 p = __half22float2(h[u]);
                const int i0 = 8 * q + 2 * u, i1 = i0 + 1;
                c[i0 % 3] += b[i0 / 3] * p.x;
                c[i1 % 3] += b[i1 / 3] * p.y;
            }
        }
    } else {
        const float4* s = N.sh + 12 * static_cast<int64_t>(k);
#pragma unroll
        for (int q = 0; q < 12; ++q) {
            const float4 f = __ldg(s + q);
            const float fv[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
            for (int u = 0; u < 4; ++u) c[(4 * q + u) % 3] += b[(4 * q + u) / 3] * fv[u];
        }
    }
    c[0] = fmaxf(c[0], 0.f), c[1] = fmaxf(c[1], 0.f), c[2] = fmaxf(c[2], 0.f);
}

// Per-ray cache of max(SH_k(d) + 1/2, 0): slot k mod kCache of the thread's column (r, g, b, k).
struct ColorCache {
    float4* col;  // &cache[0][threadIdx.x], slots kBlock apart
};

template <bool HALF>
__device__ __forceinline__ void cached_color(const Net& N, ColorCache& C, int k, const float (&b)[16],
                                             float (&c)[3], int& n_sh) {
    float4& e = C.col[(k & (kCache - 1)) * kBlock];
    const float4 v = e;
    if (__float_as_int(v.w) == k) {
        c[0] = v.x, c[1] = v.y, c[2] = v.z;
        return;
    }
    site_color<HALF>(N, k, b, c);
    e = make_float4(c[0], c[1], c[2], __int_as_float(k));
    ++n_sh;
}

__device__ __forceinline__ float3 site_point(const Net& N, int c) {
    const float4 p = __ldg(N.points + c);
    return make_float3(p.x, p.y, p.z);
}

// Greedy walk to the nearest site of x (ties broken by the smaller index), as in foam.cu.
__device__ inline int walk_to(const Net& N, int c, float x0, float x1, float x2, int& steps) {
    for (int it = 0; it < kWalkCap; ++it) {
        const float3 pc = site_point(N, c);
        float bd = (x0 - pc.x) * (x0 - pc.x) + (x1 - pc.y) * (x1 - pc.y) + (x2 - pc.z) * (x2 - pc.z);
        int bk = c;
        const int e0 = __ldg(N.offsets + c), e1 = __ldg(N.offsets + c + 1);
        for (int e = e0 + 1; e < e1; ++e) {
            const float4 ed = __ldg(N.edge + e);
            const int k = __ldg(N.nbr + e);
            const float v0 = x0 - ed.x, v1 = x1 - ed.y, v2 = x2 - ed.z;
            const float dk = v0 * v0 + v1 * v1 + v2 * v2;
            if (dk < bd || (dk == bd && before(N, k, bk))) bd = dk, bk = k;
        }
        if (bk == c) break;
        c = bk;
        ++steps;
    }
    return c;
}

// One pass over the neighborhood of cell c for the sample at o + t d: density; smallest color Mahalanobis
// distance (lazy) or color blend (fused, baked); when want_exit, the exit face of c along the ray (next =
// INT_MAX when the cell is unbounded ahead).
template <int STAGE, int COLOR, bool HALF>
__device__ inline void eval_cell(const Net& N, const Opt& M, int c, float t, const float (&o)[3],
                                 const float (&d)[3], const float (&b)[16], ColorCache& cc, bool want_exit,
                                 float& sigma, float& mmin, Blend& B, float& t_exit, int& next, int& n_dens,
                                 int& n_sh) {
    constexpr bool BLEND = COLOR != kLazy;
    const float3 pc = site_point(N, c);
    const float x0 = o[0] + t * d[0], x1 = o[1] + t * d[1], x2 = o[2] + t * d[2];
    const int e0 = __ldg(N.offsets + c), e1 = __ldg(N.offsets + c + 1);
    sigma = 0.f, mmin = kInf;
    blend_init(B);
    t_exit = kInf, next = INT_MAX;
    for (int e = e0; e < e1; ++e) {
        const float4 ed = __ldg(N.edge + e);
        const int k = __ldg(N.nbr + e);
        if (want_exit && e > e0) {
            const float n0 = ed.x - pc.x, n1 = ed.y - pc.y, n2 = ed.z - pc.z;
            const float dn = n0 * d[0] + n1 * d[1] + n2 * d[2];
            if (dn > 0.f) {
                const float m0 = 0.5f * (ed.x + pc.x) - o[0], m1 = 0.5f * (ed.y + pc.y) - o[1],
                            m2 = 0.5f * (ed.z + pc.z) - o[2];
                const float tt = (m0 * n0 + m1 * n1 + m2 * n2) / dn;
                if (tt < t_exit || (tt == t_exit && (next == INT_MAX || before(N, k, next)))) t_exit = tt, next = k;
            }
        }
        if (STAGE >= 1) {
            const float v0 = x0 - ed.x, v1 = x1 - ed.y, v2 = x2 - ed.z;
            const float r2 = v0 * v0 + v1 * v1 + v2 * v2;
            const float2 cut = cutoffs(ed);
            const float4* a = N.attr + kAttr * static_cast<int64_t>(k);
            if (r2 <= cut.x) {
                const float4 q0 = __ldg(a), q1 = __ldg(a + 1);
                sigma += __expf(q1.z - 0.5f * maha(q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, v0, v1, v2));
                ++n_dens;
            }
            if (STAGE >= 2) {
                const float lb = r2 * cut.y;
                if (BLEND ? lb <= B.m + M.color_cut : lb < mmin) {
                    const float4 q2 = __ldg(a + 2), q3 = __ldg(a + 3);
                    const float m = maha(q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, v0, v1, v2);
                    if (BLEND) {
                        if (m <= B.m + M.color_cut) {
                            float c3[3];
                            if (COLOR == kBaked) {
                                const float4 q4 = __ldg(a + 4);
                                c3[0] = q4.x, c3[1] = q4.y, c3[2] = q4.z;
                            } else {
                                cached_color<HALF>(N, cc, k, b, c3, n_sh);
                            }
                            blend_add(B, m, c3[0], c3[1], c3[2]);
                        }
                    } else {
                        mmin = fminf(mmin, m);
                    }
                }
            }
        }
    }
}

// Lazy color of the sample at o + t d in cell c, given the smallest color Mahalanobis distance m_min.
template <bool HALF>
__device__ inline void color_lazy(const Net& N, const Opt& M, int c, float t, float mmin, const float (&o)[3],
                                  const float (&d)[3], const float (&b)[16], ColorCache& cc, float (&C)[3],
                                  int& n_sh) {
    const float x0 = o[0] + t * d[0], x1 = o[1] + t * d[1], x2 = o[2] + t * d[2];
    const int e0 = __ldg(N.offsets + c), e1 = __ldg(N.offsets + c + 1);
    const float cut = mmin + M.color_cut;
    float A = 0.f, r = 0.f, g = 0.f, bl = 0.f;
    for (int e = e0; e < e1; ++e) {
        const float4 ed = __ldg(N.edge + e);
        const float v0 = x0 - ed.x, v1 = x1 - ed.y, v2 = x2 - ed.z;
        if ((v0 * v0 + v1 * v1 + v2 * v2) * cutoffs(ed).y > cut) continue;
        const int k = __ldg(N.nbr + e);
        const float4* a = N.attr + kAttr * static_cast<int64_t>(k);
        const float4 q2 = __ldg(a + 2), q3 = __ldg(a + 3);
        const float m = maha(q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, v0, v1, v2);
        if (m > cut) continue;
        const float w = __expf(-0.5f * (m - mmin));
        float c3[3];
        cached_color<HALF>(N, cc, k, b, c3, n_sh);
        A += w, r += w * c3[0], g += w * c3[1], bl += w * c3[2];
    }
    const float inv = 1.f / A;
    C[0] = r * inv, C[1] = g * inv, C[2] = bl * inv;
}

template <int STAGE, int COLOR, bool HALF>
__global__ void __launch_bounds__(kBlock)
    foam_infer_kernel(Net N, Opt M, const float* __restrict__ o_, const float* __restrict__ d_,
                      const int* __restrict__ start, const float* __restrict__ near_, const float* __restrict__ far_,
                      int64_t B, const int* __restrict__ cap, float* __restrict__ rgb, float* __restrict__ T_out,
                      float* __restrict__ depth, int* __restrict__ stats) {
    constexpr bool BLEND = STAGE >= 2 && COLOR != kLazy;
    __shared__ float4 cache[kCache * kBlock];
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= B) return;
    ColorCache cc{cache + threadIdx.x};
#pragma unroll
    for (int i = 0; i < kCache; ++i) cc.col[i * kBlock] = make_float4(0.f, 0.f, 0.f, __int_as_float(-1));
    const float o[3] = {o_[3 * r], o_[3 * r + 1], o_[3 * r + 2]};
    const float d[3] = {d_[3 * r], d_[3 * r + 1], d_[3 * r + 2]};
    const float t_near = near_[r], t_far = far_[r];
    const int max_samples = cap != nullptr ? cap[r] : INT_MAX;
    float b[16];
    if (STAGE >= 2 && COLOR != kBaked) sh_basis(d[0], d[1], d[2], M.sh_deg, b);

    int walk_steps = 0, n_dens = 0, n_sh = 0;
    int c = walk_to(N, start[r], o[0] + t_near * d[0], o[1] + t_near * d[1], o[2] + t_near * d[2], walk_steps);
    float sP, mP, t_exit;
    int next;
    Blend BP;
    eval_cell<STAGE, COLOR, HALF>(N, M, c, t_near, o, d, b, cc, true, sP, mP, BP, t_exit, next, n_dens, n_sh);
    float tP = t_near, CP[3] = {0.f, 0.f, 0.f};
    bool hasP = false;
    int cP = c;
    if (BLEND) {
        const float inv = 1.f / BP.a;
        CP[0] = BP.r * inv, CP[1] = BP.g * inv, CP[2] = BP.b * inv, hasP = true;
    }
    float T = 1.f, acc[3] = {0.f, 0.f, 0.f}, dep = 0.f;
    int n_samples = 1, n_color = 0;
    for (int it = 0; it < kMaxCells; ++it) {
        const bool last = next == INT_MAX || t_exit >= t_far || n_samples + 1 >= max_samples;
        const float tQ = last ? t_far : fmaxf(t_exit, tP);
        const int cQ = last ? c : next;
        float sQ, mQ;
        Blend BQ;
        eval_cell<STAGE, COLOR, HALF>(N, M, cQ, tQ, o, d, b, cc, !last, sQ, mQ, BQ, t_exit, next, n_dens, n_sh);
        ++n_samples;
        const float delta = tQ - tP;
        const float alpha = 1.f - __expf(-0.5f * (sP + sQ) * delta);
        const float w = T * alpha;
        float CQ[3] = {0.f, 0.f, 0.f};
        bool hasQ = false;
        if (BLEND) {
            const float inv = 1.f / BQ.a;
            CQ[0] = BQ.r * inv, CQ[1] = BQ.g * inv, CQ[2] = BQ.b * inv, hasQ = true;
        }
        if (STAGE >= 2 && w > M.w_thresh) {
            if (!BLEND) {
                if (!hasP) color_lazy<HALF>(N, M, cP, tP, mP, o, d, b, cc, CP, n_sh);
                color_lazy<HALF>(N, M, cQ, tQ, mQ, o, d, b, cc, CQ, n_sh);
                hasQ = true;
            }
#pragma unroll
            for (int a = 0; a < 3; ++a) acc[a] += w * 0.5f * (CP[a] + CQ[a]);
            dep += w * 0.5f * (tP + tQ);
            ++n_color;
        }
        T *= 1.f - alpha;
        tP = tQ, sP = sQ, mP = mQ, hasP = hasQ, cP = cQ, c = cQ;
#pragma unroll
        for (int a = 0; a < 3; ++a) CP[a] = CQ[a];
        if (last || T < M.t_thresh) break;
    }
#pragma unroll
    for (int a = 0; a < 3; ++a) rgb[3 * r + a] = acc[a];
    T_out[r] = T;
    depth[r] = dep;
    int* s = stats + 5 * r;
    s[0] = n_samples, s[1] = n_color, s[2] = walk_steps, s[3] = n_dens, s[4] = n_sh;
}

template <int STAGE, int COLOR, bool HALF>
void launch(const Net& N, const Opt& M, const torch::Tensor& o, const torch::Tensor& d, const torch::Tensor& start,
            const torch::Tensor& near, const torch::Tensor& far, int64_t B, const int* cap, torch::Tensor& rgb,
            torch::Tensor& T, torch::Tensor& depth, torch::Tensor& stats) {
    foam_infer_kernel<STAGE, COLOR, HALF><<<static_cast<int>((B + kBlock - 1) / kBlock), kBlock, 0,
                                            at::cuda::getCurrentCUDAStream()>>>(
        N, M, o.data_ptr<float>(), d.data_ptr<float>(), start.data_ptr<int>(), near.data_ptr<float>(),
        far.data_ptr<float>(), B, cap, rgb.data_ptr<float>(), T.data_ptr<float>(), depth.data_ptr<float>(),
        stats.data_ptr<int>());
}

}  // namespace

std::vector<torch::Tensor> foam_infer(torch::Tensor points, torch::Tensor offsets, torch::Tensor nbr,
                                      torch::Tensor edge, torch::Tensor attr, torch::Tensor sh, int64_t sh_deg,
                                      torch::Tensor o, torch::Tensor d, torch::Tensor start, torch::Tensor near,
                                      torch::Tensor far, double t_thresh, double w_thresh, double color_cut,
                                      int64_t color_mode, int64_t stage, torch::Tensor cap, torch::Tensor orig) {
    CHECK_INPUT(points, torch::kFloat32);
    CHECK_INPUT(offsets, torch::kInt32);
    CHECK_INPUT(nbr, torch::kInt32);
    CHECK_INPUT(edge, torch::kFloat32);
    CHECK_INPUT(attr, torch::kFloat32);
    CHECK_INPUT(o, torch::kFloat32);
    CHECK_INPUT(d, torch::kFloat32);
    CHECK_INPUT(start, torch::kInt32);
    CHECK_INPUT(near, torch::kFloat32);
    CHECK_INPUT(far, torch::kFloat32);
    CHECK_INPUT(cap, torch::kInt32);
    CHECK_INPUT(orig, torch::kInt32);
    TORCH_CHECK(sh.is_cuda() && sh.is_contiguous(), "sh must be a contiguous CUDA tensor");
    const bool half = sh.scalar_type() == torch::kFloat16;
    TORCH_CHECK(half || sh.scalar_type() == torch::kFloat32, "sh must be float32 or float16");
    const int64_t n = points.size(0);
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 4, "points must be (N, 4)");
    TORCH_CHECK(offsets.numel() == n + 1, "offsets must be (N + 1)");
    TORCH_CHECK(edge.dim() == 2 && edge.size(1) == 4 && edge.size(0) == nbr.numel(), "edge must be (E, 4)");
    TORCH_CHECK(attr.dim() == 2 && attr.size(1) == 4 * kAttr && attr.size(0) == n, "attr must be (N, 20)");
    TORCH_CHECK(color_mode >= kLazy && color_mode <= kFused, "color_mode must be 0 (lazy), 1 (baked) or 2 (fused)");
    const bool baked = color_mode == kBaked;
    TORCH_CHECK(baked || (sh.dim() == 3 && sh.size(0) == n && sh.size(1) == 16 && sh.size(2) == 3),
                "sh must be (N, 16, 3)");
    TORCH_CHECK(stage >= 0 && stage <= 2, "stage must be 0, 1 or 2");
    const at::cuda::CUDAGuard guard(o.device());
    const int64_t B = o.size(0);
    auto rgb = torch::empty({B, 3}, o.options());
    auto T = torch::empty({B}, o.options());
    auto depth = torch::empty({B}, o.options());
    auto stats = torch::empty({B, 5}, o.options().dtype(torch::kInt32));
    if (B == 0) return {rgb, T, depth, stats};
    const Net N{reinterpret_cast<const float4*>(points.data_ptr<float>()), offsets.data_ptr<int>(),
                nbr.data_ptr<int>(), reinterpret_cast<const float4*>(edge.data_ptr<float>()),
                reinterpret_cast<const float4*>(attr.data_ptr<float>()),
                baked ? nullptr : reinterpret_cast<const float4*>(sh.data_ptr()),
                orig.numel() == n ? orig.data_ptr<int>() : nullptr};
    const Opt M{static_cast<float>(t_thresh), static_cast<float>(w_thresh), static_cast<float>(color_cut),
                static_cast<int>(sh_deg)};
    const int* cp = cap.numel() == B ? cap.data_ptr<int>() : nullptr;
#define LAUNCH(S, C, H) launch<S, C, H>(N, M, o, d, start, near, far, B, cp, rgb, T, depth, stats)
    if (stage == 0)
        LAUNCH(0, kLazy, false);
    else if (stage == 1)
        LAUNCH(1, kLazy, false);
    else if (baked)
        LAUNCH(2, kBaked, false);
    else if (color_mode == kLazy)
        half ? LAUNCH(2, kLazy, true) : LAUNCH(2, kLazy, false);
    else
        half ? LAUNCH(2, kFused, true) : LAUNCH(2, kFused, false);
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rgb, T, depth, stats};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("foam_infer", &foam_infer,
          "Inference rendering at the ray / Voronoi-face interaction points (thread per ray, fused passes)");
}
