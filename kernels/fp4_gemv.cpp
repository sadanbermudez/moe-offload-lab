// fp4_gemv: how fast can this CPU multiply a vector by FP4 expert weights?
//
// One MoE expert in DeepSeek V4 Flash is three 4096 x 2048 matrices stored as FP4 (E2M1: sign, 2 exponent
// bits, 1 mantissa bit) with one power-of-two scale byte (UE8M0) per block of 32 weights. At batch size 1
// each decode step is a matrix-vector product (GEMV) per expert. This benchmark measures what a GEMV over
// that format can reach on this machine, scalar and AVX-512, across thread counts, streaming over a set of
// experts larger than the caches so the numbers reflect DRAM, not L3.
//
// The weight layout here is this benchmark's own (row-major, two weights per byte, low nibble first);
// it measures the format's cost, not colibri's code.
//
// Build: clang++ -std=c++20 -O3 -march=znver5 fp4_gemv.cpp -o fp4_gemv
// Usage: fp4_gemv [--experts N] [--threads 1,2,4,8,12] [--reps R]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define HAVE_AVX512 1
#endif

using Clock = std::chrono::steady_clock;

// Reusable spinning barrier. The last thread to arrive resets the count and bumps the generation;
// the others spin until the generation they arrived in has passed. acq_rel on the arrival and acquire on
// the generation read make each thread's writes before the barrier visible to all threads after it.
// (std::barrier from this toolchain's libc++ crashed inside ntdll on Windows, so the benchmark carries its own.)
class SpinBarrier {
public:
    explicit SpinBarrier(int n) : n_(n) {}
    void arrive_and_wait() {
        unsigned gen = gen_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
            count_.store(0, std::memory_order_relaxed);
            gen_.fetch_add(1, std::memory_order_release);
        } else {
            while (gen_.load(std::memory_order_acquire) == gen) std::this_thread::yield();
        }
    }
private:
    const int n_;
    alignas(64) std::atomic<int> count_{0};          // separate cache lines: arrivals do not disturb spinners
    alignas(64) std::atomic<unsigned> gen_{0};
};

constexpr int kBlock = 32;                       // weights per scale byte
constexpr float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                             -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// One FP4 matrix: rows x cols weights, packed two per byte, plus one UE8M0 scale per 32 weights.
struct Fp4Matrix {
    int rows, cols;
    std::vector<std::uint8_t> packed;            // rows * cols / 2
    std::vector<std::uint8_t> scales;            // rows * cols / 32, value 2^(s - 127)
    std::size_t bytes() const { return packed.size() + scales.size(); }
};

Fp4Matrix random_matrix(int rows, int cols, std::mt19937& rng) {
    Fp4Matrix m{rows, cols, std::vector<std::uint8_t>(std::size_t(rows) * cols / 2),
                std::vector<std::uint8_t>(std::size_t(rows) * cols / kBlock)};
    for (auto& b : m.packed) b = std::uint8_t(rng());
    for (auto& s : m.scales) s = std::uint8_t(120 + rng() % 8);   // scales 2^-7 .. 2^0
    return m;
}

inline float ue8m0(std::uint8_t s) {
    std::uint32_t bits = std::uint32_t(s) << 23;                   // exponent field only: exactly 2^(s-127)
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Reference: y[r] = sum_c W[r][c] * x[c], rows [r0, r1).
void gemv_scalar(const Fp4Matrix& w, const float* x, float* y, int r0, int r1) {
    const int bytes_per_row = w.cols / 2, blocks_per_row = w.cols / kBlock;
    for (int r = r0; r < r1; ++r) {
        const std::uint8_t* p = &w.packed[std::size_t(r) * bytes_per_row];
        const std::uint8_t* s = &w.scales[std::size_t(r) * blocks_per_row];
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; ++b) {
            float part = 0.0f;
            for (int i = 0; i < kBlock / 2; ++i) {
                std::uint8_t v = p[b * kBlock / 2 + i];
                part += kE2M1[v & 15] * x[b * kBlock + 2 * i] + kE2M1[v >> 4] * x[b * kBlock + 2 * i + 1];
            }
            acc += part * ue8m0(s[b]);
        }
        y[r] = acc;
    }
}

#ifdef HAVE_AVX512
// AVX-512: per 32-weight block, load 16 bytes, split nibbles, look each up in a 16-entry float table with
// one permute (vpermps), and FMA against x. The low/high nibble interleave is undone by loading x
// de-interleaved once per call (xe = even elements, xo = odd elements).
void gemv_avx512(const Fp4Matrix& w, const float* xe, const float* xo, float* y, int r0, int r1) {
    const __m512 lut = _mm512_loadu_ps(kE2M1);
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const int bytes_per_row = w.cols / 2, blocks_per_row = w.cols / kBlock;
    for (int r = r0; r < r1; ++r) {
        const std::uint8_t* p = &w.packed[std::size_t(r) * bytes_per_row];
        const std::uint8_t* s = &w.scales[std::size_t(r) * blocks_per_row];
        __m512 acc = _mm512_setzero_ps();
        for (int b = 0; b < blocks_per_row; ++b) {
            __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + b * 16));
            __m512 wl = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_and_si128(v, lo_mask)), lut);
            __m512 wh = _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(v, 4), lo_mask)), lut);
            __m512 part = _mm512_mul_ps(wl, _mm512_loadu_ps(xe + b * 16));
            part = _mm512_fmadd_ps(wh, _mm512_loadu_ps(xo + b * 16), part);
            acc = _mm512_fmadd_ps(part, _mm512_set1_ps(ue8m0(s[b])), acc);
        }
        y[r] = _mm512_reduce_add_ps(acc);
    }
}
#endif

struct Point { const char* kernel; int threads; double ms_per_expert, weight_gbps, gflops; double max_err; };

// Runs reps passes over all experts (3 matrices each) with `threads` workers splitting the rows of every
// matrix; a barrier after each matrix mirrors an engine that finishes one projection before the next.
Point bench(const char* name, bool simd, const std::vector<Fp4Matrix>& mats, int threads, int reps,
            const std::vector<float>& x, const std::vector<float>& xe, const std::vector<float>& xo) {
    std::vector<std::vector<float>> ys;                       // one output per matrix, sized to its own rows
    for (const auto& m : mats) ys.emplace_back(m.rows);
    SpinBarrier sync(threads);
    auto worker = [&](int id) {
        for (int rep = 0; rep < reps; ++rep)
            for (std::size_t m = 0; m < mats.size(); ++m) {
                const auto& w = mats[m];
                int r0 = w.rows * id / threads, r1 = w.rows * (id + 1) / threads;
#ifdef HAVE_AVX512
                if (simd) gemv_avx512(w, xe.data(), xo.data(), ys[m].data(), r0, r1);
                else
#endif
                    gemv_scalar(w, x.data(), ys[m].data(), r0, r1);
                sync.arrive_and_wait();
            }
    };
    auto t0 = Clock::now();
    {
        std::vector<std::jthread> pool;
        for (int i = 0; i < threads; ++i) pool.emplace_back(worker, i);
    }
    double s = std::chrono::duration<double>(Clock::now() - t0).count();

    // Check the first matrix against the scalar reference (catches SIMD layout mistakes).
    std::vector<float> ref(mats[0].rows);
    gemv_scalar(mats[0], x.data(), ref.data(), 0, mats[0].rows);
    double max_err = 0;
    for (int r = 0; r < mats[0].rows; ++r)
        max_err = std::max(max_err, double(std::fabs(ref[r] - ys[0][r]) / (std::fabs(ref[r]) + 1e-3)));

    double bytes = 0, flops = 0;
    for (const auto& m : mats) { bytes += m.bytes(); flops += 2.0 * m.rows * m.cols; }
    double experts = mats.size() / 3.0 * reps;
    return {name, threads, 1e3 * s / experts, bytes * reps / s / 1e9, flops * reps / s / 1e9, max_err};
}

int main(int argc, char** argv) {
    int experts = 48, reps = 3;
    std::vector<int> thread_counts{1, 2, 4, 8, 12};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--experts" && i + 1 < argc) experts = std::stoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::stoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) {
            thread_counts.clear();
            std::string s = argv[++i];
            for (std::size_t p = 0; p < s.size();) {
                std::size_t q = s.find(',', p);
                thread_counts.push_back(std::stoi(s.substr(p, q - p)));
                p = q == std::string::npos ? s.size() : q + 1;
            }
        }
    }
    const int hidden = 4096, inter = 2048;
    std::mt19937 rng(42);
    std::vector<Fp4Matrix> mats;                              // per expert: gate and up (inter x hidden), down (hidden x inter)
    for (int e = 0; e < experts; ++e) {
        mats.push_back(random_matrix(inter, hidden, rng));
        mats.push_back(random_matrix(inter, hidden, rng));
        mats.push_back(random_matrix(hidden, inter, rng));
    }
    // One input vector long enough for both shapes; the down projection reads its first `inter` entries.
    std::vector<float> x(hidden), xe(hidden / 2), xo(hidden / 2);
    std::uniform_real_distribution<float> u(-1, 1);
    for (auto& v : x) v = u(rng);
    // De-interleaved copies for the SIMD kernel: within each 32-block, even and odd positions, 16 each.
    for (int b = 0; b < hidden / kBlock; ++b)
        for (int i = 0; i < 16; ++i) { xe[b * 16 + i] = x[b * 32 + 2 * i]; xo[b * 16 + i] = x[b * 32 + 2 * i + 1]; }

    double total_bytes = 0;
    for (const auto& m : mats) total_bytes += m.bytes();
    std::printf("%d experts, %.2f GB of FP4 weights (%.1f MB per expert), %d passes per point\n",
                experts, total_bytes / 1e9, total_bytes / experts / 1e6, reps);
    std::printf("kernel,threads,ms_per_expert,weight_GB_per_s,GFLOP_per_s,max_rel_err\n");
    std::vector<std::pair<const char*, bool>> kernels{{"scalar", false}};
#ifdef HAVE_AVX512
    kernels.push_back({"avx512", true});
#endif
    for (auto [name, simd] : kernels)
        for (int t : thread_counts) {
            Point p = bench(name, simd, mats, t, reps, x, xe, xo);
            std::printf("%s,%d,%.3f,%.2f,%.1f,%.1e\n", p.kernel, p.threads, p.ms_per_expert, p.weight_gbps, p.gflops, p.max_err);
            std::fflush(stdout);
        }
}
