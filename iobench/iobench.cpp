// iobench: how fast can this machine stream MoE experts from disk?
//
// Reads blocks of a chosen size at random offsets across a set of files (the model's own shards),
// from N threads at once, bypassing the OS file cache, and reports throughput and per-read latency.
// That is the access pattern of an engine that keeps most experts on disk: every decode step pulls
// a handful of multi-megabyte expert matrices from scattered places in the weight files.
//
// Windows: CreateFile with FILE_FLAG_NO_BUFFERING (offsets, sizes and buffers sector-aligned).
// Linux:   open with O_DIRECT and pread.
//
// Build: clang++ -std=c++20 -O2 iobench.cpp -o iobench   (see CMakeLists.txt)
// Usage: iobench <dir-or-file>... [--block-kib N] [--threads a,b,c] [--seconds S] [--cached]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kAlign = 4096;      // covers 512 B and 4 KiB sector drives

// One open file, readable at any aligned offset from any thread (positional reads share no file pointer).
class File {
public:
    File(const fs::path& p, bool cached) : size_(fs::file_size(p)) {
#ifdef _WIN32
        DWORD flags = FILE_ATTRIBUTE_NORMAL | (cached ? FILE_FLAG_RANDOM_ACCESS : FILE_FLAG_NO_BUFFERING);
        h_ = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + p.string());
#else
        fd_ = ::open(p.c_str(), O_RDONLY | (cached ? 0 : O_DIRECT));
        if (fd_ < 0) throw std::runtime_error("cannot open " + p.string());
#endif
    }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    File(File&& o) noexcept : size_(o.size_) {
#ifdef _WIN32
        h_ = std::exchange(o.h_, INVALID_HANDLE_VALUE);
#else
        fd_ = std::exchange(o.fd_, -1);
#endif
    }
    ~File() {
#ifdef _WIN32
        if (h_ != INVALID_HANDLE_VALUE) CloseHandle(h_);
#else
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    std::uint64_t size() const { return size_; }

    // Reads exactly n bytes at offset; false on error or short read.
    bool read_at(void* buf, std::uint64_t n, std::uint64_t offset) const {
#ifdef _WIN32
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(offset);
        ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
        DWORD got = 0;
        return ReadFile(h_, buf, static_cast<DWORD>(n), &got, &ov) && got == n;
#else
        return ::pread(fd_, buf, n, static_cast<off_t>(offset)) == static_cast<ssize_t>(n);
#endif
    }

private:
    std::uint64_t size_;
#ifdef _WIN32
    HANDLE h_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

// Sector-aligned buffer, owned (unbuffered reads need an aligned destination).
struct AlignedBuf {
    explicit AlignedBuf(std::size_t n) {
#ifdef _WIN32
        p = _aligned_malloc(n, kAlign);
#else
        p = std::aligned_alloc(kAlign, n);
#endif
        if (!p) throw std::bad_alloc();
    }
    ~AlignedBuf() {
#ifdef _WIN32
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
    void* p;
};

struct Result {
    int threads;
    double gbps;                 // decimal GB/s, as drive makers quote
    double p50_ms, p99_ms;       // per-read latency
    std::uint64_t reads, errors;
};

Result run(const std::vector<File>& files, std::uint64_t block, int threads, double seconds) {
    std::uint64_t total_size = 0;
    for (const auto& f : files) total_size += f.size();

    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> errors{0};
    std::vector<std::vector<float>> lat_ms(threads);          // one vector per thread: no sharing, no locks

    auto worker = [&](int id) {
        AlignedBuf buf(block);
        std::mt19937_64 rng(0x9E3779B97F4A7C15ull * (id + 1));
        auto& lat = lat_ms[id];
        while (!stop.load(std::memory_order_relaxed)) {
            // Pick a file weighted by size, then an aligned offset that leaves room for a whole block.
            std::uint64_t pos = rng() % total_size;
            std::size_t fi = 0;
            while (pos >= files[fi].size()) pos -= files[fi++].size();
            const File& f = files[fi];
            if (f.size() < block) continue;
            std::uint64_t off = (rng() % ((f.size() - block) / kAlign + 1)) * kAlign;
            auto t0 = Clock::now();
            if (!f.read_at(buf.p, block, off)) { errors.fetch_add(1, std::memory_order_relaxed); continue; }
            lat.push_back(std::chrono::duration<float, std::milli>(Clock::now() - t0).count());
        }
    };

    auto t0 = Clock::now();
    {
        std::vector<std::jthread> pool;
        for (int i = 0; i < threads; ++i) pool.emplace_back(worker, i);
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        stop.store(true, std::memory_order_relaxed);
    }                                                        // jthreads join here
    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    std::vector<float> all;
    for (auto& v : lat_ms) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());
    auto pct = [&](double q) { return all.empty() ? 0.0 : double(all[std::size_t(q * (all.size() - 1))]); };
    return {threads, all.size() * double(block) / elapsed / 1e9, pct(0.50), pct(0.99), all.size(), errors.load()};
}

int main(int argc, char** argv) {
    std::vector<fs::path> inputs;
    std::uint64_t block_kib = 12288;                         // ~ one DeepSeek V4 Flash expert (3 x 4096 x 2048 at FP4)
    std::vector<int> thread_counts{1, 2, 4, 8, 16};
    double seconds = 6;
    bool cached = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--block-kib") block_kib = std::stoull(next());
        else if (a == "--seconds") seconds = std::stod(next());
        else if (a == "--cached") cached = true;
        else if (a == "--threads") {
            thread_counts.clear();
            std::string s = next();
            for (std::size_t p = 0; p < s.size();) {
                std::size_t q = s.find(',', p);
                thread_counts.push_back(std::stoi(s.substr(p, q - p)));
                p = q == std::string::npos ? s.size() : q + 1;
            }
        } else inputs.emplace_back(a);
    }
    if (inputs.empty()) {
        std::fprintf(stderr, "usage: iobench <dir-or-file>... [--block-kib N] [--threads 1,2,4] [--seconds S] [--cached]\n");
        return 2;
    }
    std::uint64_t block = block_kib * 1024;
    if (block % kAlign) { std::fprintf(stderr, "block must be a multiple of 4 KiB\n"); return 2; }

    std::vector<File> files;
    for (const auto& in : inputs) {
        if (fs::is_directory(in)) {
            for (const auto& e : fs::directory_iterator(in))
                if (e.is_regular_file() && e.path().extension() == ".safetensors") files.emplace_back(e.path(), cached);
        } else files.emplace_back(in, cached);
    }
    std::uint64_t total = 0;
    for (const auto& f : files) total += f.size();
    std::printf("files %zu, %.1f GB, block %llu KiB, %s, %.0f s per point\n", files.size(), total / 1e9,
                static_cast<unsigned long long>(block_kib), cached ? "OS cache ON" : "OS cache bypassed", seconds);
    std::printf("threads,GB_per_s,p50_ms,p99_ms,reads,errors\n");
    for (int t : thread_counts) {
        Result r = run(files, block, t, seconds);
        std::printf("%d,%.2f,%.1f,%.1f,%llu,%llu\n", r.threads, r.gbps, r.p50_ms, r.p99_ms,
                    static_cast<unsigned long long>(r.reads), static_cast<unsigned long long>(r.errors));
        std::fflush(stdout);
    }
}
