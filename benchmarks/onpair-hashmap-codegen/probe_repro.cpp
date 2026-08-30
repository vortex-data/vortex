// Minimal reproducer for the GCC-versus-LLVM gap on a Boost-style FOA (group15)
// hash probe loop.
//
// The loop is a stripped-down copy of boost::unordered_flat_map::find for a
// u64 key: one metadata group, SIMD fingerprint match, then a full key compare
// on each candidate. No insertion, no overflow probing, no allocation.
//
// The whole gap only appears when the fingerprint/key branches are
// unpredictable, so the driver sweeps hit rate from 0% to 100%.
//
//   g++     -std=c++20 -O3 -DNDEBUG -march=native probe_repro.cpp -o probe-gcc
//   clang++ -std=c++20 -O3 -DNDEBUG -march=native probe_repro.cpp -o probe-clang
//   ./probe-gcc && ./probe-clang

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kGroups = 256;      // 4096 slots, fits in L2
constexpr std::size_t kSlotsPerGroup = 16;
constexpr std::size_t kProbes = 1u << 20;

struct alignas(16) Metadata {
    uint8_t bytes[16];
};

struct Slot {
    uint64_t key;
    uint64_t value;
};

inline uint64_t mix(uint64_t value) {
    value ^= value >> 32;
    value *= 0xd6e8feb86659fd93ULL;
    value ^= value >> 32;
    return value;
}

__attribute__((noinline, used)) uint64_t probe_loop(
    const Metadata* metadata, const Slot* slots, const uint64_t* keys, std::size_t len) {
    uint64_t checksum = 0;
    for (std::size_t index = 0; index < len; ++index) {
        const uint64_t key = keys[index];
        const uint64_t hash = mix(key);
        const std::size_t group = (hash >> 24) & (kGroups - 1);
        const uint8_t fingerprint = static_cast<uint8_t>(hash) | 0x80;

        const __m128i wanted = _mm_set1_epi8(static_cast<char>(fingerprint));
        const __m128i present =
            _mm_load_si128(reinterpret_cast<const __m128i*>(metadata[group].bytes));
        uint32_t mask = static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(wanted, present)));

        while (mask != 0) {
            const std::size_t slot = group * kSlotsPerGroup + __builtin_ctz(mask);
            if (slots[slot].key == key) {
                checksum += slots[slot].value;
                break;
            }
            mask &= mask - 1;
        }
    }
    return checksum;
}

}  // namespace

int main() {
    std::vector<Metadata> metadata(kGroups);
    std::vector<Slot> slots(kGroups * kSlotsPerGroup);
    std::mt19937_64 rng(42);

    // Fill each group two-thirds full with distinct keys.
    std::vector<uint64_t> resident;
    for (std::size_t group = 0; group < kGroups; ++group) {
        for (std::size_t slot = 0; slot < 10; ++slot) {
            uint64_t key;
            do {
                key = rng();
            } while (((mix(key) >> 24) & (kGroups - 1)) != group);
            const std::size_t index = group * kSlotsPerGroup + slot;
            metadata[group].bytes[slot] = static_cast<uint8_t>(mix(key)) | 0x80;
            slots[index] = Slot{key, key & 0xffff};
            resident.push_back(key);
        }
    }

    std::printf("hit_pct,ms,ns_per_probe\n");
    for (int hit_pct : {0, 25, 50, 75, 100}) {
        std::vector<uint64_t> keys(kProbes);
        std::mt19937_64 pick(7);
        for (auto& key : keys) {
            key = (pick() % 100 < static_cast<uint64_t>(hit_pct))
                      ? resident[pick() % resident.size()]
                      : (pick() | (1ull << 63));
        }

        std::vector<double> samples;
        for (int iteration = 0; iteration < 13; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            const uint64_t checksum = probe_loop(metadata.data(), slots.data(), keys.data(), kProbes);
            const auto end = std::chrono::steady_clock::now();
            asm volatile("" ::"r"(checksum));
            if (iteration >= 3) {
                samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }
        }
        std::sort(samples.begin(), samples.end());
        const double ms = samples[samples.size() / 2];
        std::printf("%d,%.4f,%.3f\n", hit_pct, ms, ms * 1e6 / kProbes);
    }
    return 0;
}
