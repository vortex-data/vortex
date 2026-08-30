#include <boost/unordered/unordered_flat_map.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

struct __attribute__((packed)) ShortKey {
    uint64_t bytes;
    uint8_t length;

    friend bool operator==(const ShortKey& left, const ShortKey& right) noexcept {
        return left.bytes == right.bytes && left.length == right.length;
    }
};

struct ShortHash {
    using is_avalanching = std::true_type;

    size_t operator()(const ShortKey& key) const noexcept {
        const uint64_t x = key.bytes ^ (static_cast<uint64_t>(key.length) << 56);
        uint64_t hash = x * 0x9e3779b97f4a7c15ULL;
        hash ^= hash >> 32;
        return static_cast<size_t>(hash);
    }
};

struct LongHash {
    using is_avalanching = std::true_type;

    size_t operator()(uint64_t value) const noexcept {
        value ^= value >> 32;
        value *= 0xd6e8feb86659fd93ULL;
        value ^= value >> 32;
        return static_cast<size_t>(value);
    }
};

struct Trace {
    std::vector<std::pair<ShortKey, uint16_t>> short_entries;
    std::vector<std::pair<uint64_t, uint16_t>> long_entries;
    std::vector<ShortKey> short_probes;
    std::vector<uint64_t> long_probes;
};

using LongMap = boost::unordered_flat_map<uint64_t, uint16_t, LongHash>;

extern "C" __attribute__((noinline, used)) uint64_t asm_boost_long(
    const LongMap* map, const uint64_t* keys, std::size_t len) {
    uint64_t checksum = 0;
    for (std::size_t index = 0; index < len; ++index) {
        const auto found = map->find(keys[index]);
        if (found != map->end()) checksum += found->second;
    }
    return checksum;
}

uint64_t read_u64(std::istream& input) {
    uint64_t value;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("truncated u64");
    return value;
}

uint16_t read_u16(std::istream& input) {
    uint16_t value;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("truncated u16");
    return value;
}

Trace load_trace(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open trace");
    char magic[8];
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, "OPHASH01", 8) != 0)
        throw std::runtime_error("invalid trace");
    const size_t short_entry_count = read_u64(input);
    const size_t long_entry_count = read_u64(input);
    const size_t short_probe_count = read_u64(input);
    const size_t long_probe_count = read_u64(input);

    Trace trace;
    trace.short_entries.reserve(short_entry_count);
    trace.long_entries.reserve(long_entry_count);
    trace.short_probes.reserve(short_probe_count);
    trace.long_probes.reserve(long_probe_count);
    for (size_t index = 0; index < short_entry_count; ++index) {
        ShortKey key;
        input.read(reinterpret_cast<char*>(&key), sizeof(key));
        trace.short_entries.emplace_back(key, read_u16(input));
    }
    for (size_t index = 0; index < long_entry_count; ++index)
        trace.long_entries.emplace_back(read_u64(input), read_u16(input));
    for (size_t index = 0; index < short_probe_count; ++index) {
        ShortKey key;
        input.read(reinterpret_cast<char*>(&key), sizeof(key));
        if (!input) throw std::runtime_error("truncated short probe");
        trace.short_probes.push_back(key);
    }
    for (size_t index = 0; index < long_probe_count; ++index)
        trace.long_probes.push_back(read_u64(input));
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("trailing trace bytes");
    return trace;
}

template <typename Function>
double measure(size_t warmups, size_t iterations, Function&& function) {
    uint64_t checksum = 0;
    for (size_t index = 0; index < warmups; ++index) checksum ^= function();
    std::vector<double> samples;
    samples.reserve(iterations);
    for (size_t index = 0; index < iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        checksum ^= function();
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    asm volatile("" : "+r"(checksum) : : "memory");
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: bench_cpp TRACE\n";
        return 2;
    }
    const size_t warmups = std::getenv("HASH_WARMUPS")
        ? std::stoull(std::getenv("HASH_WARMUPS")) : 3;
    const size_t iterations = std::getenv("HASH_ITERATIONS")
        ? std::stoull(std::getenv("HASH_ITERATIONS")) : 15;
    const Trace trace = load_trace(argv[1]);

    boost::unordered_flat_map<ShortKey, uint16_t, ShortHash> short_map;
    short_map.reserve(trace.short_entries.size());
    for (const auto& [key, value] : trace.short_entries) short_map.emplace(key, value);
    LongMap long_map;
    long_map.reserve(trace.long_entries.size());
    for (const auto& [key, value] : trace.long_entries) long_map.emplace(key, value);

    const double short_ms = measure(warmups, iterations, [&]() {
        uint64_t checksum = 0;
        for (const ShortKey& key : trace.short_probes) {
            const auto found = short_map.find(key);
            if (found != short_map.end()) checksum += found->second;
        }
        return checksum;
    });
    const double long_ms = measure(warmups, iterations, [&]() {
        uint64_t checksum = 0;
        for (const uint64_t key : trace.long_probes) {
            const auto found = long_map.find(key);
            if (found != long_map.end()) checksum += found->second;
        }
        return checksum;
    });
    const size_t probes = trace.short_probes.size() + trace.long_probes.size();
    const double total_ms = short_ms + long_ms;
    const double ns_per_probe = total_ms * 1e6 / static_cast<double>(probes);
    std::cout << "trace,short_entries=" << trace.short_entries.size()
              << ",long_entries=" << trace.long_entries.size()
              << ",short_buckets=" << short_map.bucket_count()
              << ",long_buckets=" << long_map.bucket_count()
              << ",short_probes=" << trace.short_probes.size()
              << ",long_probes=" << trace.long_probes.size()
              << ",total_probes=" << probes << '\n';
    std::cout << "hash,name=boost-unordered_flat_map-reference"
              << ",warmups=" << warmups << ",iterations=" << iterations
              << ",short_ms=" << short_ms << ",long_ms=" << long_ms
              << ",total_ms=" << total_ms << ",ns_per_probe=" << ns_per_probe
              << ",mprobes_s=" << 1000.0 / ns_per_probe << '\n';
}


