// Decompression benchmark for the onpair_cpp implementation: bulk decode of a
// whole column, and random-access decode of individual rows in shuffled order.
//
// usage: onpair_decode DATASET.txt [bits] [iterations]

#include <onpair/api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace op = onpair;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: onpair_decode DATASET.txt [bits] [iterations]\n");
        return 2;
    }
    const int bits = argc > 2 ? std::atoi(argv[2]) : 16;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 5;

    std::vector<char> data;
    std::vector<uint32_t> offsets{0};
    {
        std::ifstream input(argv[1]);
        std::string line;
        while (std::getline(input, line)) {
            data.insert(data.end(), line.begin(), line.end());
            offsets.push_back(static_cast<uint32_t>(data.size()));
        }
    }
    const size_t rows = offsets.size() - 1;
    const double megabytes = static_cast<double>(data.size()) / (1024.0 * 1024.0);

    op::encoding::TrainingConfig cfg;
    cfg.bits = static_cast<op::BitWidth>(bits);
    cfg.threshold = op::encoding::DynamicThreshold{0.15};
    cfg.seed = 42;

    op::OnPairColumn column =
        op::OnPairColumn::compress(data.data(), offsets.data(), rows, cfg);
    op::OnPairColumnView view = column.view();

    // Bulk decode.
    std::vector<char> out(data.size() + op::DECOMPRESS_BUFFER_PADDING);
    std::vector<double> bulk_ms;
    size_t written = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        written = view.decompress_all(out.data());
        const auto stop = std::chrono::steady_clock::now();
        bulk_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    const bool bulk_ok =
        written == data.size() && std::memcmp(out.data(), data.data(), data.size()) == 0;

    // Random access in shuffled order.
    std::vector<uint32_t> order(rows);
    std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), std::mt19937_64(42));

    std::vector<char> row(64 * 1024 + op::DECOMPRESS_BUFFER_PADDING);
    std::vector<double> random_ms;
    uint64_t checksum = 0;
    bool random_ok = true;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        uint64_t sum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (const uint32_t index : order) {
            sum += view.decompress(index, row.data());
        }
        const auto stop = std::chrono::steady_clock::now();
        random_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        checksum = sum;
    }
    for (const uint32_t index : order) {
        const size_t length = view.decompress(index, row.data());
        const size_t expected = offsets[index + 1] - offsets[index];
        if (length != expected ||
            std::memcmp(row.data(), data.data() + offsets[index], expected) != 0) {
            random_ok = false;
            break;
        }
    }

    std::sort(bulk_ms.begin(), bulk_ms.end());
    std::sort(random_ms.begin(), random_ms.end());
    const double bulk = bulk_ms[bulk_ms.size() / 2];
    const double random = random_ms[random_ms.size() / 2];

    std::printf(
        "decode,impl=onpair_cpp,dataset=%s,bits=%d,rows=%zu,mib=%.2f,"
        "compressed_mib=%.2f,bulk_ms=%.2f,bulk_mibs=%.1f,random_ms=%.2f,"
        "random_ns_per_row=%.1f,checksum=%llu,bulk_ok=%d,random_ok=%d\n",
        argv[1], bits, rows, megabytes,
        double(column.bytes_used()) / 1048576.0,
        bulk, megabytes / (bulk / 1000.0), random,
        random * 1e6 / double(rows), (unsigned long long)checksum, bulk_ok, random_ok);
    return 0;
}
