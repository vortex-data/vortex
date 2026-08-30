// Is the C++ per-row decompress() paying for dispatch_bits on every call?
#include <onpair/api.h>
#include <onpair/decoding/token_cursor.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>
namespace op = onpair;
int main(int argc, char** argv) {
    const int bits = argc > 2 ? std::atoi(argv[2]) : 16;
    std::vector<char> data; std::vector<uint32_t> offsets{0};
    { std::ifstream in(argv[1]); std::string line;
      while (std::getline(in, line)) { data.insert(data.end(), line.begin(), line.end());
        offsets.push_back((uint32_t)data.size()); } }
    const size_t rows = offsets.size() - 1;
    op::encoding::TrainingConfig cfg; cfg.bits = (op::BitWidth)bits;
    cfg.threshold = op::encoding::DynamicThreshold{0.15}; cfg.seed = 42;
    auto col = op::OnPairColumn::compress(data.data(), offsets.data(), rows, cfg);
    auto view = col.view();
    std::vector<uint32_t> order(rows); std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), std::mt19937_64(42));
    std::vector<char> buf(64*1024 + op::DECOMPRESS_BUFFER_PADDING);

    std::vector<double> a, b;
    for (int it = 0; it < 7; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        size_t s = 0; for (uint32_t k : order) s += view.decompress(k, buf.data());
        auto t1 = std::chrono::steady_clock::now();
        asm volatile("" :: "r"(s));
        a.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());

        // Same work, bit width resolved once outside the row loop.
        auto sv = view.store(); auto dv = view.dictionary();
        const uint8_t* dbytes = dv.raw_bytes(); const uint32_t* doff = dv.raw_offsets();
        t0 = std::chrono::steady_clock::now();
        size_t s2 = 0;
        op::dispatch_bits(sv.bits(), [&](auto bw) {
            for (uint32_t k : order) {
                auto span = sv.string_span(k);
                op::decoding::TokenCursor<bw.value> cur(sv.packed_data(), span);
                size_t w = 0;
                while (cur.has_more()) { op::Token t = cur.next(); uint32_t o = doff[t];
                    std::memcpy(buf.data() + w, dbytes + o, op::MAX_TOKEN_SIZE);
                    w += doff[t+1] - o; }
                s2 += w;
            }
        });
        t1 = std::chrono::steady_clock::now();
        asm volatile("" :: "r"(s2));
        b.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
    }
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    std::printf("bits=%d rows=%zu  api_ns_per_row=%.1f  hoisted_ns_per_row=%.1f  speedup=%.2fx\n",
                bits, rows, a[a.size()/2]*1e6/rows, b[b.size()/2]*1e6/rows,
                a[a.size()/2]/b[b.size()/2]);
    return 0;
}
