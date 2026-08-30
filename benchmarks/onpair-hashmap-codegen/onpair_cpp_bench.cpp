// Times OnPair training and parsing separately, using the paper's own C++
// library (github.com/gargiulofrancesco/onpair_cpp), so the GCC-versus-Clang
// comparison runs over the real algorithm rather than a hashmap microbenchmark.
//
// usage: onpair_bench DATASET.txt [bits] [iterations]
//   DATASET.txt is one string per line.

#include <onpair/core/store.h>
#include <onpair/core/types.h>
#include <onpair/encoding/parsing/parser.h>
#include <onpair/encoding/training/config.h>
#include <onpair/encoding/training/trainer.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace op = onpair::encoding;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: onpair_bench DATASET.txt [bits] [iterations]\n");
        return 2;
    }
    const int bits = argc > 2 ? std::atoi(argv[2]) : 16;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 5;

    std::vector<uint8_t> data;
    std::vector<uint32_t> offsets{0};
    {
        std::ifstream input(argv[1]);
        if (!input) {
            std::fprintf(stderr, "cannot open %s\n", argv[1]);
            return 2;
        }
        std::string line;
        while (std::getline(input, line)) {
            data.insert(data.end(), line.begin(), line.end());
            offsets.push_back(static_cast<uint32_t>(data.size()));
        }
    }
    const size_t rows = offsets.size() - 1;
    const double megabytes = static_cast<double>(data.size()) / (1024.0 * 1024.0);

    op::TrainingConfig cfg;
    cfg.bits = static_cast<onpair::BitWidth>(bits);
    cfg.threshold = op::DynamicThreshold{0.15};
    cfg.seed = 42;

    std::vector<double> train_ms, parse_ms;
    uint64_t tokens = 0, dict_tokens = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        op::TrainResult trained = op::train(data.data(), offsets.data(), rows, cfg);
        auto middle = std::chrono::steady_clock::now();

        onpair::Store store;
        op::parse(data.data(), offsets.data(), rows, trained.lpm, cfg.bits, store);
        auto stop = std::chrono::steady_clock::now();

        train_ms.push_back(std::chrono::duration<double, std::milli>(middle - start).count());
        parse_ms.push_back(std::chrono::duration<double, std::milli>(stop - middle).count());
        tokens = store.num_tokens();
        dict_tokens = trained.dict.num_tokens();
    }
    std::sort(train_ms.begin(), train_ms.end());
    std::sort(parse_ms.begin(), parse_ms.end());
    const double train = train_ms[train_ms.size() / 2];
    const double parse = parse_ms[parse_ms.size() / 2];

    std::printf(
        "onpair,dataset=%s,bits=%d,rows=%zu,mib=%.2f,dict_tokens=%llu,tokens=%llu,"
        "train_ms=%.2f,parse_ms=%.2f,train_mibs=%.1f,parse_mibs=%.1f\n",
        argv[1], bits, rows, megabytes, (unsigned long long)dict_tokens,
        (unsigned long long)tokens, train, parse, megabytes / (train / 1000.0),
        megabytes / (parse / 1000.0));
    return 0;
}
