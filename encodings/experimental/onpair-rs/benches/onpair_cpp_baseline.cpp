#include <onpair/encoding/parsing/parser.h>
#include <onpair/encoding/training/trainer.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct Corpus {
    std::vector<uint8_t> data;
    std::vector<uint32_t> offsets;
};

uint32_t read_u32(const uint8_t *pointer) {
    uint32_t value;
    std::memcpy(&value, pointer, sizeof(value));
    return value;
}

uint64_t read_u64(const uint8_t *pointer) {
    uint64_t value;
    std::memcpy(&value, pointer, sizeof(value));
    return value;
}

Corpus load_corpus(const char *path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open corpus");
    }
    const auto size = static_cast<size_t>(stream.tellg());
    stream.seekg(0);
    std::vector<uint8_t> input(size);
    stream.read(reinterpret_cast<char *>(input.data()), input.size());
    if (size < 24 || std::memcmp(input.data(), "ONPAIR01", 8) != 0) {
        throw std::runtime_error("invalid corpus");
    }

    const size_t payload = read_u64(input.data() + 8);
    const size_t rows = read_u64(input.data() + 16);
    Corpus corpus;
    corpus.data.reserve(payload);
    corpus.offsets.reserve(rows + 1);
    corpus.offsets.push_back(0);
    size_t cursor = 24;
    for (size_t row = 0; row < rows; ++row) {
        const size_t length = read_u32(input.data() + cursor);
        cursor += 4;
        corpus.data.insert(corpus.data.end(), input.begin() + cursor, input.begin() + cursor + length);
        cursor += length;
        corpus.offsets.push_back(static_cast<uint32_t>(corpus.data.size()));
    }
    if (cursor != input.size() || corpus.data.size() != payload) {
        throw std::runtime_error("truncated corpus");
    }
    return corpus;
}

template <typename Function>
double median_ms(size_t iterations, Function &&function) {
    function();
    std::vector<double> samples;
    samples.reserve(iterations);
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        const auto begin = std::chrono::steady_clock::now();
        function();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

volatile size_t checksum = 0;

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: onpair_cpp_baseline CORPUS\n";
        return 2;
    }
    const auto corpus = load_corpus(argv[1]);
    const auto bits = static_cast<onpair::BitWidth>(
        std::getenv("ONPAIR_BITS") ? std::atoi(std::getenv("ONPAIR_BITS")) : 16);
    const size_t iterations = std::getenv("ONPAIR_REPORT_ITERATIONS")
                                  ? std::strtoul(std::getenv("ONPAIR_REPORT_ITERATIONS"), nullptr, 10)
                                  : 5;
    onpair::encoding::TrainingConfig config;
    config.bits = bits;
    const double sample_fraction = std::getenv("ONPAIR_SAMPLE_FRACTION")
                                       ? std::strtod(std::getenv("ONPAIR_SAMPLE_FRACTION"), nullptr)
                                       : 0.5;
    config.threshold = onpair::encoding::DynamicThreshold {sample_fraction};
    config.seed = 42;

    auto trained =
        onpair::encoding::train(corpus.data.data(), corpus.offsets.data(), corpus.offsets.size() - 1, config);
    const double train_ms = median_ms(iterations, [&] {
        auto value = onpair::encoding::train(corpus.data.data(),
                                             corpus.offsets.data(),
                                             corpus.offsets.size() - 1,
                                             config);
        checksum = checksum + value.dict.num_tokens();
    });
    const double parse_ms = median_ms(iterations, [&] {
        onpair::Store store;
        onpair::encoding::parse(corpus.data.data(),
                                corpus.offsets.data(),
                                corpus.offsets.size() - 1,
                                trained.lpm,
                                bits,
                                store);
        checksum = checksum + store.packed.size() + store.boundaries.size();
    });
    const double full_ms = median_ms(iterations, [&] {
        auto value = onpair::encoding::train(corpus.data.data(),
                                             corpus.offsets.data(),
                                             corpus.offsets.size() - 1,
                                             config);
        onpair::Store store;
        onpair::encoding::parse(corpus.data.data(),
                                corpus.offsets.data(),
                                corpus.offsets.size() - 1,
                                value.lpm,
                                bits,
                                store);
        checksum = checksum + store.packed.size() + value.dict.num_tokens();
    });
    const double gigabytes = static_cast<double>(corpus.data.size()) / 1e9;
    std::cout << "cpp bits=" << unsigned(bits) << " rows=" << corpus.offsets.size() - 1
              << " dict_tokens=" << trained.dict.num_tokens() << '\n'
              << "train median_ms=" << train_ms << " gb_s=" << gigabytes / (train_ms / 1000.0) << '\n'
              << "parse median_ms=" << parse_ms << " gb_s=" << gigabytes / (parse_ms / 1000.0) << '\n'
              << "full median_ms=" << full_ms << " gb_s=" << gigabytes / (full_ms / 1000.0) << '\n';
}
