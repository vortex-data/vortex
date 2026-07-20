// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/array.hpp"
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <vortex/data_source.hpp>
#include <vortex/writer.hpp>

namespace vortex_test {

namespace fs = std::filesystem;
using namespace vortex;

class TempPath {
public:
    TempPath() = default;
    explicit TempPath(fs::path p) : path_(std::move(p)) {
    }

    TempPath(const TempPath &) = delete;
    TempPath &operator=(const TempPath &) = delete;

    TempPath(TempPath &&other) noexcept : path_(std::move(other.path_)) {
        other.path_.clear();
    }
    TempPath &operator=(TempPath &&other) noexcept {
        if (this != &other) {
            reset();
            path_ = std::move(other.path_);
            other.path_.clear();
        }
        return *this;
    }

    ~TempPath() {
        reset();
    }

    const fs::path &path() const noexcept {
        return path_;
    }
    std::string string() const {
        return path_.string();
    }

    static TempPath unique() {
        auto dir = fs::temp_directory_path() / "vortex_cxx_test";
        fs::create_directories(dir);
        std::string name = std::to_string(std::random_device {}()) + ".vortex";
        return TempPath {dir / name};
    }

private:
    void reset() noexcept {
        if (!path_.empty()) {
            std::error_code ec;
            fs::remove(path_, ec);
        }
    }

    fs::path path_;
};

inline DataType sample_dtype() {
    return dtype::struct_({
        {"age", dtype::uint8()},
        {"height", dtype::uint16(dtype::Nullable)},
    });
}

constexpr size_t SAMPLE_ROWS = 100;

inline std::vector<uint8_t> sample_ages() {
    std::vector<uint8_t> buf(SAMPLE_ROWS);
    for (size_t i = 0; i < SAMPLE_ROWS; ++i) {
        buf[i] = static_cast<uint8_t>(i);
    }
    return buf;
}

inline std::vector<uint16_t> sample_heights() {
    std::vector<uint16_t> buf(SAMPLE_ROWS);
    for (size_t i = 0; i < SAMPLE_ROWS; ++i) {
        buf[i] = static_cast<uint16_t>((i + 1) % 200);
    }
    return buf;
}

inline Array sample_array() {
    auto ages = sample_ages();
    auto heights = sample_heights();
    return make_struct({
        {"age", Array::primitive<uint8_t>(ages)},
        {"height", Array::primitive<uint16_t>(heights, ValidityType::AllValid)},
    });
}

inline TempPath write_sample(const Session &session) {
    TempPath path = TempPath::unique();
    Writer writer = Writer::open(session, path.string(), sample_dtype());
    writer.push(sample_array());
    writer.finish();
    return path;
}
} // namespace vortex_test
