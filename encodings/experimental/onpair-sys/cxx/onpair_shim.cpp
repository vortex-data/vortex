// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "onpair_shim.h"

#include <onpair/api.h>

#include <cstdlib>
#include <cstring>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using onpair::DECOMPRESS_BUFFER_PADDING;
using onpair::DictionaryView;
using onpair::OnPairColumn;
using onpair::OnPairColumnView;
using onpair::StoreView;
using onpair::encoding::DynamicThreshold;
using onpair::encoding::TrainingConfig;

namespace {

struct ColumnHandle {
    OnPairColumn column;
    std::optional<OnPairColumnView> view;

    const OnPairColumnView& get_view() {
        if (!view) {
            view.emplace(column.view());
        }
        return *view;
    }
};

void clear_bitmap(uint8_t* out, size_t n) noexcept {
    std::memset(out, 0, (n + 7) / 8);
}

inline void set_bit(uint8_t* out, size_t i) noexcept {
    out[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
}

// Upper bound for the size of a single decompressed row. We don't have a
// per-row decoder capacity API, so we conservatively use total bytes_used()
// + padding, which is always at least as large as any single row.
size_t row_decompress_capacity(const OnPairColumnView& view) noexcept {
    return view.bytes_used() + DECOMPRESS_BUFFER_PADDING + 1;
}

// uint64 → uint32 offset copy. The C++ API takes uint32_t offsets; our FFI
// stays uint64 so Rust callers don't have to truncate. We bail out on
// overflow rather than silently wrapping.
bool offsets_fit_u32(const uint64_t* offsets, size_t n_plus_one) noexcept {
    for (size_t i = 0; i < n_plus_one; ++i) {
        if (offsets[i] > static_cast<uint64_t>(UINT32_MAX)) {
            return false;
        }
    }
    return true;
}

} // namespace

extern "C" {

OnPairStatus onpair_column_compress(
    const uint8_t* bytes,
    const uint64_t* offsets,
    size_t n,
    OnPairTrainingConfig config,
    OnPairColumnHandle** out_handle) {
    if (out_handle == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    *out_handle = nullptr;
    if ((bytes == nullptr && n > 0) || offsets == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    if (config.bits < 9 || config.bits > 16) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    if (!offsets_fit_u32(offsets, n + 1)) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    try {
        TrainingConfig tc{};
        tc.bits = static_cast<uint8_t>(config.bits);
        tc.threshold = DynamicThreshold{config.threshold};
        if (config.seed != 0) {
            tc.seed = config.seed;
        }

        // Re-pack uint64 → uint32 in a temporary so we can call the
        // (data, offsets, n, cfg) overload that takes uint32 offsets.
        std::vector<uint32_t> off32(n + 1);
        for (size_t i = 0; i < n + 1; ++i) {
            off32[i] = static_cast<uint32_t>(offsets[i]);
        }

        auto column = OnPairColumn::compress(
            reinterpret_cast<const char*>(bytes),
            off32.data(),
            n,
            tc);
        auto handle = std::make_unique<ColumnHandle>();
        handle->column = std::move(column);
        *out_handle = reinterpret_cast<OnPairColumnHandle*>(handle.release());
        return ONPAIR_OK;
    } catch (const std::bad_alloc&) {
        return ONPAIR_ERR_OOM;
    } catch (...) {
        return ONPAIR_ERR_INTERNAL;
    }
}

OnPairStatus onpair_column_deserialize(
    const uint8_t* data,
    size_t len,
    OnPairColumnHandle** out_handle) {
    if (out_handle == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    *out_handle = nullptr;
    if (data == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    try {
        std::stringstream ss;
        ss.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));
        auto column = OnPairColumn::read_from(ss);
        auto handle = std::make_unique<ColumnHandle>();
        handle->column = std::move(column);
        *out_handle = reinterpret_cast<OnPairColumnHandle*>(handle.release());
        return ONPAIR_OK;
    } catch (const std::bad_alloc&) {
        return ONPAIR_ERR_OOM;
    } catch (...) {
        return ONPAIR_ERR_BAD_FORMAT;
    }
}

OnPairStatus onpair_column_serialize(
    const OnPairColumnHandle* handle,
    uint8_t** out_data,
    size_t* out_len) {
    if (handle == nullptr || out_data == nullptr || out_len == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    *out_data = nullptr;
    *out_len = 0;
    try {
        const auto* h = reinterpret_cast<const ColumnHandle*>(handle);
        std::stringstream ss;
        h->column.write_to(ss);
        const std::string s = ss.str();
        auto* buf = static_cast<uint8_t*>(std::malloc(s.size() == 0 ? 1 : s.size()));
        if (buf == nullptr) {
            return ONPAIR_ERR_OOM;
        }
        std::memcpy(buf, s.data(), s.size());
        *out_data = buf;
        *out_len = s.size();
        return ONPAIR_OK;
    } catch (const std::bad_alloc&) {
        return ONPAIR_ERR_OOM;
    } catch (...) {
        return ONPAIR_ERR_INTERNAL;
    }
}

void onpair_column_free(OnPairColumnHandle* handle) {
    delete reinterpret_cast<ColumnHandle*>(handle);
}

void onpair_buffer_free(uint8_t* data, size_t /*len*/) {
    std::free(data);
}

size_t onpair_column_len(const OnPairColumnHandle* handle) {
    if (handle == nullptr) {
        return 0;
    }
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    return h->get_view().num_strings();
}

uint32_t onpair_column_bits(const OnPairColumnHandle* handle) {
    if (handle == nullptr) {
        return 0;
    }
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    return static_cast<uint32_t>(h->get_view().bits());
}

size_t onpair_column_dict_size(const OnPairColumnHandle* handle) {
    if (handle == nullptr) {
        return 0;
    }
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    return h->get_view().dictionary().num_tokens();
}

OnPairStatus onpair_column_decompress(
    const OnPairColumnHandle* handle,
    size_t row_id,
    uint8_t* out_buf,
    size_t out_capacity,
    size_t* out_len) {
    if (handle == nullptr || out_buf == nullptr || out_len == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    *out_len = 0;
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    try {
        const auto& view = h->get_view();
        if (row_id >= view.num_strings()) {
            return ONPAIR_ERR_OUT_OF_RANGE;
        }
        // The decoder over-copies by DECOMPRESS_BUFFER_PADDING bytes per token,
        // so the caller's buffer must include that headroom.
        const size_t needed = row_decompress_capacity(view);
        if (needed > out_capacity) {
            return ONPAIR_ERR_OOM;
        }
        *out_len = view.decompress(row_id, reinterpret_cast<char*>(out_buf));
        return ONPAIR_OK;
    } catch (...) {
        return ONPAIR_ERR_INTERNAL;
    }
}

size_t onpair_column_decompress_capacity(const OnPairColumnHandle* handle) {
    if (handle == nullptr) {
        return DECOMPRESS_BUFFER_PADDING;
    }
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    return row_decompress_capacity(h->get_view());
}

OnPairStatus onpair_column_equals_into(
    const OnPairColumnHandle* handle,
    const uint8_t* needle,
    size_t needle_len,
    uint8_t* out_bits) {
    if (handle == nullptr || out_bits == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    try {
        const auto& view = h->get_view();
        clear_bitmap(out_bits, view.num_strings());
        view.equals(
            std::string_view(reinterpret_cast<const char*>(needle), needle_len),
            [out_bits](size_t idx) { set_bit(out_bits, idx); });
        return ONPAIR_OK;
    } catch (const std::bad_alloc&) {
        return ONPAIR_ERR_OOM;
    } catch (...) {
        return ONPAIR_ERR_INTERNAL;
    }
}

OnPairStatus onpair_column_starts_with_into(
    const OnPairColumnHandle* handle,
    const uint8_t* needle,
    size_t needle_len,
    uint8_t* out_bits) {
    if (handle == nullptr || out_bits == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    try {
        const auto& view = h->get_view();
        clear_bitmap(out_bits, view.num_strings());
        view.starts_with(
            std::string_view(reinterpret_cast<const char*>(needle), needle_len),
            [out_bits](size_t idx) { set_bit(out_bits, idx); });
        return ONPAIR_OK;
    } catch (const std::bad_alloc&) {
        return ONPAIR_ERR_OOM;
    } catch (...) {
        return ONPAIR_ERR_INTERNAL;
    }
}

OnPairStatus onpair_column_contains_into(
    const OnPairColumnHandle* handle,
    const uint8_t* needle,
    size_t needle_len,
    uint8_t* out_bits) {
    if (handle == nullptr || out_bits == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    try {
        const auto& view = h->get_view();
        clear_bitmap(out_bits, view.num_strings());
        view.contains(
            std::string_view(reinterpret_cast<const char*>(needle), needle_len),
            [out_bits](size_t idx) { set_bit(out_bits, idx); });
        return ONPAIR_OK;
    } catch (const std::bad_alloc&) {
        return ONPAIR_ERR_OOM;
    } catch (...) {
        return ONPAIR_ERR_INTERNAL;
    }
}

OnPairStatus onpair_column_dict_copy(
    const OnPairColumnHandle* handle,
    uint8_t* out_bytes,
    size_t bytes_capacity,
    uint64_t* out_offsets) {
    if (handle == nullptr || out_offsets == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    try {
        const auto& dv = h->get_view().dictionary();
        const size_t n = dv.num_tokens();
        const auto* raw_off = dv.raw_offsets();
        const auto* raw_bytes_ptr = dv.raw_bytes();
        const size_t total = raw_off[n];
        if (total > bytes_capacity) {
            return ONPAIR_ERR_OOM;
        }
        if (total > 0 && out_bytes != nullptr) {
            std::memcpy(out_bytes, raw_bytes_ptr, total);
        }
        for (size_t i = 0; i <= n; ++i) {
            out_offsets[i] = static_cast<uint64_t>(raw_off[i]);
        }
        return ONPAIR_OK;
    } catch (...) {
        return ONPAIR_ERR_INTERNAL;
    }
}

size_t onpair_column_dict_bytes(const OnPairColumnHandle* handle) {
    if (handle == nullptr) {
        return 0;
    }
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    try {
        const auto& dv = h->get_view().dictionary();
        return dv.bytes_used();
    } catch (...) {
        return 0;
    }
}

OnPairStatus onpair_column_parts(
    const OnPairColumnHandle* handle,
    OnPairColumnParts*        out_parts) {
    if (handle == nullptr || out_parts == nullptr) {
        return ONPAIR_ERR_INVALID_ARG;
    }
    auto* h = const_cast<ColumnHandle*>(reinterpret_cast<const ColumnHandle*>(handle));
    try {
        const auto& view = h->get_view();
        const DictionaryView& dv = view.dictionary();
        const StoreView&      sv = view.store();

        const size_t   dict_size  = dv.num_tokens();
        const uint32_t* dict_off  = dv.raw_offsets();
        const size_t   dict_bytes = dict_size == 0 ? 0 : dict_off[dict_size];

        const size_t   num_rows   = sv.num_strings();
        const uint32_t bw         = static_cast<uint32_t>(sv.bits());
        const size_t   tokens     = sv.num_tokens();
        // The packed stream is laid out by BitWriter as a vector<uint64_t>;
        // round-up-to-u64 of (tokens * bits) bits.
        const size_t   packed_u64 = (tokens * bw + 63) / 64;

        out_parts->dict_bytes           = dv.raw_bytes();
        out_parts->dict_bytes_len       = dict_bytes;
        out_parts->dict_offsets         = dict_off;
        out_parts->dict_offsets_len     = dict_size + 1;
        out_parts->codes_packed         = sv.packed_data();
        out_parts->codes_packed_u64_len = packed_u64;
        out_parts->codes_boundaries     = sv.boundaries();
        out_parts->codes_boundaries_len = num_rows + 1;
        out_parts->bits                 = bw;
        out_parts->num_rows             = num_rows;
        return ONPAIR_OK;
    } catch (...) {
        return ONPAIR_ERR_INTERNAL;
    }
}

} // extern "C"
