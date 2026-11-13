// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/attributes.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vortex::cuda {
namespace detail {

struct string_dictionary_inputs {
    std::vector<cudf::column_view> nonempty;
    std::vector<cudf::column_view> keys;
    cudf::size_type key_count = 0;
};

/**
 * Validate dictionary metadata and collect nonempty batches.
 *
 * Requires string keys and integer codes; checks aggregate row/key counts against
 * cudf::size_type. Reads only host metadata, so callers must ensure valid row codes
 * are in range. Empty batches are type-checked but contribute no keys or rows.
 */
inline string_dictionary_inputs validate_string_dictionaries(std::vector<cudf::column_view> const &batches) {
    string_dictionary_inputs inputs;
    inputs.nonempty.reserve(batches.size());
    inputs.keys.reserve(batches.size());
    cudf::size_type rows = 0;
    constexpr auto limit = std::numeric_limits<cudf::size_type>::max();
    for (auto const &batch : batches) {
        if (batch.type().id() != cudf::type_id::DICTIONARY32 || batch.num_children() != 2) {
            throw std::invalid_argument("expected string dictionary columns");
        }
        cudf::dictionary_column_view dict {batch};
        if (dict.keys().type().id() != cudf::type_id::STRING || !cudf::is_integral(dict.indices().type()) ||
            dict.indices().type().id() == cudf::type_id::BOOL8) {
            throw std::invalid_argument("expected string keys and integer dictionary codes");
        }
        if (batch.is_empty()) {
            continue;
        }
        if (batch.size() > limit - rows || dict.keys_size() > limit - inputs.key_count) {
            throw std::overflow_error("dictionary row/key count exceeds cudf::size_type");
        }
        rows += batch.size();
        inputs.key_count += dict.keys_size();
        inputs.nonempty.push_back(batch);
        inputs.keys.push_back(dict.keys());
    }
    return inputs;
}

// RMM releases temporaries on their allocation stream, after queued consumers.
// Ownership slots are allocated before any device work starts.
struct string_dictionary_buffers {
    rmm::cuda_stream_view stream;
    rmm::device_async_resource_ref mr; // Returned buffers only; temporaries use cuDF's current resource.
    std::vector<std::unique_ptr<cudf::column>> adjusted;
    std::vector<std::unique_ptr<cudf::scalar>> bases;
    std::unique_ptr<cudf::column> all_keys, all_codes, nonnull_codes, result;
    std::unique_ptr<cudf::scalar> sentinel;
    std::unique_ptr<cudf::table> gathered;

    /**
     * Allocate ownership slots for batch temporaries.
     *
     * Creates host-side slots for at most `batches` inputs without launching GPU
     * work. All device allocations and their deallocations use the same stream.
     */
    string_dictionary_buffers(std::size_t batches,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
        : stream(stream), mr(mr), adjusted(batches), bases(batches) {
    }
};

/**
 * Offset and concatenate dictionary codes into a single INT32 column.
 *
 * Uses each batch's starting position in the combined keys, preserving row validity.
 * Requires validated, nonempty inputs and one ownership slot per batch. Stores the
 * combined codes and all device temporaries in `buffers` without synchronizing.
 */
inline void concatenate_dictionary_codes(string_dictionary_inputs const &inputs,
                                         string_dictionary_buffers &buffers) {
    std::vector<cudf::column_view> codes;
    codes.reserve(inputs.nonempty.size());
    cudf::size_type offset = 0;
    for (std::size_t i = 0; i < inputs.nonempty.size(); ++i) {
        cudf::dictionary_column_view dict {inputs.nonempty[i]};
        buffers.bases[i] =
            std::make_unique<cudf::numeric_scalar<cudf::size_type>>(offset, true, buffers.stream);
        buffers.adjusted[i] = cudf::binary_operation(dict.get_indices_annotated(),
                                                     *buffers.bases[i],
                                                     cudf::binary_operator::ADD,
                                                     cudf::data_type {cudf::type_id::INT32},
                                                     buffers.stream);
        codes.push_back(buffers.adjusted[i]->view());
        offset += dict.keys_size();
    }
    buffers.all_codes = cudf::concatenate(codes, buffers.stream);
}

/**
 * Find a fixed byte width that can be gathered as an integer.
 *
 * Requires non-null keys and rows. Other widths, variable lengths, and outputs
 * exceeding INT32 offsets use the general string path instead.
 */
inline cudf::size_type fixed_string_width(string_dictionary_inputs const &inputs,
                                          string_dictionary_buffers const &buffers) {
    if (inputs.key_count == 0 || buffers.all_keys->has_nulls()) {
        return 0;
    }
    int64_t rows = 0;
    for (auto const &batch : inputs.nonempty) {
        if (batch.has_nulls()) {
            return 0;
        }
        rows += batch.size();
    }
    auto lengths = cudf::strings::count_bytes(cudf::strings_column_view {buffers.all_keys->view()},
                                              buffers.stream);
    auto bounds = cudf::minmax(lengths->view(), buffers.stream);
    auto const width =
        static_cast<cudf::numeric_scalar<cudf::size_type> &>(*bounds.first).value(buffers.stream);
    auto const max =
        static_cast<cudf::numeric_scalar<cudf::size_type> &>(*bounds.second).value(buffers.stream);
    constexpr auto limit = std::numeric_limits<cudf::size_type>::max();
    return width == max && (width == 1 || width == 2 || width == 4 || width == 8) && rows < limit &&
                   rows * width <= limit
               ? width
               : 0;
}

/**
 * Gather fixed-width string bytes without remapping dictionary codes.
 *
 * Requires the non-null, uniform-width inputs checked by `fixed_string_width`.
 * Treats key payloads as integers, concatenates gathered payloads, then builds
 * string offsets once. Different key orders and sliced row codes remain valid.
 */
inline void concatenate_fixed_width_strings(string_dictionary_inputs const &inputs,
                                            cudf::size_type width,
                                            string_dictionary_buffers &buffers) {
    auto const type = cudf::data_type {width == 1   ? cudf::type_id::UINT8
                                       : width == 2 ? cudf::type_id::UINT16
                                       : width == 4 ? cudf::type_id::UINT32
                                                    : cudf::type_id::UINT64};
    auto chars = cudf::strings_column_view {buffers.all_keys->view()}.chars_begin(buffers.stream);
    std::vector<cudf::column_view> payloads;
    payloads.reserve(inputs.nonempty.size());
    for (std::size_t i = 0; i < inputs.nonempty.size(); ++i) {
        cudf::dictionary_column_view dict {inputs.nonempty[i]};
        cudf::column_view keys {type, dict.keys_size(), chars, nullptr, 0};
        auto gathered = cudf::gather(cudf::table_view {{keys}},
                                     dict.get_indices_annotated(),
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     cudf::negative_index_policy::NOT_ALLOWED,
                                     buffers.stream);
        buffers.adjusted[i] = std::move(gathered->release().front());
        payloads.push_back(buffers.adjusted[i]->view());
        chars += static_cast<int64_t>(dict.keys_size()) * width;
    }
    auto values = cudf::concatenate(payloads, buffers.stream, buffers.mr);
    auto const rows = values->size();
    cudf::numeric_scalar<cudf::size_type> zero {0, true, buffers.stream};
    cudf::numeric_scalar<cudf::size_type> step {width, true, buffers.stream};
    auto offsets = cudf::sequence(rows + 1, zero, step, buffers.stream, buffers.mr);
    auto data = values->release();
    buffers.result =
        cudf::make_strings_column(rows, std::move(offsets), std::move(*data.data), 0, rmm::device_buffer {});
}

/**
 * Gather final strings from the combined dictionary codes.
 *
 * Requires concatenated keys and codes prepared by `concatenate_dictionary_codes`.
 * Preserves key order and duplicates and maps null codes to an
 * out-of-range sentinel. Stores the result and stream-ordered temporaries in
 * `buffers` without an explicit synchronization.
 */
inline void gather_dictionary_strings(string_dictionary_inputs const &inputs,
                                      string_dictionary_buffers &buffers) {
    auto map = buffers.all_codes->view();
    if (buffers.all_codes->null_count() != 0) {
        // Gather maps cannot contain nulls. An out-of-range sentinel creates
        // null output rows without dereferencing their arbitrary codes.
        buffers.sentinel = std::make_unique<cudf::numeric_scalar<cudf::size_type>>(inputs.key_count,
                                                                                   true,
                                                                                   buffers.stream);
        buffers.nonnull_codes = cudf::replace_nulls(map, *buffers.sentinel, buffers.stream);
        map = buffers.nonnull_codes->view();
    }
    buffers.gathered = cudf::gather(cudf::table_view {{buffers.all_keys->view()}},
                                    map,
                                    cudf::out_of_bounds_policy::NULLIFY,
                                    cudf::negative_index_policy::NOT_ALLOWED,
                                    buffers.stream,
                                    buffers.mr);
    buffers.result = std::move(buffers.gathered->release().front());
}

} // namespace detail

/**
 * Concatenate string dictionaries into one owned, fully expanded STRING column.
 *
 * Targets libcudf 26.06. Keys may differ, repeat, or be unsorted; empty input returns
 * an empty STRING column. Rejects non-string-dictionary inputs and throws
 * std::overflow_error when aggregate row/key counts exceed cudf::size_type.
 *
 * Valid row codes must index their dictionary. Null-key slots must be folded into
 * parent validity, as in Vortex exports; null-row codes may be arbitrary. Keep
 * inputs alive and establish producer dependencies on `stream` before calling.
 *
 * Fixed-width non-null strings use numeric payload gathers; other strings combine
 * keys and codes before expansion. Neither path sorts or deduplicates keys.
 * Like cuDF, work and temporary deallocation are stream-ordered. The caller must
 * keep inputs alive until `stream` completes, including after an error, and establish
 * dependencies before consuming the result on another stream. Returned buffers use `mr`;
 * temporaries use cuDF's current device resource. Both resources must outlive their
 * allocations and queued work.
 */
inline std::unique_ptr<cudf::column>
concatenate_string_dictionaries(std::vector<cudf::column_view> const &batches,
                                rmm::cuda_stream_view stream = cudf::get_default_stream(),
                                rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) {
    detail::string_dictionary_buffers buffers {batches.size(), stream, mr};
    auto const inputs = detail::validate_string_dictionaries(batches);
    if (inputs.nonempty.empty()) {
        buffers.result = cudf::make_empty_column(cudf::type_id::STRING);
    } else if (inputs.nonempty.size() == 1) {
        buffers.result =
            cudf::dictionary::decode(cudf::dictionary_column_view {inputs.nonempty.front()}, stream, mr);
    } else {
        buffers.all_keys = cudf::concatenate(inputs.keys, stream);
        if (auto const width = detail::fixed_string_width(inputs, buffers)) {
            detail::concatenate_fixed_width_strings(inputs, width, buffers);
        } else {
            detail::concatenate_dictionary_codes(inputs, buffers);
            detail::gather_dictionary_strings(inputs, buffers);
        }
    }
    return std::move(buffers.result);
}

} // namespace vortex::cuda
