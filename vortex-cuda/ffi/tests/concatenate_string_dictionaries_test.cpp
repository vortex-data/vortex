// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <vortex_cudf_dictionary.hpp>

#include <cudf/copying.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

std::unique_ptr<cudf::column> concatenate_with_defaults(std::vector<cudf::column_view> const &batches);

namespace {
using strings = cudf::test::strings_column_wrapper;
template <typename T>
using numbers = cudf::test::fixed_width_column_wrapper<T>;
using ConcatenateStringDictionaries = cudf::test::BaseFixture;

// Preserve unsorted/duplicate keys; lift code validity to the parent as the exporter
// does. Callers also fold null-key slots into that validity before calling this helper.
std::unique_ptr<cudf::column> dictionary(std::unique_ptr<cudf::column> keys,
                                         std::unique_ptr<cudf::column> indices) {
    auto const size = indices->size();
    auto const type = indices->type();
    auto const null_count = indices->null_count();
    auto contents = indices->release();
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(
        std::make_unique<cudf::column>(type, size, std::move(*contents.data), rmm::device_buffer {}, 0));
    children.push_back(std::move(keys));
    return std::make_unique<cudf::column>(cudf::data_type {cudf::type_id::DICTIONARY32},
                                          size,
                                          rmm::device_buffer {},
                                          std::move(*contents.null_mask),
                                          null_count,
                                          std::move(children));
}

std::unique_ptr<cudf::column> concatenate(std::vector<cudf::column_view> const &batches) {
    return vortex::cuda::concatenate_string_dictionaries(batches, cudf::test::get_default_stream());
}

void expect_values(cudf::column_view actual, cudf::column_view expected) {
    ASSERT_EQ(actual.type().id(), cudf::type_id::STRING);
    // Compare values and validity, allowing an otherwise redundant all-valid mask.
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(actual, expected);
}

/**
 * Check that the caller's resource funds only returned buffers.
 *
 * Total allocations must remain live with the result and be released with it.
 */
void expect_output_only_allocations(std::vector<cudf::column_view> const &batches,
                                    cudf::column_view expected) {
    rmm::mr::statistics_resource_adaptor output_mr {
        cuda::mr::any_resource<cuda::mr::device_accessible> {cudf::get_current_device_resource_ref()}};
    auto const stream = cudf::test::get_default_stream();
    auto result = vortex::cuda::concatenate_string_dictionaries(batches,
                                                                stream,
                                                                rmm::device_async_resource_ref {output_mr});
    stream.synchronize();
    auto const bytes = output_mr.get_bytes_counter();
    EXPECT_GT(bytes.value, 0);
    EXPECT_EQ(bytes.total, bytes.value);
    EXPECT_EQ(bytes.value, static_cast<int64_t>(result->alloc_size()));
    expect_values(result->view(), expected);
    result.reset();
    stream.synchronize();
    EXPECT_EQ(output_mr.get_bytes_counter().value, 0);
}

template <typename T>
class CodeWidths : public cudf::test::BaseFixture {};
using IntegerTypes =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(CodeWidths, IntegerTypes);

TYPED_TEST(CodeWidths, SingleAndDifferentUnsortedDuplicateKeys) {
    auto a = dictionary(strings {"z", "a", "z"}.release(), numbers<TypeParam> {2, 1, 0}.release());
    auto b = dictionary(strings {"b", "z", "a", "b"}.release(), numbers<int32_t> {3, 2, 1, 0}.release());
    strings single {"z", "a", "z"};
    expect_values(concatenate({a->view()})->view(), single);
    strings combined {"z", "a", "z", "b", "a", "z", "b"};
    expect_values(concatenate({a->view(), b->view()})->view(), combined);
}

TYPED_TEST(CodeWidths, VariableWidthKeysWithNonzeroCodeOffset) {
    auto a = dictionary(strings {"b", "zz", "a", "b"}.release(), numbers<int32_t> {3, 2, 1, 0}.release());
    auto b = dictionary(strings {"zz", "a", "zz"}.release(), numbers<TypeParam> {2, 1, 0}.release());
    strings combined {"b", "a", "zz", "b", "zz", "a", "zz"};
    expect_values(concatenate({a->view(), b->view()})->view(), combined);
}

TEST_F(ConcatenateStringDictionaries, NullCodesAndNullKeysUseParentValidity) {
    auto a = dictionary(
        strings {{"z", "unused", "a"}, {true, false, true}}.release(),
        numbers<int64_t> {{int64_t {2}, int64_t {1}, std::numeric_limits<int64_t>::min(), int64_t {0}},
                          {true, false, false, true}}
            .release());
    auto b = dictionary(strings({"unused", "b"}, {false, true}).release(),
                        numbers<uint64_t> {{std::numeric_limits<uint64_t>::max(), uint64_t {0}, uint64_t {1}},
                                           {false, false, true}}
                            .release());
    strings single {{"a", "", "", "z"}, {true, false, false, true}};
    expect_values(concatenate({a->view()})->view(), single);
    strings combined {{"a", "", "", "z", "", "", "b"}, {true, false, false, true, false, false, true}};
    expect_values(concatenate({a->view(), b->view()})->view(), combined);
}

TEST_F(ConcatenateStringDictionaries, SlicesRespectCodeOffsetAndValidity) {
    auto a = dictionary(strings {"z", "a", "z"}.release(),
                        numbers<int32_t> {{0, 1, -100, 2, 0}, {true, true, false, true, true}}.release());
    auto b = dictionary(strings {"b", "c"}.release(), numbers<int8_t> {0, 1, 0}.release());
    auto sa = cudf::slice(a->view(), {1, 4}, cudf::test::get_default_stream()).front();
    auto sb = cudf::slice(b->view(), {1, 3}, cudf::test::get_default_stream()).front();
    strings single {{"a", "", "z"}, {true, false, true}};
    expect_values(concatenate({sa})->view(), single);
    strings combined {{"a", "", "z", "c", "b"}, {true, false, true, true, true}};
    expect_values(concatenate({sa, sb})->view(), combined);
}

TEST_F(ConcatenateStringDictionaries, EmptyInputAndEmptyBatches) {
    auto empty = dictionary(strings {}.release(), numbers<int8_t> {}.release());
    auto unused = dictionary(strings {"unused"}.release(), numbers<uint64_t> {}.release());
    strings expected_empty;
    expect_values(concatenate({})->view(), expected_empty);
    expect_values(concatenate({empty->view()})->view(), expected_empty);
    expect_values(concatenate({empty->view(), unused->view()})->view(), expected_empty);
    auto a = dictionary(strings {"z", "a"}.release(), numbers<int32_t> {1, 0}.release());
    auto b = dictionary(strings {"b"}.release(), numbers<int16_t> {0}.release());
    strings single {"a", "z"};
    expect_values(concatenate({unused->view(), a->view(), empty->view()})->view(), single);
    strings combined {"a", "z", "b"};
    expect_values(concatenate({empty->view(), a->view(), unused->view(), b->view()})->view(), combined);
}

TEST_F(ConcatenateStringDictionaries, AllNullIncludingZeroKeys) {
    auto a = dictionary(strings {}.release(), numbers<int32_t> {{-1, 1000}, {false, false}}.release());
    auto b = dictionary(strings {"unused"}.release(), numbers<int8_t> {{-127}, {false}}.release());
    strings single({"", ""}, {false, false});
    expect_values(concatenate({a->view()})->view(), single);
    strings combined {{"", "", ""}, {false, false, false}};
    expect_values(concatenate({a->view(), b->view()})->view(), combined);
    strings zero_keys {{"", "", "", ""}, {false, false, false, false}};
    expect_values(concatenate({a->view(), a->view()})->view(), zero_keys);
}

TEST_F(ConcatenateStringDictionaries, FixedWidthPayloadsAndVariableWidthFallback) {
    for (auto const &values : std::vector<std::vector<std::string>> {{"z", std::string {"\0", 1}},
                                                                     {"é", "ab"},
                                                                     {"🙂", "abcd"},
                                                                     {"🙂🙂", "abcdefgh"},
                                                                     {"abc", "雪"},
                                                                     {"", "abcd"},
                                                                     {"", ""}}) {
        auto const &x = values[0];
        auto const &y = values[1];
        auto a = dictionary(strings {y, x, y}.release(), numbers<int8_t> {0, 1, 0, 1}.release());
        auto b = dictionary(strings {x, y}.release(), numbers<int64_t> {1, 0, 1}.release());
        auto slice = cudf::slice(a->view(), {1, 3}, cudf::test::get_default_stream()).front();
        strings expected {x, y, y, x, y};
        expect_values(concatenate({slice, b->view()})->view(), expected);
    }
}

TEST_F(ConcatenateStringDictionaries, LongUtf8AndEmbeddedNul) {
    std::string long_value;
    for (int i = 0; i < 8192; ++i) {
        long_value += "雪🙂é";
    }
    std::string embedded {"a\0b", 3};
    auto a = dictionary(strings {long_value, "é", long_value, embedded}.release(),
                        numbers<int32_t> {2, 1, 3}.release());
    auto b = dictionary(strings {"🙂", long_value, ""}.release(), numbers<int16_t> {1, 0, 2}.release());
    strings expected {long_value, "é", embedded, long_value, "🙂", ""};
    expect_values(concatenate({a->view(), b->view()})->view(), expected);
}

TEST_F(ConcatenateStringDictionaries, OwnedOutputAndDefaultsAcrossTranslationUnits) {
    for (bool multiple : {false, true}) {
        std::unique_ptr<cudf::column> result;
        {
            auto a = dictionary(strings {"z", "a", "z"}.release(), numbers<int32_t> {2, 1}.release());
            auto b = dictionary(strings {"b"}.release(), numbers<int8_t> {0}.release());
            cudf::test::get_default_stream().synchronize();
            std::vector<cudf::column_view> batches {a->view()};
            if (multiple) {
                batches.push_back(b->view());
            }
            result = concatenate_with_defaults(batches);
            cudf::get_default_stream().synchronize();
        }
        strings single {"z", "a"};
        strings combined {"z", "a", "b"};
        expect_values(result->view(), multiple ? cudf::column_view(combined) : cudf::column_view(single));
    }
}

TEST_F(ConcatenateStringDictionaries, RejectsPlainMixedNumericNestedAndNonIntegerCodes) {
    strings plain {"a"};
    auto valid = dictionary(strings {"a"}.release(), numbers<int8_t> {0}.release());
    auto numeric = dictionary(numbers<int32_t> {1}.release(), numbers<int8_t> {0}.release());
    auto empty_numeric = dictionary(numbers<int32_t> {}.release(), numbers<int8_t> {}.release());
    auto boolean = dictionary(strings {"a"}.release(), numbers<bool> {false}.release());
    auto floating = dictionary(strings {"a"}.release(), numbers<float> {0}.release());
    cudf::test::structs_column_wrapper nested {{plain}};
    for (cudf::column_view invalid : std::vector<cudf::column_view> {plain,
                                                                     numeric->view(),
                                                                     empty_numeric->view(),
                                                                     boolean->view(),
                                                                     floating->view(),
                                                                     nested}) {
        EXPECT_THROW(concatenate({invalid}), std::invalid_argument);
        EXPECT_THROW(concatenate({valid->view(), invalid}), std::invalid_argument);
        EXPECT_THROW(concatenate({invalid, valid->view()}), std::invalid_argument);
    }
}

TEST_F(ConcatenateStringDictionaries, PreflightOverflowAndIgnoredEmptyBatchKeys) {
    auto one = dictionary(strings {"key"}.release(), numbers<int32_t> {0}.release());
    auto const source = one->view();
    auto metadata = [&](cudf::size_type rows, cudf::size_type keys) {
        // Inflated views must never reach device operations: only overflow preflight,
        // or ignored keys belonging to a zero-row batch, may inspect this metadata.
        auto const k = source.child(1);
        cudf::column_view codes {source.child(0).type(), rows, source.child(0).head<void>(), nullptr, 0};
        cudf::column_view inflated {k.type(),
                                    keys,
                                    k.head<void>(),
                                    nullptr,
                                    0,
                                    0,
                                    std::vector<cudf::column_view>(k.child_begin(), k.child_end())};
        return cudf::column_view {source.type(), rows, nullptr, nullptr, 0, 0, {codes, inflated}};
    };
    auto const max = std::numeric_limits<cudf::size_type>::max();
    EXPECT_THROW(concatenate({metadata(max, 1), source}), std::overflow_error);
    EXPECT_THROW(concatenate({metadata(1, max), source}), std::overflow_error);
    strings empty;
    expect_values(concatenate({metadata(0, max), metadata(0, max)})->view(), empty);
    strings single {"key"};
    expect_values(concatenate({metadata(0, max), source})->view(), single);
    strings combined {"key", "key"};
    expect_values(concatenate({source, metadata(0, max), source})->view(), combined);
}

TEST_F(ConcatenateStringDictionaries, OutputResourceForFixedWidthStrings) {
    auto a = dictionary(strings {"z", "a"}.release(), numbers<int8_t> {1, 0}.release());
    auto b = dictionary(strings {"b"}.release(), numbers<int16_t> {0}.release());
    strings expected {"a", "z", "b"};
    expect_output_only_allocations({a->view(), b->view()}, expected);
}

TEST_F(ConcatenateStringDictionaries, OutputResourceForVariableWidthStrings) {
    auto a = dictionary(strings {"zz", "a"}.release(), numbers<int8_t> {1, 0}.release());
    auto b = dictionary(strings {"bbb"}.release(), numbers<int16_t> {0}.release());
    strings expected {"a", "zz", "bbb"};
    expect_output_only_allocations({a->view(), b->view()}, expected);
}

TEST_F(ConcatenateStringDictionaries, OutputResourceForNullableStrings) {
    auto a = dictionary(strings {"zz", "a"}.release(), numbers<int8_t> {{1, 0}, {true, false}}.release());
    auto b = dictionary(strings {"bbb"}.release(), numbers<int16_t> {0}.release());
    strings expected({"a", "", "bbb"}, {true, false, true});
    expect_output_only_allocations({a->view(), b->view()}, expected);
}

TEST_F(ConcatenateStringDictionaries, NonDefaultConsumerStream) {
    rmm::cuda_stream stream;
    auto a = dictionary(strings {"z", "a"}.release(), numbers<int8_t> {1, 0}.release());
    auto b = dictionary(strings {"b"}.release(), numbers<int16_t> {0}.release());
    // Producer completion is the caller's responsibility, including with explicit streams.
    cudf::test::get_default_stream().synchronize();
    auto result = vortex::cuda::concatenate_string_dictionaries({a->view(), b->view()}, stream.view(), mr());
    stream.synchronize();
    strings expected {"a", "z", "b"};
    expect_values(result->view(), expected);

    EXPECT_THROW(vortex::cuda::concatenate_string_dictionaries({a->view(), expected}, stream.view(), mr()),
                 std::invalid_argument);
    stream.synchronize();
}

} // namespace
