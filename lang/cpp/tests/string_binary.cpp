// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <nanoarrow/nanoarrow.h>

#include <catch2/catch_test_macros.hpp>
#include <nanoarrow/nanoarrow.hpp>

typedef struct ArrowSchema FFI_ArrowSchema;
typedef struct ArrowArray FFI_ArrowArray;
typedef struct ArrowArrayStream FFI_ArrowArrayStream;
#define USE_OWN_ARROW 1

#include <string>
#include <string_view>
#include <vector>
#include <vortex/data_source.hpp>

#include "common.hpp"

using namespace vortex;
using namespace std::string_view_literals;
using vortex_test::TempPath;

namespace {

Array strings_from_arrow(std::span<const std::string_view> values, bool with_null) {
    nanoarrow::UniqueSchema schema;
    REQUIRE(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_STRING) == NANOARROW_OK);
    nanoarrow::UniqueArray arr;
    REQUIRE(ArrowArrayInitFromSchema(arr.get(), schema.get(), nullptr) == NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(arr.get()) == NANOARROW_OK);
    for (const auto &value : values) {
        const ArrowStringView view {value.data(), static_cast<int64_t>(value.size())};
        REQUIRE(ArrowArrayAppendString(arr.get(), view) == NANOARROW_OK);
    }
    if (with_null) {
        REQUIRE(ArrowArrayAppendNull(arr.get(), 1) == NANOARROW_OK);
    }
    REQUIRE(ArrowArrayFinishBuildingDefault(arr.get(), nullptr) == NANOARROW_OK);

    ArrowArray raw_arr = {};
    ArrowSchema raw_schema = {};
    ArrowArrayMove(arr.get(), &raw_arr);
    ArrowSchemaMove(schema.get(), &raw_schema);
    return Array::from_arrow(Session(), &raw_arr, &raw_schema, true);
}

Array bytes_from_arrow(std::span<const std::string_view> values) {
    nanoarrow::UniqueSchema schema;
    REQUIRE(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_BINARY) == NANOARROW_OK);
    nanoarrow::UniqueArray arr;
    REQUIRE(ArrowArrayInitFromSchema(arr.get(), schema.get(), nullptr) == NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(arr.get()) == NANOARROW_OK);
    for (const auto &value : values) {
        const ArrowBufferView view {{reinterpret_cast<const uint8_t *>(value.data())},
                                    static_cast<int64_t>(value.size())};
        REQUIRE(ArrowArrayAppendBytes(arr.get(), view) == NANOARROW_OK);
    }
    REQUIRE(ArrowArrayFinishBuildingDefault(arr.get(), nullptr) == NANOARROW_OK);

    ArrowArray raw_arr = {};
    ArrowSchema raw_schema = {};
    ArrowArrayMove(arr.get(), &raw_arr);
    ArrowSchemaMove(schema.get(), &raw_schema);
    return Array::from_arrow(Session(), &raw_arr, &raw_schema, true);
}

TEST_CASE("String view over utf8 array", "[strings]") {
    Session session;

    const std::string long1(40, 'x');
    const std::vector<std::string_view> values = {"short"sv, "Широкая строка"sv, long1, ""sv};
    Array array = strings_from_arrow(values, true);

    StringView view = array.strings(session);
    REQUIRE(view.size() == values.size() + 1);
    for (size_t i = 0; i < values.size(); ++i) {
        REQUIRE_FALSE(view.is_null(i));
        REQUIRE(view[i] == values[i]);
    }
    REQUIRE(view.is_null(values.size()));
    REQUIRE_THROWS_AS(view[view.size()], VortexException);
}

TEST_CASE("Bytes view over binary array", "[strings]") {
    Session session;

    // Includes a NUL byte
    const std::vector<std::string_view> values = {std::string_view {"abc\0def", 7}, "ffff"sv};
    Array array = bytes_from_arrow(values);

    BytesView view = array.bytes(session);
    REQUIRE(view.size() == 2);
    for (size_t i = 0; i < values.size(); ++i) {
        const auto bytes = view[i];
        REQUIRE(std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()) == values[i]);
    }
}

TEST_CASE("Strings roundtrip", "[strings]") {
    Session session;
    TempPath path = TempPath::unique();

    const std::string long1(64, 'y');
    const std::vector<std::string_view> values = {"inlined"sv, long1};
    Array strings = strings_from_arrow(values, false);

    Writer writer =
        Writer::open(session, path.string(), dtype::struct_({{"s", dtype::utf8(dtype::Nullable)}}));
    writer.push(make_struct({{"s", strings}}));
    writer.finish();

    DataSource ds = DataSource::open(session, {path.string()});
    Scan scan = ds.scan();
    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());
    auto batch = partition->next();
    REQUIRE(batch.has_value());

    StringView view = batch->field(0).strings(session);
    REQUIRE(view.size() == 2);
    REQUIRE(view[0] == "inlined"sv);
    REQUIRE(view[1] == long1);
}

TEST_CASE("strings() on a non-utf8 array", "[strings]") {
    Session session;
    std::vector<int32_t> data = {1};
    Array a = Array::primitive<int32_t>(data);
    REQUIRE_THROWS_AS(a.strings(session), VortexException);
    REQUIRE_THROWS_AS(a.bytes(session), VortexException);
}
} // namespace
