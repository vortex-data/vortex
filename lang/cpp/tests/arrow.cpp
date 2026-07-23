// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <nanoarrow/nanoarrow.h>

#include <catch2/catch_test_macros.hpp>
#include <nanoarrow/nanoarrow.hpp>

typedef struct ArrowSchema FFI_ArrowSchema;
typedef struct ArrowArray FFI_ArrowArray;
typedef struct ArrowArrayStream FFI_ArrowArrayStream;
#define USE_OWN_ARROW 1

#include <vortex/data_source.hpp>

#include "common.hpp"

using namespace vortex;
using vortex_test::sample_dtype;
using vortex_test::TempPath;
using vortex_test::write_sample;

namespace {
using enum vortex::PType;

TEST_CASE("dtype to ArrowSchema", "[arrow]") {
    Session session;
    DataType d = sample_dtype();
    ArrowSchema schema = d.to_arrow(session);

    nanoarrow::UniqueSchema unique_schema;
    ArrowSchemaMove(&schema, unique_schema.get());
    REQUIRE(unique_schema->format != nullptr);
    REQUIRE(unique_schema->n_children == 2);
}

TEST_CASE("dtype from ArrowSchema", "[arrow]") {
    nanoarrow::UniqueSchema schema;
    REQUIRE(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_STRUCT) == NANOARROW_OK);
    REQUIRE(ArrowSchemaAllocateChildren(schema.get(), 1) == NANOARROW_OK);
    REQUIRE(ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_INT64) == NANOARROW_OK);
    REQUIRE(ArrowSchemaSetName(schema->children[0], "n") == NANOARROW_OK);

    ArrowSchema raw = {};
    ArrowSchemaMove(schema.get(), &raw);
    Session session;
    DataType d = DataType::from_arrow(session, &raw);
    REQUIRE(d.variant() == DataTypeVariant::Struct);
    const std::vector<StructField> fields = d.fields();
    REQUIRE(fields.size() == 1);
    REQUIRE(fields[0].name == "n");
    REQUIRE(fields[0].dtype.primitive_type() == I64);
}

TEST_CASE("Import Arrow array as Vortex array", "[arrow]") {
    Session session;
    nanoarrow::UniqueSchema schema;
    REQUIRE(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_STRUCT) == NANOARROW_OK);
    REQUIRE(ArrowSchemaAllocateChildren(schema.get(), 1) == NANOARROW_OK);
    REQUIRE(ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_INT32) == NANOARROW_OK);
    REQUIRE(ArrowSchemaSetName(schema->children[0], "a") == NANOARROW_OK);

    nanoarrow::UniqueArray arr;
    REQUIRE(ArrowArrayInitFromSchema(arr.get(), schema.get(), nullptr) == NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(arr.get()) == NANOARROW_OK);
    for (int i : {10, 20, 30}) {
        REQUIRE(ArrowArrayAppendInt(arr->children[0], i) == NANOARROW_OK);
        REQUIRE(ArrowArrayFinishElement(arr.get()) == NANOARROW_OK);
    }
    REQUIRE(ArrowArrayFinishBuildingDefault(arr.get(), nullptr) == NANOARROW_OK);

    ArrowArray raw_arr = {};
    ArrowSchema raw_schema = {};
    ArrowArrayMove(arr.get(), &raw_arr);
    ArrowSchemaMove(schema.get(), &raw_schema);

    Array vx = Array::from_arrow(session, &raw_arr, &raw_schema, false);
    REQUIRE(vx.size() == 3);
    REQUIRE(vx.has_dtype(DataTypeVariant::Struct));

    Array a = vx.field(0);
    REQUIRE(a.is_primitive(I32));
    auto view = a.values<int32_t>(session);
    REQUIRE(view.values()[0] == 10);
    REQUIRE(view.values()[2] == 30);
}

TEST_CASE("Scan partition to ArrowArrayStream", "[arrow]") {
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    Scan scan = ds.scan();
    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());

    ArrowStream vx_stream = std::move(partition.value()).into_arrow_stream();
    nanoarrow::UniqueArrayStream owned;
    ArrowArrayStreamMove(vx_stream.raw(), owned.get());

    nanoarrow::UniqueSchema schema;
    ArrowError err {};
    REQUIRE(ArrowArrayStreamGetSchema(owned.get(), schema.get(), &err) == NANOARROW_OK);
    REQUIRE(schema->n_children == 2);

    size_t rows = 0;
    while (true) {
        nanoarrow::UniqueArray chunk;
        int rc = owned->get_next(owned.get(), chunk.get());
        REQUIRE(rc == NANOARROW_OK);
        if (chunk->release == nullptr) {
            break;
        }
        rows += chunk->length;
    }
    REQUIRE(rows == vortex_test::SAMPLE_ROWS);
}
} // namespace
