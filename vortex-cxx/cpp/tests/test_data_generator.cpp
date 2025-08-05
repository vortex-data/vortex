// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "test_data_generator.hpp"
#include <cstddef>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/hpp/array_stream.hpp>
#include <vector>
#include <cstdlib>

namespace vortex {
namespace testing {

    ArrowArrayStream CreateTestDataStream() {
        // Create a simple two-column struct with int32 data
        // Schema: struct{a: int32, b: int32}
        nanoarrow::UniqueSchema schema;
        ArrowSchemaInit(schema.get());
        NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(schema.get(), NANOARROW_TYPE_STRUCT));
        NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateChildren(schema.get(), 2));
        ArrowSchemaInit(schema->children[0]);
        ArrowSchemaInit(schema->children[1]);
        NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[0], "a"));
        NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT32));
        NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[1], "b"));
        NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(schema->children[1], NANOARROW_TYPE_INT32));

        // Create arrays for each field
        nanoarrow::UniqueArray field_a, field_b;
        NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(field_a.get(), NANOARROW_TYPE_INT32));
        NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(field_b.get(), NANOARROW_TYPE_INT32));

        // Reserve for 5 elements
        NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(field_a.get()));
        NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(field_b.get()));

        // Add data: [10, 20, 30, 40, 50]
        std::vector<int32_t> data = {10, 20, 30, 40, 50};
        for (int32_t value : data) {
            NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(field_a.get(), value));
            NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(field_b.get(), value));
        }

        NANOARROW_THROW_NOT_OK(
            ArrowArrayFinishBuilding(field_a.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));
        NANOARROW_THROW_NOT_OK(
            ArrowArrayFinishBuilding(field_b.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

        // Create struct array
        nanoarrow::UniqueArray struct_array;
        NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(struct_array.get(), NANOARROW_TYPE_STRUCT));
        NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(struct_array.get(), 2));
        struct_array->length = 5;
        ArrowArrayMove(field_a.get(), struct_array->children[0]);
        ArrowArrayMove(field_b.get(), struct_array->children[1]);

        // Create vector and move array into it
        std::vector<nanoarrow::UniqueArray> arrays;
        arrays.push_back(std::move(struct_array));

        // Create stream
        ArrowArrayStream stream;
        nanoarrow::VectorArrayStream vector_stream(schema.get(), std::move(arrays));
        vector_stream.ToArrayStream(&stream);

        return stream;
    }

    ArrowArrayStream CreateTestData1MStream() {
        constexpr size_t NUM_ROWS = 1024UL * 1024;

        // Create schema: struct{id: int64, value: int32}
        nanoarrow::UniqueSchema schema;
        ArrowSchemaInit(schema.get());
        NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(schema.get(), NANOARROW_TYPE_STRUCT));
        NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateChildren(schema.get(), 2));
        ArrowSchemaInit(schema->children[0]);
        ArrowSchemaInit(schema->children[1]);
        NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[0], "id"));
        NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(schema->children[0], NANOARROW_TYPE_INT64));
        NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[1], "value"));
        NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(schema->children[1], NANOARROW_TYPE_INT32));

        // Create arrays for each field
        nanoarrow::UniqueArray id_field, value_field;
        NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(id_field.get(), NANOARROW_TYPE_INT64));
        NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(value_field.get(), NANOARROW_TYPE_INT32));

        // Reserve space
        NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(id_field.get()));
        NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(value_field.get()));

        // Add data
        for (size_t i = 0; i < NUM_ROWS; ++i) {
            NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(id_field.get(), static_cast<int64_t>(i)));
            NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(value_field.get(), static_cast<int32_t>(i * 2)));
        }

        NANOARROW_THROW_NOT_OK(
            ArrowArrayFinishBuilding(id_field.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));
        NANOARROW_THROW_NOT_OK(
            ArrowArrayFinishBuilding(value_field.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

        // Create struct array
        nanoarrow::UniqueArray struct_array;
        NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(struct_array.get(), NANOARROW_TYPE_STRUCT));
        NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(struct_array.get(), 2));
        struct_array->length = NUM_ROWS;
        ArrowArrayMove(id_field.get(), struct_array->children[0]);
        ArrowArrayMove(value_field.get(), struct_array->children[1]);

        // Create vector and move array into it
        std::vector<nanoarrow::UniqueArray> arrays;
        arrays.push_back(std::move(struct_array));

        // Create stream
        ArrowArrayStream stream;
        nanoarrow::VectorArrayStream vector_stream(schema.get(), std::move(arrays));
        vector_stream.ToArrayStream(&stream);

        return stream;
    }

} // namespace testing
} // namespace vortex