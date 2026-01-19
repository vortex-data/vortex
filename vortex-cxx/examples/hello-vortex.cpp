// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "nanoarrow/common/inline_types.h"
#include "nanoarrow/hpp/unique.hpp"
#include "nanoarrow/nanoarrow.hpp"
#include "vortex/file.hpp"
#include "vortex/scan.hpp"
#include "vortex/write_options.hpp"
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <vector>

/// Create test data with structure {a: [10, 20, 30, 40, 50], b: [100, 200, 300, 400, 500]}
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

    // Add data: a=[10, 20, 30, 40, 50], b=[100, 200, 300, 400, 500]
    std::vector<int32_t> data_a = {10, 20, 30, 40, 50};
    std::vector<int32_t> data_b = {100, 200, 300, 400, 500};
    for (size_t i = 0; i < data_a.size(); ++i) {
        NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(field_a.get(), data_a[i]));
        NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(field_b.get(), data_b[i]));
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

int main() {
    // Create a temporary file path
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::string vortex_file = (temp_dir / "hello_vortex_example.vortex").string();

    std::cout << "=== Vortex C++ Example ===" << '\n';
    std::cout << "Writing to: " << vortex_file << '\n';

    // Write test data to a Vortex file
    {
        auto stream = CreateTestDataStream();
        vortex::VortexWriteOptions write_options;
        write_options.WriteArrayStream(stream, vortex_file);
        std::cout << "Wrote test data to file" << '\n';
    }

    auto check_stream = [](ArrowArrayStream &stream) {
        nanoarrow::UniqueArray array;
        int get_next_result = stream.get_next(&stream, array.get());
        assert(get_next_result == 0);
        std::cout << "Number of rows: " << array->length << '\n';
        std::cout << "Number of columns in schema: " << array->n_children << '\n';
    };

    // 1. Classic C++ builder pattern
    std::cout << "\n1. Classic C++ builder pattern:" << '\n';
    {
        auto builder = vortex::VortexFile::Open(vortex_file).CreateScanBuilder();
        builder.WithLimit(3);
        auto stream = std::move(builder).IntoStream();
        check_stream(stream);
    }
    // 2. One-line Rusty function chain
    std::cout << "\n2. One-line Rusty function chain:" << '\n';
    {
        auto stream = vortex::VortexFile::Open(vortex_file).CreateScanBuilder().WithLimit(3).IntoStream();
        check_stream(stream);
    }
    // 3. Conditionally set the builder
    std::cout << "\n3. Conditionally set the builder:" << '\n';
    {
        auto limit = 1;
        auto builder = vortex::VortexFile::Open(vortex_file).CreateScanBuilder();
        if (limit > 0) {
            // prefer C++ way
            builder.WithLimit(1);
            // Rusty way is Ok, but you have to move the builder to an rvalue.
            // builder = std::move(builder).WithLimit(3);
        }
        auto stream = std::move(builder).IntoStream();
        check_stream(stream);
    }

    // Clean up
    std::filesystem::remove(vortex_file);
    std::cout << "\nCleaned up temporary file" << '\n';

    return 0;
}