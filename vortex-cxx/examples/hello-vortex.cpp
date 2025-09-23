// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "nanoarrow/common/inline_types.h"
#include "nanoarrow/hpp/unique.hpp"
#include "vortex/file.hpp"
#include "vortex/scan.hpp"
#include <cassert>
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <vortex_file>" << '\n';
        return 1;
    }
    std::string vortex_file = argv[1];
    auto check_stream = [](ArrowArrayStream &stream) {
        nanoarrow::UniqueArray array;
        int get_next_result = stream.get_next(&stream, array.get());
        assert(get_next_result == 0);
        std::cout << "Number of rows: " << array->length << '\n';
        std::cout << "Number of columns in schema: " << array->n_children << '\n';
    };
    // 1. Classic C++ builder pattern
    {
        auto builder = vortex::VortexFile::Open(vortex_file).CreateScanBuilder();
        builder.WithLimit(3);
        auto stream = std::move(builder).IntoStream();
        check_stream(stream);
    }
    // 2. One-line Rusty function chain
    {
        auto stream = vortex::VortexFile::Open(vortex_file).CreateScanBuilder().WithLimit(3).IntoStream();
        check_stream(stream);
    }
    // 3. Conditionally set the builder
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
    return 0;
}