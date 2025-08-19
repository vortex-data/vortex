// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
    // 1. bad, dangling lvalue reference!
    // auto& builder = vortex::VortexFile::Open(vortex_file).CreateScanBuilder().WithLimit(3);
    // auto stream = std::move(builder).IntoStream();
    // 2. good, prefer
    auto builder = vortex::VortexFile::Open(vortex_file).CreateScanBuilder();
    builder.WithLimit(3);
    auto stream = std::move(builder).IntoStream();
    // 3. feasible
    // auto stream =
    //     std::move(vortex::VortexFile::Open(vortex_file).CreateScanBuilder().WithLimit(3)).IntoStream();
    nanoarrow::UniqueArray array;
    int get_next_result = stream.get_next(&stream, array.get());
    assert(get_next_result == 0);
    std::cout << "Number of rows: " << array->length << '\n';
    std::cout << "Number of columns in schema: " << array->n_children << '\n';
    return 0;
}