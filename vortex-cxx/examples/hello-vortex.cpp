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
    auto file = vortex::VortexFile::Open(vortex_file);
    auto stream = file.CreateScanBuilder().WithLimit(100).IntoStream();
    nanoarrow::UniqueArray array;
    int get_next_result = stream.get_next(&stream, array.get());
    assert(get_next_result == 0);
    std::cout << "Number of rows: " << array->length << '\n';
    std::cout << "Number of columns in schema: " << array->n_children << '\n';
    return 0;
}