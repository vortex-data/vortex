// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "nanoarrow/hpp/unique.hpp"
#include "vortex/file.hpp"
#include "vortex/scan.hpp"
#include "vortex/thread_pool.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <vortex_file>" << '\n';
        return 1;
    }
    std::string vortex_file = argv[1];
    vortex::ConfigureThreadPool(1);
    auto file = vortex::VortexFile::Open(vortex_file);
    auto scan_builder = file.CreateScanBuilder();
    auto [array, schema] = scan_builder.IntoArray();
    nanoarrow::UniqueArray array_obj;
    ArrowArrayMove(&array, array_obj.get());
    nanoarrow::UniqueSchema schema_obj;
    ArrowSchemaMove(&schema, schema_obj.get());
    std::cout << "Number of rows: " << array_obj->length << '\n';
    std::cout << "Number of columns in schema: " << schema_obj->n_children << '\n';
    return 0;
}