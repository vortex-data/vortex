// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <iostream>

#include <vortex/data_source.hpp>
#include <vortex/writer.hpp>

using namespace std::string_view_literals;
using namespace vortex;
using namespace expr;
using namespace ops; // overloaded >= for Expressions
namespace fs = std::filesystem;

int main() {
    // docs:begin:example
    const Session session;
    const DataSource ds = DataSource::open(session, {"people*.vortex", "me.vortex"});
    Scan scan = ds.scan({.filter = col("height") >= lit<uint16_t>(50)});

    for (Partition &partition : scan.partitions()) {
        for (Array &array : partition.batches()) {
            const Array age = array.field("age");
            const PrimitiveView<uint8_t> age_view = age.values<uint8_t>(session);
            const std::span<const uint8_t> age_values = age_view.values();
            for (uint8_t value : age_values) {
                std::cout << int(value) << " ";
            }
        }
    }
    std::cout << "\n";
    // docs:end:example

    return 0;
}
