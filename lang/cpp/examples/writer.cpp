// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "vortex/array.hpp"
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <vortex/dtype.hpp>
#include <vortex/writer.hpp>

using namespace vortex;
using namespace expr::ops;
using dtype::Nullable;
using enum ValidityType;

int main(int argc, char **argv) {
    if (argc != 2) {
        return 1;
    }

    const Session session;
    const DataType dtype = dtype::struct_({
        {"age", dtype::uint8()},
        {"height", dtype::uint16(Nullable)},
    });

    constexpr size_t SAMPLE_ROWS = 100;
    std::vector<uint8_t> age_buffer(SAMPLE_ROWS);
    std::vector<uint16_t> height_buffer(SAMPLE_ROWS);
    for (size_t i = 0; i < SAMPLE_ROWS; ++i) {
        age_buffer[i] = static_cast<uint8_t>(i);
        height_buffer[i] = static_cast<uint16_t>((i + 1) % 200);
    }

    Array age = Array::primitive<uint8_t>(age_buffer);
    Array array = make_struct({
        {"age", age},
        {"height", Array::primitive<uint16_t>(height_buffer, AllValid)},
    });

    Expression age_gt_10 = expr::col("age") > expr::lit<uint8_t>(10);
    Array validity_array = array.apply(age_gt_10);

    const Validity validity = Validity::from_array(validity_array);
    Array array2 = make_struct({
        {"age", age},
        {"height", Array::primitive<uint16_t>(height_buffer, validity)},
    });

    Writer writer = Writer::open(session, argv[1], dtype);
    writer.push({array, array2});
    writer.finish();

    return 0;
}
