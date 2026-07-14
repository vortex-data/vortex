// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <fstream>
#include <thread>
#include <vector>
#include <vortex/scan.hpp>

#include "common.hpp"
#include "vortex/estimate.hpp"

using namespace vortex;
using Catch::Matchers::ContainsSubstring;
using vortex_test::SAMPLE_ROWS;
using vortex_test::TempPath;
using vortex_test::write_sample;

namespace {
using enum vortex::PType;

TEST_CASE("Non-existent data source", "[datasource]") {
    Session session;

    try {
        DataSource::open(session, {"nonexistent"});
        FAIL("expected exception");
    } catch (const VortexException &e) {
        REQUIRE_THAT(std::string(e.what()), ContainsSubstring("No files matched"));
    }
}

TEST_CASE("Read dtype and row count", "[datasource]") {
    Session session;
    TempPath path = write_sample(session);

    std::array<std::string, 1> paths = {path.string()};

    DataSource other = DataSource::open(session, paths);
    DataSource ds(other);

    Estimate row_count = ds.row_count();
    REQUIRE(row_count.type() == EstimateType::Exact);
    REQUIRE(row_count.value() == SAMPLE_ROWS);

    DataType dt = ds.dtype();
    REQUIRE(dt.variant() == DataTypeVariant::Struct);
    const std::vector<StructField> fields = dt.fields();
    REQUIRE(fields.size() == 2);
    REQUIRE(fields[0].name == "age");
    REQUIRE(fields[1].name == "height");
}

void verify_age_field(const Session &session,
                      const Array &age,
                      uint8_t first = 0,
                      size_t rows = SAMPLE_ROWS) {
    REQUIRE(age.is_primitive(U8));
    REQUIRE(age.size() == rows);
    auto view = age.values<uint8_t>(session);
    for (size_t i = 0; i < rows; ++i) {
        REQUIRE(view.values()[i] == static_cast<uint8_t>(first + i));
    }
}

void verify_sample_array(const Session &session, const Array &array) {
    REQUIRE(array.size() == SAMPLE_ROWS);
    REQUIRE(array.has_dtype(DataTypeVariant::Struct));
    verify_age_field(session, array.field(0));

    Array height = array.field(1);
    REQUIRE(height.is_primitive(U16));
    auto view = height.values<uint16_t>(session);
    for (size_t i = 0; i < SAMPLE_ROWS; ++i) {
        REQUIRE(view.values()[i] == (i + 1) % 200);
    }

    REQUIRE_THROWS_AS(array.field(2), VortexException);
}

TEST_CASE("Basic scan", "[scan]") {
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    Scan scan = ds.scan();
    REQUIRE(scan.partition_count().type() == EstimateType::Exact);
    REQUIRE(scan.partition_count().value() == 1);
    REQUIRE(scan.dtype().variant() == DataTypeVariant::Struct);

    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());

    Estimate rows = partition->row_count();
    REQUIRE(rows.type() == EstimateType::Exact);
    REQUIRE(rows.value() == SAMPLE_ROWS);

    REQUIRE_FALSE(scan.next_partition().has_value());

    auto array = partition->next();
    REQUIRE(array.has_value());
    REQUIRE_FALSE(partition->next().has_value());

    verify_sample_array(session, *array);
}

TEST_CASE("Range-for over partitions and batches", "[scan]") {
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    Scan scan = ds.scan();
    size_t partitions = 0;
    size_t rows = 0;
    for (Partition &partition : scan.partitions()) {
        ++partitions;
        for (Array &batch : partition.batches()) {
            rows += batch.size();
        }
    }
    REQUIRE(partitions == 1);
    REQUIRE(rows == SAMPLE_ROWS);
}

TEST_CASE("Scan from buffer", "[scan]") {
    Session session;
    TempPath path = write_sample(session);

    std::ifstream file(path.path(), std::ios::binary | std::ios::ate);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> buffer(static_cast<size_t>(size));
    REQUIRE(file.read(reinterpret_cast<char *>(buffer.data()), size));

    DataSource ds = DataSource::from_buffer(session, buffer);
    Scan scan = ds.scan();
    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());
    auto array = partition->next();
    REQUIRE(array.has_value());
    verify_sample_array(session, *array);
}

TEST_CASE("Multiple paths", "[scan]") {
    Session session;
    TempPath first = write_sample(session);
    TempPath second = write_sample(session);

    const std::vector<std::string> paths = {first.string(), second.string()};
    DataSource ds = DataSource::open(session, paths);
    REQUIRE(ds.row_count().value_or(0) >= SAMPLE_ROWS);
}

TEST_CASE("Multithreaded scan", "[scan]") {
    Session session;

    constexpr size_t NUM_FILES = 8;
    std::vector<TempPath> paths;
    std::vector<std::string> path_strings;
    for (size_t i = 0; i < NUM_FILES; ++i) {
        paths.push_back(write_sample(session));
        path_strings.push_back(paths.back().string());
    }

    DataSource ds = DataSource::open(session, path_strings);
    Scan scan = ds.scan();

    std::vector<std::thread> threads;
    threads.reserve(NUM_FILES);
    std::vector<size_t> rows(NUM_FILES, 0);

    for (size_t i = 0; i < NUM_FILES; ++i) {
        threads.emplace_back([&, i] {
            while (auto partition = scan.next_partition()) {
                while (auto batch = partition->next()) {
                    rows[i] += batch->size();
                }
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }

    size_t total = 0;
    for (size_t n : rows) {
        total += n;
    }
    REQUIRE(total == NUM_FILES * SAMPLE_ROWS);
}

TEST_CASE("Project field", "[projection]") {
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    Scan scan = ds.scan({.projection = expr::col("age")});
    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());
    auto array = partition->next();
    REQUIRE(array.has_value());
    verify_age_field(session, *array);
}

TEST_CASE("Project none fields", "[projection]") {
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    Scan scan = ds.scan({.projection = expr::root().select({})});
    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());
    auto array = partition->next();
    REQUIRE(array.has_value());
    REQUIRE(array->dtype().fields().empty());
}

TEST_CASE("Project multiple fields", "[projection]") {
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    Scan scan = ds.scan({.projection = expr::root().select({"age", "height"})});
    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());
    auto array = partition->next();
    REQUIRE(array.has_value());
    verify_age_field(session, array->field("age"));
}

TEST_CASE("Filter age", "[filter]") {
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    constexpr uint8_t threshold = 50;
    Scan scan = ds.scan({
        .filter = expr::gte(expr::col("age"), expr::lit<uint8_t>(threshold)),
    });
    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());
    auto array = partition->next();
    REQUIRE(array.has_value());

    REQUIRE(array->size() == SAMPLE_ROWS - threshold);
    verify_age_field(session, array->field(0), threshold, SAMPLE_ROWS - threshold);
}

TEST_CASE("Filter invalid values", "[filter]") {
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    Scan scan = ds.scan({.filter = expr::col("age").is_null()});
    auto partition = scan.next_partition();
    REQUIRE(!partition.has_value());
}

TEST_CASE("Filter with operators", "[filter]") {
    using namespace vortex::expr::ops;
    Session session;
    TempPath path = write_sample(session);
    const std::string path_string = path.string();
    std::array<std::string_view, 1> paths = {path_string};
    DataSource ds = DataSource::open(session, paths);

    Scan scan1 = ds.scan({
        .filter = expr::col("age") >= expr::lit<uint8_t>(90) && expr::col("age") < expr::lit<uint8_t>(95),
    });

    Scan scan2 = ds.scan({
        .filter = !(expr::col("age") < expr::lit<uint8_t>(90)) && expr::col("age") < expr::lit<uint8_t>(95),
    });

    Scan scans[2] = {std::move(scan1), std::move(scan2)};
    for (Scan &scan : scans) {
        auto partition = scan.next_partition();
        REQUIRE(partition.has_value());
        auto array = partition->next();
        REQUIRE(array.has_value());
        REQUIRE(array->size() == 5);
    }
}

TEST_CASE("Type-mismatched filter", "[filter]") {
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    Scan scan = ds.scan({
        .filter = expr::eq(expr::col("age"), expr::lit<int32_t>(67)),
    });
    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());
    REQUIRE_THROWS_AS(partition->next(), VortexException);
}

TEST_CASE("Row range and limit", "[scan]") {
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    Scan scan = ds.scan({
        .row_range = RowRange {10, 25},
        .limit = 5,
    });
    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());
    auto array = partition->next();
    REQUIRE(array.has_value());

    REQUIRE(array->size() == 5);
    verify_age_field(session, array->field(0), 10, 5);
}

TEST_CASE("Selection", "[scan]") {
    using enum Selection::Kind;
    Session session;
    TempPath path = write_sample(session);
    DataSource ds = DataSource::open(session, {path.string()});

    Scan scan = ds.scan({
        .selection = {{Include, {3, 5, 8}}},
    });
    auto partition = scan.next_partition();
    REQUIRE(partition.has_value());
    auto array = partition->next();
    REQUIRE(array.has_value());
    REQUIRE(array->size() == 3);

    auto ages = array->field(0).values<uint8_t>(session);
    REQUIRE(ages.values()[0] == 3);
    REQUIRE(ages.values()[1] == 5);
    REQUIRE(ages.values()[2] == 8);

    scan = ds.scan({
        .selection = {{Exclude, {0, 1, 2}}},
    });
    partition = scan.next_partition();
    REQUIRE(partition.has_value());
    array = partition->next();
    REQUIRE(array.has_value());
    REQUIRE(array->size() == 97);

    ages = array->field(0).values<uint8_t>(session);
    REQUIRE(ages.values()[0] == 3);
}
} // namespace
