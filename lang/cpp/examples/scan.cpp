// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include <vortex/data_source.hpp>
#include <vortex/estimate.hpp>

using vortex::DataSource;
using vortex::Estimate;
using vortex::Scan;
using vortex::Session;

static void print_estimate(const char *what, const Estimate &est) {
    using enum vortex::EstimateType;
    switch (est.type()) {
    case Unknown:
        std::cout << what << ": unknown\n";
        break;
    case Exact:
        std::cout << what << ": " << est.value() << '\n';
        break;
    case Inexact:
        std::cout << what << ": at most " << est.value() << '\n';
        break;
    }
}

struct ScanStats {
    size_t partitions = 0;
    size_t arrays = 0;
    size_t rows = 0;
};

static ScanStats worker(Scan &scan) {
    ScanStats stats;
    while (auto partition = scan.next_partition()) {
        ++stats.partitions;
        while (auto array = partition->next()) {
            ++stats.arrays;
            stats.rows += array->size();
        }
    }
    return stats;
}

bool parse_arguments(int argc, char **argv, size_t &num_threads, std::string_view &files) {
    int opt = 0;
    while ((opt = getopt(argc, argv, "j:")) != -1) {
        switch (opt) {
        case 'j':
            num_threads = static_cast<size_t>(std::atoi(optarg));
            break;
        default:
            std::cerr << "Multi-threaded file scan\nUsage: scan [-j "
                         "threads] <file glob>\n";
            return false;
        }
    }
    if (optind + 1 != argc) {
        std::cerr << "Multi-threaded file scan\nUsage: scan [-j threads] <file "
                     "glob>\n";
        return false;
    }

    files = argv[optind];
    return true;
}

int main(int argc, char **argv) {
    size_t num_threads = 0;
    std::string_view files;
    if (!parse_arguments(argc, argv, num_threads, files)) {
        return 1;
    }
    std::cout << "Opening files: " << files << '\n';

    const Session session;
    const DataSource ds = DataSource::open(session, {files});

    print_estimate("Data source row count", ds.row_count());

    Scan scan = ds.scan();
    print_estimate("Partition count", scan.partition_count());

    if (num_threads == 0) {
        num_threads = scan.partition_count().value_or(1);
    }

    std::cout << "Starting scan, using " << num_threads << " threads\n";
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    std::vector<ScanStats> results(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back([i, &scan, &results] { results[i] = worker(scan); });
    }
    for (auto &t : threads) {
        t.join();
    }

    ScanStats total;
    for (const auto &r : results) {
        total.partitions += r.partitions;
        total.arrays += r.arrays;
        total.rows += r.rows;
    }
    std::cout << "Finished scan, processed " << total.partitions << " partitions, " << total.arrays
              << " arrays, " << total.rows << " rows\n";
    return 0;
}
