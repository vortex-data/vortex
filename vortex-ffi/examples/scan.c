// SPDX-License-Identifier: CC-BY-4.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "vortex.h"
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_THREADS 64

const char *usage = "Multi-threaded file scan\n"
                    "Usage: scan [-j threads] <file glob>\n";

void print_estimate(const char *what, const vx_estimate *estimate) {
    switch (estimate->type) {
    case VX_ESTIMATE_UNKNOWN:
        printf("%s: unknown\n", what);
        return;
    case VX_ESTIMATE_EXACT:
        printf("%s: %lu\n", what, estimate->estimate);
        return;
    case VX_ESTIMATE_INEXACT:
        printf("%s: approximately %lu\n", what, estimate->estimate);
        break;
    }
}

void print_error(const char *what, const vx_error *error) {
    const vx_string *str = vx_error_get_message(error);
    fprintf(stderr, "%s: %.*s\n", what, (int)vx_string_len(str), vx_string_ptr(str));
}

struct scan_thread_info {
    pthread_t thread_id;
    pthread_mutex_t *mutex;
    vx_scan *scan;
    size_t partitions, arrays, rows;
    vx_error *error;
};

void *execute_scan_thread(void *arg) {
    struct scan_thread_info *info = arg;
    while (true) {
        // A partition is an independent unit of work a thread can work on.
        pthread_mutex_lock(info->mutex);
        vx_partition *partition = vx_scan_next_partition(info->scan, &info->error);
        pthread_mutex_unlock(info->mutex);

        if (partition == NULL && info->error == NULL) {
            break; // partition iterator exhausted
        }
        if (partition == NULL && info->error != NULL) {
            return NULL; // partition was not scanned due to an error
        }
        ++info->partitions;

        vx_estimate row_count;
        if (vx_partition_row_count(partition, &row_count, &info->error)) {
            vx_partition_free(partition);
            return NULL;
        }

        printf("Thread %lu processing partition %lu, ", info->thread_id + 1, info->partitions);
        print_estimate("row count", &row_count);

        // An array is a batch of rows from a partition
        const vx_array *array = NULL;
        while ((array = vx_partition_next(partition, &info->error)) != NULL) {
            ++info->arrays;
            info->rows += vx_array_len(array);
            vx_array_free(array);
        }

        vx_partition_free(partition);

        if (info->error != NULL) {
            return NULL;
        }
    }

    printf("Thread %lu finished, processed %lu partitions, %lu arrays, %lu rows\n",
           info->thread_id + 1,
           info->partitions,
           info->arrays,
           info->rows);
    return NULL;
}

vx_error *execute_scan(vx_scan *scan, pthread_t num_threads) {
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_t threads[MAX_THREADS];
    struct scan_thread_info infos[MAX_THREADS] = {0};

    printf("Starting scan, using %lu threads\n", num_threads);
    for (pthread_t id = 0; id < num_threads; ++id) {
        struct scan_thread_info *info = &infos[id];
        info->thread_id = id;
        info->mutex = &mutex;
        info->scan = scan;
        pthread_create(&threads[id], NULL, execute_scan_thread, info);
    }

    size_t partitions = 0, arrays = 0, rows = 0;
    for (pthread_t id = 0; id < num_threads; ++id) {
        pthread_join(threads[id], NULL);
        struct scan_thread_info *info = &infos[id];

        if (info->error != NULL) {
            // Don't join other threads as program will be terminated early
            return info->error;
        }

        partitions += info->partitions;
        arrays += info->arrays;
        rows += info->rows;
    }

    printf("Finished scan, processed %lu partitions, %lu arrays, %lu rows\n", partitions, arrays, rows);
    return NULL;
}

int parse_options(int argc, char *argv[], pthread_t *threads, char **paths) {
    int opt;
    while ((opt = getopt(argc, argv, "j:")) != -1) {
        switch (opt) {
        case 'j':
            *threads = atoi(optarg);
            break;
        default:
            fprintf(stderr, "%s", usage);
            return 1;
        }
    }

    if (*threads != 0 && (*threads < 1 || *threads > MAX_THREADS)) {
        fprintf(stderr, "Invalid thread count %lu, expected [1; 64]\n", *threads);
        return 1;
    }

    if (optind + 1 != argc) {
        fprintf(stderr, "%s", usage);
        return 1;
    }

    *paths = argv[optind];
    return 0;
}

int main(int argc, char *argv[]) {
    pthread_t threads = 0;
    char *paths;
    if (parse_options(argc, argv, &threads, &paths)) {
        return 1;
    }

    vx_session *session = vx_session_new();
    if (session == NULL) {
        fprintf(stderr, "Failed to create Vortex session\n");
        return 1;
    }

    printf("Opening files: %s\n", paths);

    // A datasource is a reference to some files.
    // We can request multiple scans from a data source.
    vx_data_source_options ds_options = {.paths = paths};
    vx_error *error = NULL;
    const vx_data_source *data_source = vx_data_source_new(session, &ds_options, &error);
    if (data_source == NULL) {
        print_error("Failed to create data source", error);
        // Returned errors are owned and need to be freed
        vx_error_free(error);
        vx_session_free(session);
        return 1;
    }

    vx_estimate row_count;
    vx_data_source_get_row_count(data_source, &row_count);
    print_estimate("Data source row count", &row_count);

    // A scan is a single traversal of a data source.
    // Here we request a scan without any filters, projections, or limiting.
    vx_scan_options scan_options = {0};
    vx_estimate partition_estimate;
    vx_scan *scan = vx_data_source_scan(data_source, &scan_options, &partition_estimate, &error);
    if (scan == NULL) {
        print_error("Failed to create scan", error);
        vx_error_free(error);
        vx_data_source_free(data_source);
        vx_session_free(session);
        return 1;
    }

    // Caller can use partition estimates to schedule worker threads.
    print_estimate("Partition count", &partition_estimate);
    if (threads == 0) {
        threads = partition_estimate.type == VX_ESTIMATE_UNKNOWN ? 1 : partition_estimate.estimate;
    }

    error = execute_scan(scan, threads);
    if (error != NULL) {
        print_error("Failed to scan", error);
        vx_error_free(error);
    }

    vx_scan_free(scan);
    vx_data_source_free(data_source);
    vx_session_free(session);
    return 0;
}
