// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Hot-page-cache read scaling: N threads read 1 MiB chunks at random offsets for a fixed time.
//   iobench <pread|pread-fresh|mmap> <threads> <seconds> <file> [file...]
// pread: each chunk is pread() into a freshly malloc'ed buffer (what the scan's IO pool does).
// mmap:  each chunk is summed in place from a shared read-only mapping (zero-copy consumption).
// With one file every thread reads that file; with several, thread i reads file i % nfiles.
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CHUNK (1u << 20)

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;
    int use_mmap;
    double seconds;
    uint64_t bytes;
    uint64_t sum;
    unsigned seed;
} job_t;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *run(void *arg) {
    job_t *j = arg;
    double end = now() + j->seconds;
    uint64_t chunks = j->size / CHUNK;
    while (now() < end) {
        for (int k = 0; k < 16; k++) {
            uint64_t off = ((uint64_t)rand_r(&j->seed) % chunks) * CHUNK;
            if (j->use_mmap == 1) {
                const uint64_t *p = (const uint64_t *)(j->map + off);
                uint64_t s = 0;
                for (size_t i = 0; i < CHUNK / 8; i++) s += p[i];
                j->sum += s;
            } else {
                // use_mmap == 2: a fresh anonymous mapping per read, unmapped afterwards (an allocator
                // that hands out untouched pages and returns them to the OS).
                uint8_t *buf = j->use_mmap == 2
                    ? mmap(NULL, CHUNK, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
                    : malloc(CHUNK);
                if (pread(j->fd, buf, CHUNK, (off_t)off) != CHUNK) abort();
                j->sum += buf[0] + buf[CHUNK - 1];
                if (j->use_mmap == 2) munmap(buf, CHUNK); else free(buf);
            }
            j->bytes += CHUNK;
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) return 2;
    int use_mmap = strcmp(argv[1], "mmap") == 0 ? 1 : (strcmp(argv[1], "pread-fresh") == 0 ? 2 : 0);
    int threads = atoi(argv[2]);
    double seconds = atof(argv[3]);
    int nfiles = argc - 4;
    int *fds = calloc(nfiles, sizeof(int));
    const uint8_t **maps = calloc(nfiles, sizeof(uint8_t *));
    uint64_t *sizes = calloc(nfiles, sizeof(uint64_t));
    for (int i = 0; i < nfiles; i++) {
        struct stat st;
        fds[i] = open(argv[4 + i], O_RDONLY);
        if (fds[i] < 0 || fstat(fds[i], &st) != 0) return 3;
        sizes[i] = (uint64_t)st.st_size;
        if (use_mmap == 1) {
            maps[i] = mmap(NULL, sizes[i], PROT_READ, MAP_SHARED, fds[i], 0);
            if (maps[i] == MAP_FAILED) return 4;
        }
    }
    pthread_t *tids = calloc(threads, sizeof(pthread_t));
    job_t *jobs = calloc(threads, sizeof(job_t));
    double t0 = now();
    for (int i = 0; i < threads; i++) {
        jobs[i] = (job_t){fds[i % nfiles], maps[i % nfiles], sizes[i % nfiles], use_mmap, seconds, 0, 0, 1234u + i};
        pthread_create(&tids[i], NULL, run, &jobs[i]);
    }
    uint64_t total = 0, sum = 0;
    for (int i = 0; i < threads; i++) {
        pthread_join(tids[i], NULL);
        total += jobs[i].bytes;
        sum += jobs[i].sum;
    }
    double dt = now() - t0;
    printf("%s threads=%d files=%d: %.2f GB/s total, %.2f GB/s per thread (checksum %llu)\n", argv[1], threads,
           nfiles, total / dt / 1e9, total / dt / 1e9 / threads, (unsigned long long)sum);
    return 0;
}
