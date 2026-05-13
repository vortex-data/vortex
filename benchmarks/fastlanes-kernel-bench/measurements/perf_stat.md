# Hardware-counter measurements (skipped)

## Status

**Skipped** -- `perf` is not installed in this container and cannot be
installed without changes to the base image:

```
$ which perf
(no output)
$ perf --version
bash: perf: command not found
$ find / -name perf -executable -type f 2>/dev/null
(no results)
$ apt-get list --installed 2>/dev/null | grep -iE "linux-tools|linux-perf"
(no results)
```

`/proc/sys/kernel/perf_event_paranoid` reports `2`, which would in any case
have required `--user-mode` (`cycles:u` form) for any counter access, but the
binary itself is unavailable. Kernel is `6.18.5`.

## Workaround

Step 3 (`llvm-mca`) substitutes here: it is a static throughput / port-pressure
model based on a published Sapphire/Emerald-Rapids scheduling table. It cannot
catch real cache-miss or front-end stall behaviour, but it can give the
*theoretical* port-pressure breakdown that perf-stat would otherwise
measure. Conclusions about "memory-bound vs ALU-bound" therefore rely on the
llvm-mca port pressure plus the Step 4 memcpy baseline, not on direct
counter readings.

Step 4's `bare_unpack_ns / memcpy_ns` ratio is the closest *empirical*
proxy: if the ratio is near 1.0 the kernel is memory-limited; if it is much
greater than 1.0 the kernel is doing real ALU work above the memory floor.
