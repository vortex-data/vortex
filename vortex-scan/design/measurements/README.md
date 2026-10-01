# Scan V2 measurements

Scripts and results behind [../STATUS.md](../STATUS.md). Everything writes next to itself, so
run the scripts from this directory. Raw output (`raw/`, `raw-warmfirst/`, `prof/`) is ignored
by git; the tables are kept.

| File | What it does |
|---|---|
| `bench_ab.py` | Interleaved V1 (`VORTEX_SCAN_V2` unset) against V2 (`=1`) runs of `duckdb-bench` or `datafusion-bench`, hot or cold. Appends tables to `results.md`. Usage is in its docstring. `ROOT` defaults to the worktree the script is in and `BIN_DIR` to `$ROOT/target/bench-bins/9c6392e5aa`; STATUS.md says where the data and binaries actually are. |
| `evict.py` | Evicts files from the macOS page cache without root (`msync(MS_INVALIDATE)`) and reports residency with `mincore()`. Used by the cold mode. |
| `run_matrix.sh` | The whole matrix: six hot and six cold steps over TPC-H SF1, SF10 and ClickBench on both engines. |
| `run_hot_rerun.sh` | The hot steps again plus the warm-first measurement, for a quieter machine. |
| `bench_first.py` | Warm-first variant: the first iteration of a process with the data already cached. Appends to `results-warmfirst.md`. |
| `run_sweeps.sh`, `prof_sweep.py` | Samply profile of every query, V1 and V2, hot. Writes `prof/` and `sweeps.log`. |
| `prof_attr.py`, `proflib.py` | Attribute a Samply profile to scan and engine categories (IO, decode, filter kernels, pruning, scheduling, conversion, blocked). |
| `prof_cold.py` | Samply profiles of cold runs. |
| `sweep_summary.py` | Summarises `sweeps.log` per engine and suite; its output is `sweep-summary.md`. |
| `final_tables.py` | Builds consolidated per-query tables from the results files and profiles. |
| `iobench.c` | Hot page-cache read scaling: N threads reading 1 MiB chunks with `pread` or `mmap`. Build with `cc -O2 -o iobench iobench.c`. |
| `randread.py` | Random 4 KiB read latency probe, to check that eviction worked. |
| `attr_table.py`, `sweep_dist.py` | Build the attribution and scan-share distribution tables in the reports below. |
| `results.md` | Hot and cold tables for the baseline commit `9c6392e5aa`, measured 2026-09-30. Sections marked SUPERSEDED were rerun further down. |
| `results-warmfirst.md` | Warm-first tables for the same commit. Incomplete: DuckDB TPC-H only. |
| `sweeps.log`, `sweep-summary.md` | The per-query scan-share sweep and its summary. |
| `REPORT_TABLES.md` | Consolidated per-query hot and cold tables. |
| `REPORT_SCAN_SHARE.md` | Scan share of wall time per engine and suite: distribution, and the geomean if the scan cost fell by 30, 50, or 100%. |
| `REPORT_SWEEP_PER_QUERY.md` | Scan share and blocked time for every query. |
| `REPORT_ATTRIBUTION.md` | Where the time goes, V1 against V2, for twelve representative queries. |

Samply must be installed for the profiling scripts, and the bench binaries must exist under
`target/bench-bins/<commit>/` (see STATUS.md for the build command).
