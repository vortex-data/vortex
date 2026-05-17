// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Final benchmark: **ord-number generation only**, isolated from the
//! merge driver, across every encoding and across all three designs:
//!
//!   * `ord_iter`   — the converged `OrdIter` trait (chunked, scratch-backed)
//!   * `ord_direct` — direct per-row OVC access (the hand-specialised path)
//!   * `ord_memcmp` — materialize to a contiguous u8 byte buffer
//!
//! Also benches **`skip`** (the duplicate-bypass shortcut, OrdIter-only)
//! to validate it is essentially O(1) regardless of n.
//!
//! Run with:
//!   cargo test --release -p vortex-array ord_bench::tests::bench \
//!       -- --ignored --nocapture --test-threads=1

#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names
)]

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::ord_common::{
        ConstantI64, DictI64, PrimI64, RunEndI64, VarBin, build_dict, build_prim, build_runend,
        build_varbin,
    };
    use crate::ord_direct::{
        constant_ord_at, dict_ord_at, prim_ord_at, runend_ord_at, varbin_ord_at,
    };
    use crate::ord_iter::{
        ClosureIter, ConstantIter, DictIter, MultiColI32Iter, OrdChunk, OrdIter, PrimIter,
        RunEndIter, Scratch, VarBinIter,
    };
    use crate::ord_memcmp::{
        materialize_constant, materialize_dict, materialize_prim, materialize_runend,
        materialize_varbin,
    };

    const N: usize = 200_000;
    const ITERS: u32 = 10;
    const CHUNK: usize = 1024;

    fn run(label: &str, mut f: impl FnMut() -> u64) {
        let _ = f();
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS {
            acc = acc.wrapping_add(std::hint::black_box(f()));
        }
        let ns = t.elapsed().as_nanos() as f64 / (u64::from(ITERS) * N as u64) as f64;
        println!("    {:<38} {:>9.3} ns/row   acc={acc}", label, ns);
    }

    /// Drain an `OrdIter` chunk by chunk into a black-box accumulator.
    fn drain_iter(iter: &mut dyn OrdIter, scratch: &mut Scratch) -> u64 {
        let mut acc = 0u64;
        while let Some(chunk) = iter.next_chunk(CHUNK, scratch) {
            match chunk {
                OrdChunk::Constant { value, len } => {
                    acc = acc.wrapping_add(value.wrapping_mul(len as u64));
                }
                OrdChunk::RunEnd { run_ends, values, .. } => {
                    for (k, &v) in values.iter().enumerate() {
                        let prev = if k == 0 { 0 } else { run_ends[k - 1] as u64 };
                        let len = run_ends[k] as u64 - prev;
                        acc = acc.wrapping_add(v.wrapping_mul(len));
                    }
                }
                OrdChunk::Dense(vs) => {
                    for &v in vs {
                        acc = acc.wrapping_add(v);
                    }
                }
            }
        }
        acc
    }

    /// Compute via direct per-row OVC compute (the inline path used by
    /// the hand-specialised merge functions in `ord_direct`).
    fn drain_direct_prim(p: &PrimI64<'_>) -> u64 {
        (0..p.data.len()).fold(0u64, |acc, i| acc.wrapping_add(prim_ord_at(p, i)))
    }
    fn drain_direct_dict(d: &DictI64<'_>) -> u64 {
        (0..d.codes.len()).fold(0u64, |acc, i| acc.wrapping_add(dict_ord_at(d, i)))
    }
    fn drain_direct_runend(r: &RunEndI64<'_>) -> u64 {
        (0..r.len).fold(0u64, |acc, i| acc.wrapping_add(runend_ord_at(r, i)))
    }
    fn drain_direct_constant(c: &ConstantI64) -> u64 {
        let v = constant_ord_at(c);
        v.wrapping_mul(c.len as u64)
    }
    fn drain_direct_varbin(v: &VarBin<'_>) -> u64 {
        (0..v.len()).fold(0u64, |acc, i| acc.wrapping_add(varbin_ord_at(v, i)))
    }

    /// Materialize once, then drain the byte buffer as u64 (so the read is
    /// included in the measurement, matching what a downstream consumer
    /// would do).
    fn drain_memcmp_buf(buf: &[u8], stride: usize) -> u64 {
        let n = buf.len() / stride;
        let mut acc = 0u64;
        for i in 0..n {
            let s = i * stride;
            let mut a = [0u8; 8];
            a.copy_from_slice(&buf[s..s + 8.min(stride)]);
            acc = acc.wrapping_add(u64::from_be_bytes(a));
        }
        acc
    }

    #[test]
    #[ignore = "benchmark, run explicitly"]
    fn bench_ord_generation() {
        println!("\n== Ord-number generation, isolated. N={N} rows/side. ==\n");

        // 1. PRIMITIVE (disjoint, sorted)
        let prim = build_prim(N, 0);
        let p = PrimI64 { data: &prim };
        println!("-- PRIMITIVE i64 ({} rows, sorted) --", N);
        run("OrdIter::drain (chunked into scratch)", || {
            let mut it = PrimIter::new(&prim);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });
        run("direct ord-value per row", || drain_direct_prim(&p));
        run("memcmp materialize + drain bytes", || {
            let buf = materialize_prim(&p);
            drain_memcmp_buf(&buf, 8)
        });
        run("memcmp materialize ONLY (no drain)", || materialize_prim(&p).len() as u64);

        // 2. DICT — sorted dict, 256 distinct (typical low-cardinality)
        println!("\n-- DICT (256 distinct, sorted dict, sorted codes) --");
        let (codes, dict) = build_dict(N, 256, 0);
        let d = DictI64 { codes: &codes, dict: &dict };
        run("OrdIter::drain", || {
            let mut it = DictIter::new(&codes);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });
        run("direct ord-value per row", || drain_direct_dict(&d));
        run("memcmp materialize + drain", || {
            let buf = materialize_dict(&d);
            drain_memcmp_buf(&buf, 8)
        });

        // 3. RUN-END — three run-length regimes
        for &(runs, run_len_label) in &[(2000, "100 rows/run"), (20_000, "10 rows/run"), (200, "1000 rows/run")] {
            let run_len = N / runs;
            let (ends, vals, total) = build_runend(runs, run_len, 0);
            let r = RunEndI64 { run_ends: &ends, values: &vals, len: total };
            println!("\n-- RUN-END ({}, {runs} runs total) --", run_len_label);
            run("OrdIter::drain (structural runs!)", || {
                let mut it = RunEndIter::new(&ends, &vals, total);
                let mut sc = Scratch::new(CHUNK);
                drain_iter(&mut it, &mut sc)
            });
            run("direct ord-value per row (binsearch)", || drain_direct_runend(&r));
            run("memcmp materialize + drain", || {
                let buf = materialize_runend(&r);
                drain_memcmp_buf(&buf, 8)
            });
        }

        // 4. CONSTANT
        let c = ConstantI64 { value: 42, len: N };
        println!("\n-- CONSTANT --");
        run("OrdIter::drain (single Constant chunk)", || {
            let mut it = ConstantIter::new(42, N);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });
        run("direct ord-value (one read, N counts)", || drain_direct_constant(&c));
        run("memcmp materialize + drain", || {
            let buf = materialize_constant(&c);
            drain_memcmp_buf(&buf, 8)
        });

        // 5. VARBIN at two key widths
        for &width in &[50usize, 200] {
            let (offsets, data) = build_varbin(N, width, 0);
            let v = VarBin { offsets: &offsets, data: &data };
            println!("\n-- VARBIN ({width}B values, leading u64 key) --");
            run("OrdIter::drain (first-8B prefix)", || {
                let mut it = VarBinIter::new(&offsets, &data);
                let mut sc = Scratch::new(CHUNK);
                drain_iter(&mut it, &mut sc)
            });
            run("direct first-8B per row", || drain_direct_varbin(&v));
            run("memcmp materialize + drain", || {
                let buf = materialize_varbin(&v, width);
                drain_memcmp_buf(&buf, width)
            });
        }

        // ─── SKIP cost ───────────────────────────────────────────────────────
        println!("\n== OrdIter::skip — duplicate-bypass shortcut ==");
        println!("    skip should be ~O(1) regardless of n.\n");

        // skip 50% then drain rest
        run("PrimIter: skip(N/2) then drain rest", || {
            let mut it = PrimIter::new(&prim);
            it.skip(N / 2);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });
        run("RunEndIter (100/run): skip(N/2) + drain", || {
            let (ends, vals, total) = build_runend(2000, N / 2000, 0);
            let mut it = RunEndIter::new(&ends, &vals, total);
            it.skip(N / 2);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });
        run("ConstantIter: skip(N/2) + drain", || {
            let mut it = ConstantIter::new(42, N);
            it.skip(N / 2);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });

        // skip 99% (almost everything)
        run("PrimIter: skip(99%) + drain", || {
            let mut it = PrimIter::new(&prim);
            it.skip(N * 99 / 100);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });

        // skip ALL — should be near-zero work
        run("PrimIter: skip(N) (no drain)", || {
            let mut it = PrimIter::new(&prim);
            it.skip(N);
            it.ord_len() as u64
        });
        run("ConstantIter: skip(N) (no drain)", || {
            let mut it = ConstantIter::new(42, N);
            it.skip(N);
            it.ord_len() as u64
        });
        run("RunEndIter (100/run): skip(N)", || {
            let (ends, vals, total) = build_runend(2000, N / 2000, 0);
            let mut it = RunEndIter::new(&ends, &vals, total);
            it.skip(total);
            it.ord_len() as u64
        });

        // ─── REAL EXAMPLES: realistic mixed sizes ────────────────────────────
        println!("\n== Real-world-shaped examples (one side each) ==\n");

        // (a) Tiny dict (16 distinct values, common for categoricals)
        let (codes_lo, dict_lo) = build_dict(N, 16, 0);
        let d_lo = DictI64 { codes: &codes_lo, dict: &dict_lo };
        println!("-- DICT 16 distinct (categorical) --");
        run("OrdIter", || {
            let mut it = DictIter::new(&codes_lo);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });
        run("direct", || drain_direct_dict(&d_lo));
        run("memcmp", || {
            let buf = materialize_dict(&d_lo);
            drain_memcmp_buf(&buf, 8)
        });

        // (b) High-cardinality dict (10K distinct — borderline)
        let (codes_hi, dict_hi) = build_dict(N, 10_000, 0);
        let d_hi = DictI64 { codes: &codes_hi, dict: &dict_hi };
        println!("\n-- DICT 10K distinct (high-cardinality) --");
        run("OrdIter", || {
            let mut it = DictIter::new(&codes_hi);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });
        run("direct", || drain_direct_dict(&d_hi));
        run("memcmp", || {
            let buf = materialize_dict(&d_hi);
            drain_memcmp_buf(&buf, 8)
        });

        // (c) RunEnd with very long runs (1000 rows/run)
        let (ends_long, vals_long, total_long) = build_runend(200, 1000, 0);
        let r_long = RunEndI64 { run_ends: &ends_long, values: &vals_long, len: total_long };
        println!("\n-- RUN-END 1000 rows/run (200 runs) --");
        run("OrdIter (structural)", || {
            let mut it = RunEndIter::new(&ends_long, &vals_long, total_long);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });
        run("direct (binsearch per row)", || drain_direct_runend(&r_long));
        run("memcmp (expand each run)", || {
            let buf = materialize_runend(&r_long);
            drain_memcmp_buf(&buf, 8)
        });

        // (c+) Multi-column: 2 i32 cols packed into one OVC u64
        println!("\n-- MULTI-COLUMN (2 i32 cols packed into one OVC u64) --");
        let mc_col0: Vec<i32> = (0..N as i32).collect();
        let mc_col1: Vec<i32> = (0..N as i32).map(|i| i % 13).collect();
        run("OrdIter MultiColI32 (no merge)", || {
            let mut it = MultiColI32Iter::new(&mc_col0, &mc_col1);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });

        // (c++) Fallback closure-based iter — proves any encoding can join.
        println!("\n-- FALLBACK ClosureIter (any encoding can use this) --");
        let cl_data: Vec<i64> = (0..N as i64).collect();
        run("ClosureIter wrapping a slice", || {
            let mut it = ClosureIter::new(N, |i| cl_data[i]);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });
        // Reference: a real PrimIter on the same data.
        run("PrimIter reference (specialised path)", || {
            let mut it = PrimIter::new(&cl_data);
            let mut sc = Scratch::new(CHUNK);
            drain_iter(&mut it, &mut sc)
        });

        // (d) VarBin 1KB values (URLs, paths)
        let (off_1k, data_1k) = build_varbin(N / 10, 1024, 0);
        let v_1k = VarBin { offsets: &off_1k, data: &data_1k };
        let n_real = N / 10;
        println!("\n-- VARBIN 1KB values ({n_real} rows) --");
        run_n(
            "OrdIter (first-8B)",
            n_real,
            || {
                let mut it = VarBinIter::new(&off_1k, &data_1k);
                let mut sc = Scratch::new(CHUNK);
                drain_iter(&mut it, &mut sc)
            },
        );
        run_n("direct", n_real, || drain_direct_varbin(&v_1k));
        run_n("memcmp materialize+drain (stride=1024)", n_real, || {
            let buf = materialize_varbin(&v_1k, 1024);
            drain_memcmp_buf(&buf, 1024)
        });
    }

    /// Variant of `run` that lets us override N when the test data has a
    /// different row count.
    fn run_n(label: &str, n: usize, mut f: impl FnMut() -> u64) {
        let _ = f();
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..ITERS {
            acc = acc.wrapping_add(std::hint::black_box(f()));
        }
        let ns = t.elapsed().as_nanos() as f64 / (u64::from(ITERS) * n as u64) as f64;
        println!("    {:<38} {:>9.3} ns/row   acc={acc}", label, ns);
    }
}
