// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rederive and verify the fixed-width filter dispatch thresholds on this machine.
//!
//! The fixed-width filter in `vortex-array` picks between a SIMD compress kernel, a
//! byte-compress LUT, a scalar bitmap walk, and cached index/slice strategies using density
//! thresholds benchmarked per architecture (see `arrays/filter/execute/buffer.rs` and
//! `arrays/filter/execute/simd_compress`). Normal dispatch runs exactly one strategy per mask,
//! so this binary instead times every strategy directly across a density sweep, derives the
//! empirical crossover points for this CPU, and checks how much performance the compiled-in
//! thresholds leave on the table against the fastest measured strategy.
//!
//! Run it on an otherwise idle machine:
//!
//! ```sh
//! cargo run --release -p vortex-array --features _test-harness \
//!     --example filter_threshold_verification
//! ```
//!
//! Options:
//!
//! - `--len <elements>`: buffer length (default 4096, matching `benches/filter_fixed_width.rs`)
//! - `--step <density>`: density grid step (default 0.025)
//! - `--seeds <n>`: random masks averaged per density (default 2)
//! - `--tolerance <ratio>`: max relative regret before a width fails (default 0.25)
//! - `--csv`: also dump the raw per-density timings for offline analysis
//!
//! The exit code is non-zero when the configured dispatch is slower than the fastest measured
//! strategy by more than the tolerance at two consecutive densities. Only out-of-place filtering
//! is timed; in-place dispatch shares the same thresholds. Wall-clock timings are noisy, so
//! treat a single WARN as a prompt to re-run, and a FAIL as a prompt to rederive the thresholds
//! from the measured crossovers this binary prints.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::exit
)]

use std::env::consts::ARCH;
use std::hint::black_box;
use std::time::Instant;

use mimalloc::MiMalloc;
use vortex_array::dtype::i256;
use vortex_array::test_harness::filter_thresholds as ft;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

struct Config {
    len: usize,
    step: f64,
    seeds: u64,
    tolerance: f64,
    csv: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            len: 4096,
            step: 0.025,
            seeds: 2,
            tolerance: 0.25,
            csv: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Strategy {
    BitmapWalk,
    ByteCompress,
    Simd,
}

impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Strategy::BitmapWalk => "bitmap walk",
            Strategy::ByteCompress => "byte compress",
            Strategy::Simd => "SIMD compress",
        }
    }
}

/// Timings for one density grid point, averaged across seeds.
struct Point {
    density: f64,
    bitmap_ns: f64,
    byte_ns: f64,
    simd_ns: Option<f64>,
    indices_ns: f64,
}

impl Point {
    fn time_of(&self, strategy: Strategy) -> f64 {
        match strategy {
            Strategy::BitmapWalk => self.bitmap_ns,
            Strategy::ByteCompress => self.byte_ns,
            Strategy::Simd => self.simd_ns.unwrap_or(f64::INFINITY),
        }
    }

    fn fastest(&self) -> (Strategy, f64) {
        [Strategy::BitmapWalk, Strategy::ByteCompress, Strategy::Simd]
            .into_iter()
            .map(|strategy| (strategy, self.time_of(strategy)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap()
    }
}

fn main() {
    let cfg = parse_args();

    println!("== fixed-width filter threshold verification ==");
    println!("arch: {ARCH}");
    println!(
        "len: {} elements, density step {}, {} seed(s), tolerance {:.0}%",
        cfg.len,
        cfg.step,
        cfg.seeds,
        cfg.tolerance * 100.0
    );
    println!();

    let masks = build_masks(&cfg);

    let mut all_pass = true;
    all_pass &= run_width::<u8>("u8", &cfg, &masks, |index| index as u8);
    all_pass &= run_width::<u16>("u16", &cfg, &masks, |index| index as u16);
    all_pass &= run_width::<u32>("u32", &cfg, &masks, |index| index as u32);
    all_pass &= run_width::<u64>("u64", &cfg, &masks, |index| index as u64);
    all_pass &= run_width::<u128>("u128", &cfg, &masks, |index| index as u128);
    all_pass &= run_width::<i256>("i256", &cfg, &masks, |index| i256::from_i128(index as i128));

    run_slices_section(&cfg);

    println!();
    if all_pass {
        println!("RESULT: PASS");
    } else {
        println!("RESULT: FAIL — rederive the flagged thresholds from the measured crossovers");
        std::process::exit(1);
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .unwrap_or_else(|| die(&format!("{name} requires a value")))
        };
        match arg.as_str() {
            "--len" => cfg.len = value("--len").parse().unwrap_or_else(|_| die("bad --len")),
            "--step" => {
                cfg.step = value("--step")
                    .parse()
                    .unwrap_or_else(|_| die("bad --step"))
            }
            "--seeds" => {
                cfg.seeds = value("--seeds")
                    .parse()
                    .unwrap_or_else(|_| die("bad --seeds"))
            }
            "--tolerance" => {
                cfg.tolerance = value("--tolerance")
                    .parse()
                    .unwrap_or_else(|_| die("bad --tolerance"))
            }
            "--csv" => cfg.csv = true,
            "--help" | "-h" => {
                println!(
                    "usage: filter_threshold_verification \
                     [--len N] [--step F] [--seeds N] [--tolerance F] [--csv]"
                );
                std::process::exit(0);
            }
            other => die(&format!("unknown argument: {other}")),
        }
    }
    if cfg.len < ft::SIMD_MIN_LEN {
        die(&format!("--len must be at least {}", ft::SIMD_MIN_LEN));
    }
    if !(0.001..=0.5).contains(&cfg.step) || cfg.seeds == 0 {
        die("--step must be in [0.001, 0.5] and --seeds nonzero");
    }
    cfg
}

fn die(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

/// Random masks per grid density, one per seed. All widths reuse the same masks.
fn build_masks(cfg: &Config) -> Vec<(f64, Vec<Mask>)> {
    let mut grid = vec![0.01];
    let mut density = cfg.step;
    while density < 0.99 {
        grid.push(density);
        density += cfg.step;
    }
    grid.push(0.99);

    grid.into_iter()
        .map(|target| {
            let masks = (0..cfg.seeds)
                .map(|seed| random_mask(cfg.len, target, seed))
                .collect();
            (target, masks)
        })
        .collect()
}

/// Deterministic xorshift Bernoulli mask, forced to stay mixed (never all-true/all-false).
fn random_mask(len: usize, density: f64, seed: u64) -> Mask {
    let threshold = (density * u64::MAX as f64) as u64;
    let mut state = 0x1234_5678_9abc_def0u64 ^ (seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut bits: Vec<bool> = (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state <= threshold
        })
        .collect();
    if bits.iter().all(|&bit| bit) {
        bits[0] = false;
    }
    if bits.iter().all(|&bit| !bit) {
        bits[0] = true;
    }
    Mask::from_buffer(BitBuffer::from_iter(bits))
}

fn mask_values(mask: &Mask) -> &MaskValues {
    mask.values().expect("masks are forced to stay mixed")
}

/// Time one call of `f`: warm up, then take the best of several fixed-size batches.
fn bench_ns(mut f: impl FnMut()) -> f64 {
    f();
    let start = Instant::now();
    f();
    let once_ns = start.elapsed().as_nanos().max(50) as u64;
    let iters = (500_000 / once_ns).clamp(1, 100_000) as usize;

    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iters {
            f();
        }
        best = best.min(start.elapsed().as_nanos() as f64 / iters as f64);
    }
    best
}

fn sweep<T: Copy>(values: &[T], masks: &[(f64, Vec<Mask>)]) -> Vec<Point> {
    let simd_available =
        ft::filter_by_simd_compress(values, mask_values(&masks[masks.len() / 2].1[0])).is_some();

    masks
        .iter()
        .map(|(_, seed_masks)| {
            let mut density = 0.0;
            let mut bitmap_ns = 0.0;
            let mut byte_ns = 0.0;
            let mut simd_ns = 0.0;
            let mut indices_ns = 0.0;

            for mask in seed_masks {
                let mv = mask_values(mask);
                let indices: Vec<usize> = mv
                    .bit_buffer()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, bit)| bit.then_some(index))
                    .collect();

                density += mv.true_count() as f64 / mv.len() as f64;
                bitmap_ns += bench_ns(|| {
                    black_box(ft::filter_by_bitmap_walk(black_box(values), black_box(mv)));
                });
                byte_ns += bench_ns(|| {
                    black_box(ft::filter_by_byte_compress(
                        black_box(values),
                        black_box(mv),
                    ));
                });
                if simd_available {
                    simd_ns += bench_ns(|| {
                        black_box(ft::filter_by_simd_compress(
                            black_box(values),
                            black_box(mv),
                        ));
                    });
                }
                indices_ns += bench_ns(|| {
                    black_box(ft::filter_by_indices(
                        black_box(values),
                        black_box(&indices),
                    ));
                });
            }

            let seeds = seed_masks.len() as f64;
            Point {
                density: density / seeds,
                bitmap_ns: bitmap_ns / seeds,
                byte_ns: byte_ns / seeds,
                simd_ns: simd_available.then_some(simd_ns / seeds),
                indices_ns: indices_ns / seeds,
            }
        })
        .collect()
}

/// The strategy the compiled-in thresholds pick for a bitmap-only mask (ladder priorities 3-5).
fn configured_pick<T>(density: f64, len: usize) -> Strategy {
    if let Some((_, band)) = ft::simd_density_band::<T>()
        && len >= ft::SIMD_MIN_LEN
        && band.contains(&density)
    {
        return Strategy::Simd;
    }
    if density >= ft::byte_compress_density_threshold::<T>() {
        Strategy::ByteCompress
    } else {
        Strategy::BitmapWalk
    }
}

fn verify_strategies_agree<T: Copy + PartialEq + std::fmt::Debug>(values: &[T], mask: &Mask) {
    let mv = mask_values(mask);
    let expected = ft::filter_by_bitmap_walk(values, mv);
    assert_eq!(
        ft::filter_by_byte_compress(values, mv).as_slice(),
        expected.as_slice(),
        "byte compress disagrees with the bitmap walk"
    );
    if let Some(filtered) = ft::filter_by_simd_compress(values, mv) {
        assert_eq!(
            filtered.as_slice(),
            expected.as_slice(),
            "SIMD compress disagrees with the bitmap walk"
        );
    }
}

fn run_width<T: Copy + PartialEq + std::fmt::Debug>(
    name: &str,
    cfg: &Config,
    masks: &[(f64, Vec<Mask>)],
    make: impl Fn(usize) -> T,
) -> bool {
    let values: Vec<T> = (0..cfg.len).map(make).collect();
    verify_strategies_agree(&values, &masks[masks.len() / 2].1[0]);

    let points = sweep(&values, masks);

    println!("-- width {}B ({name}) --", size_of::<T>());
    match ft::simd_density_band::<T>() {
        Some((kernel, band)) => {
            if band.end.is_finite() {
                println!(
                    "configured: SIMD {kernel} for {:.3} <= d < {:.3}",
                    band.start, band.end
                );
            } else {
                println!("configured: SIMD {kernel} for d >= {:.3}", band.start);
            }
        }
        None => println!("configured: no SIMD kernel for this width"),
    }
    println!(
        "configured: byte compress over bitmap walk for d >= {:.3}",
        ft::byte_compress_density_threshold::<T>()
    );

    if cfg.csv {
        for point in &points {
            println!(
                "csv,sweep,{name},{:.4},{:.1},{:.1},{},{:.1}",
                point.density,
                point.bitmap_ns,
                point.byte_ns,
                point
                    .simd_ns
                    .map_or_else(|| "-".to_string(), |ns| format!("{ns:.1}")),
                point.indices_ns,
            );
        }
    }

    report_measured_crossovers(&points);
    let pass = report_regret::<T>(cfg, &points);
    report_indices_crossover(&points);
    println!();
    pass
}

fn report_measured_crossovers(points: &[Point]) {
    // SIMD band: the longest run of grid points where SIMD is the fastest strategy.
    if points.iter().any(|point| point.simd_ns.is_some()) {
        let simd_wins: Vec<bool> = points
            .iter()
            .map(|point| point.fastest().0 == Strategy::Simd)
            .collect();
        match longest_true_run(&simd_wins) {
            Some((first, last)) => {
                let upper = if last == points.len() - 1 {
                    "the top of the sweep".to_string()
                } else {
                    format!("d ~= {:.3}", points[last].density)
                };
                println!(
                    "measured:   SIMD is fastest from d ~= {:.3} to {upper}",
                    points[first].density
                );
            }
            None => println!("measured:   SIMD is never the fastest strategy"),
        }
    }

    // Byte compress vs bitmap walk: first density where byte compress wins and keeps winning at
    // the next grid point (smoothing single-point noise).
    let byte_crossover = points.windows(2).position(|pair| {
        pair[0].byte_ns < pair[0].bitmap_ns && pair[1].byte_ns < pair[1].bitmap_ns
    });
    match byte_crossover {
        Some(0) => println!("measured:   byte compress beats bitmap walk at every density"),
        Some(index) => println!(
            "measured:   byte compress beats bitmap walk from d ~= {:.3}",
            points[index].density
        ),
        None => println!("measured:   byte compress never beats bitmap walk"),
    }
}

fn longest_true_run(wins: &[bool]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    let mut start = None;
    for (index, &win) in wins.iter().enumerate() {
        match (win, start) {
            (true, None) => start = Some(index),
            (false, Some(run_start)) => {
                if best.is_none_or(|(b0, b1)| index - 1 - run_start > b1 - b0) {
                    best = Some((run_start, index - 1));
                }
                start = None;
            }
            _ => {}
        }
    }
    if let Some(run_start) = start
        && best.is_none_or(|(b0, b1)| wins.len() - 1 - run_start > b1 - b0)
    {
        best = Some((run_start, wins.len() - 1));
    }
    best
}

/// Compare the configured pick against the fastest measured strategy at every density.
fn report_regret<T>(cfg: &Config, points: &[Point]) -> bool {
    let mut max_regret = 0.0f64;
    let mut max_regret_density = 0.0;
    let mut over: Vec<bool> = Vec::with_capacity(points.len());

    for point in points {
        let pick = configured_pick::<T>(point.density, cfg.len);
        let (fastest, fastest_ns) = point.fastest();
        let regret = if pick == fastest {
            0.0
        } else {
            point.time_of(pick) / fastest_ns - 1.0
        };
        if regret > max_regret {
            max_regret = regret;
            max_regret_density = point.density;
        }
        over.push(regret > cfg.tolerance);
    }

    let fail = over.windows(2).any(|pair| pair[0] && pair[1]);
    let verdict = if fail {
        "FAIL"
    } else if over.iter().any(|&o| o) {
        "WARN (single noisy point over tolerance)"
    } else {
        "PASS"
    };
    println!(
        "dispatch:   max regret {:.1}% at d ~= {max_regret_density:.3} -> {verdict}",
        max_regret * 100.0
    );

    if fail {
        for (point, &is_over) in points.iter().zip(&over) {
            if is_over {
                let pick = configured_pick::<T>(point.density, cfg.len);
                let (fastest, fastest_ns) = point.fastest();
                println!(
                    "            d ~= {:.3}: configured {} {:.0}ns vs fastest {} {:.0}ns",
                    point.density,
                    pick.name(),
                    point.time_of(pick),
                    fastest.name(),
                    fastest_ns,
                );
            }
        }
    }
    !fail
}

/// Verify [`ft::CACHED_INDICES_MAX_DENSITY`]: dispatch gathers by cached indices up to that
/// density, so the gather should stay competitive with the fastest bitmap strategy below it.
fn report_indices_crossover(points: &[Point]) {
    let loses = points.windows(2).position(|pair| {
        pair[0].indices_ns > pair[0].fastest().1 && pair[1].indices_ns > pair[1].fastest().1
    });
    let measured = match loses {
        Some(index) => format!("stops winning at d ~= {:.3}", points[index].density),
        None => "wins at every density".to_string(),
    };
    let configured = ft::CACHED_INDICES_MAX_DENSITY;
    let ok = match loses {
        Some(index) => (points[index].density - configured).abs() <= 0.2,
        None => true,
    };
    println!(
        "indices:    gather by cached indices {measured} (configured max {configured:.2}) -> {}",
        if ok { "OK" } else { "WARN" }
    );
}

/// Verify [`ft::MIN_SLICES_AVERAGE_RUN_LENGTH`]: dispatch copies cached slices when the average
/// selected run is at least that long, so the slice copy should win from there on.
fn run_slices_section(cfg: &Config) {
    println!(
        "-- cached-slices crossover (configured min average run length {}) --",
        ft::MIN_SLICES_AVERAGE_RUN_LENGTH
    );
    run_slices_width::<u8>("u8", cfg, |index| index as u8);
    run_slices_width::<u32>("u32", cfg, |index| index as u32);
    run_slices_width::<u64>("u64", cfg, |index| index as u64);
    run_slices_width::<i256>("i256", cfg, |index| i256::from_i128(index as i128));
}

fn run_slices_width<T: Copy>(name: &str, cfg: &Config, make: impl Fn(usize) -> T) {
    let values: Vec<T> = (0..cfg.len).map(make).collect();
    let run_lengths = [2usize, 4, 6, 8, 12, 16, 32];
    let mut first_win: Option<usize> = None;

    for run_length in run_lengths {
        // Alternating keep/skip runs of `run_length`, i.e. average run length == `run_length`
        // at density 0.5.
        let slices: Vec<(usize, usize)> = (0..cfg.len / (2 * run_length))
            .map(|run| (run * 2 * run_length, run * 2 * run_length + run_length))
            .collect();
        let true_count: usize = slices.iter().map(|(start, end)| end - start).sum();
        let mask = Mask::from_slices(cfg.len, slices.clone());
        let mv = mask_values(&mask);
        // Materialize the bitmap outside the timed region, as dispatch sees it cached.
        let _ = mv.bit_buffer();

        let slices_ns = bench_ns(|| {
            black_box(ft::filter_by_slices(
                black_box(&values),
                black_box(&slices),
                true_count,
            ));
        });
        let bitmap_ns = bench_ns(|| {
            black_box(ft::filter_by_bitmap_walk(black_box(&values), black_box(mv)));
        });
        let byte_ns = bench_ns(|| {
            black_box(ft::filter_by_byte_compress(
                black_box(&values),
                black_box(mv),
            ));
        });
        let simd_ns = ft::filter_by_simd_compress(&values, mv).map(|_| {
            bench_ns(|| {
                black_box(ft::filter_by_simd_compress(
                    black_box(&values),
                    black_box(mv),
                ));
            })
        });
        let best_other = bitmap_ns.min(byte_ns).min(simd_ns.unwrap_or(f64::INFINITY));

        if cfg.csv {
            println!(
                "csv,slices,{name},{run_length},{slices_ns:.1},{bitmap_ns:.1},{byte_ns:.1},{}",
                simd_ns.map_or_else(|| "-".to_string(), |ns| format!("{ns:.1}")),
            );
        }
        if slices_ns < best_other {
            first_win.get_or_insert(run_length);
        } else {
            first_win = None;
        }
    }

    let configured = ft::MIN_SLICES_AVERAGE_RUN_LENGTH;
    match first_win {
        Some(run_length) => println!(
            "{name}: slice copy wins from run length >= {run_length} -> {}",
            if run_length <= 2 * configured {
                "OK"
            } else {
                "WARN"
            }
        ),
        None => println!("{name}: slice copy never wins -> WARN"),
    }
}
