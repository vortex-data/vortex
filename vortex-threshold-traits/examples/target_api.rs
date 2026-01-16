//! TARGET API EXAMPLE - This shows the stats-based benchmark API.
//!
//! This file demonstrates the StatsBench API for threshold finding.
//! Phase 2 (Stats-Based API) is now implemented!

#![allow(dead_code, unused_imports, unused_variables)]

// === IMPORTS ===

// Core types from vortex-threshold-traits
// For the example algorithm
use rand::Rng;
use rand::SeedableRng;
use vortex_threshold_traits::Scale;
use vortex_threshold_traits::StatsBench;
use vortex_threshold_traits::StatsGrid;
use vortex_threshold_traits::StatsPoint;

// ============================================================================
// STEP 1: Define your Data type
// ============================================================================

/// The raw input to our algorithm.
/// For rank: a bitmap and a position to query.
#[derive(Clone)]
struct RankData {
    bitmap: Vec<u64>,
    position: usize,
}

// ============================================================================
// STEP 2: Define Stats computed from Data
// ============================================================================

/// Stats computed from data, used for dispatch decisions.
/// These are the dimensions we search over.
#[derive(Clone, Debug)]
struct RankStats {
    /// Length of the bitmap in u64 words
    len: usize,
    /// Fraction of bits set (0.0 to 1.0)
    density: f64,
}

impl RankStats {
    /// Compute stats from data (called by the framework)
    fn compute(data: &RankData) -> Self {
        let total_bits = data.bitmap.len() * 64;
        let set_bits: usize = data.bitmap.iter().map(|w| w.count_ones() as usize).sum();
        let density = if total_bits > 0 {
            set_bits as f64 / total_bits as f64
        } else {
            0.0
        };
        Self {
            len: data.bitmap.len(),
            density,
        }
    }

    /// Generate data matching these stats (called by the framework)
    fn generate(&self, seed: u64) -> RankData {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Generate bitmap with target density
        let bitmap: Vec<u64> = (0..self.len)
            .map(|_| {
                let mut word = 0u64;
                for bit in 0..64 {
                    if rng.random::<f64>() < self.density {
                        word |= 1 << bit;
                    }
                }
                word
            })
            .collect();

        // Random position within the bitmap
        let max_pos = self.len * 64;
        let position = if max_pos > 0 {
            rng.random_range(0..max_pos)
        } else {
            0
        };

        RankData { bitmap, position }
    }
}

// ============================================================================
// STEP 3: Define per-variant parameters with ParamGrid derive
// ============================================================================

/// Parameters for the chunked variant.
/// The derive macro generates:
/// - `fn iter()` that yields all grid combinations
/// - `fn default()` that returns the #[default] values
// #[derive(Clone, ParamGrid)]
#[derive(Clone)]
struct ChunkedParams {
    /// Process this many words at a time
    // #[grid(1, 2, 4, 8, 16)]
    // #[default = 4]
    chunk_size: usize,
}

/// Parameters for the AVX2 variant.
// #[derive(Clone, ParamGrid)]
#[derive(Clone)]
struct Avx2Params {
    /// Loop unroll factor
    // #[grid(1, 2, 4)]
    // #[default = 2]
    unroll: usize,

    /// Prefetch distance in cache lines
    // #[grid(0, 4, 8)]
    // #[default = 4]
    prefetch: usize,
}

// ============================================================================
// STEP 4: Implement the algorithm variants
// ============================================================================

/// Naive bit-by-bit rank implementation
fn rank_naive(data: &RankData) -> usize {
    let mut count = 0;
    for (word_idx, &word) in data.bitmap.iter().enumerate() {
        let word_start = word_idx * 64;
        let word_end = word_start + 64;

        if data.position < word_start {
            break;
        }

        if data.position >= word_end {
            // Count all bits in this word
            count += word.count_ones() as usize;
        } else {
            // Partial word - mask off bits beyond position
            let bits_to_count = data.position - word_start + 1;
            let mask = (1u64 << bits_to_count) - 1;
            count += (word & mask).count_ones() as usize;
            break;
        }
    }
    count
}

/// Chunked implementation with tunable chunk_size
fn rank_chunked(data: &RankData, params: &ChunkedParams) -> usize {
    let target_word = data.position / 64;
    let bit_in_word = data.position % 64;

    let mut count = 0;

    // Process full chunks
    let full_chunks = target_word / params.chunk_size;
    for chunk_idx in 0..full_chunks {
        let start = chunk_idx * params.chunk_size;
        let end = start + params.chunk_size;
        for &word in &data.bitmap[start..end] {
            count += word.count_ones() as usize;
        }
    }

    // Process remaining full words
    let remaining_start = full_chunks * params.chunk_size;
    for &word in &data.bitmap[remaining_start..target_word] {
        count += word.count_ones() as usize;
    }

    // Process partial final word
    if target_word < data.bitmap.len() {
        let word = data.bitmap[target_word];
        let mask = (1u64 << (bit_in_word + 1)) - 1;
        count += (word & mask).count_ones() as usize;
    }

    count
}

/// AVX2 implementation with tunable unroll and prefetch
#[cfg(target_arch = "x86_64")]
fn rank_avx2(data: &RankData, params: &Avx2Params) -> usize {
    // Simplified - real impl would use AVX2 intrinsics
    // The params would control loop unrolling and prefetch distance
    let _ = params.unroll;
    let _ = params.prefetch;
    rank_naive(data)
}

#[cfg(not(target_arch = "x86_64"))]
fn rank_avx2(data: &RankData, params: &Avx2Params) -> usize {
    rank_naive(data)
}

// ============================================================================
// STEP 5: Define the benchmark using the StatsBench builder API
// ============================================================================

/// Create a rank benchmark using the StatsBench API.
///
/// This demonstrates the Phase 2 stats-based API.
fn create_rank_benchmark() -> vortex_threshold_traits::BuiltStatsBench<RankData, RankStats, usize> {
    StatsBench::<RankData, RankStats, usize>::new("rank")
        // Stats computation: Data -> Stats
        .stats(RankStats::compute)
        // Data generation: Stats -> Data
        .generate(|stats: &RankStats, seed| stats.generate(seed))
        // Define the stats grid to search over
        .stats_grid(
            StatsGrid::new()
                .dimension("len", Scale::log2(6, 10)) // 64 to 1024 words
                .dimension("density", Scale::steps(0.0, 1.0, 3)), // 0%, 50%, 100%
        )
        // Baseline variant (used for correctness checking)
        .baseline("naive", rank_naive)
        // Chunked variant (with fixed params for now - ParamGrid derive is Phase 3)
        .variant("chunked", |data| {
            rank_chunked(data, &ChunkedParams { chunk_size: 4 })
        })
        .build()
}

// ============================================================================
// STEP 6: Run in bench mode (quick comparison with defaults)
// ============================================================================

/// Run a quick benchmark at a specific stats point.
fn run_bench_mode() {
    let benchmark = create_rank_benchmark();

    // Quick benchmark at a specific stats point
    let results = benchmark
        .bench()
        .at(StatsPoint::new().with("len", 1024.0).with("density", 0.5))
        .run(|point| RankStats {
            len: point.get_usize("len").unwrap_or(1024),
            density: point.get("density").unwrap_or(0.5),
        });

    // Print results
    results.print();
}

// ============================================================================
// STEP 7: Run in search mode (full grid search)
// ============================================================================

/// Run a full grid search over all stats combinations.
fn run_search_mode() {
    let benchmark = create_rank_benchmark();

    // Full grid search over all (stats × variant) combinations
    let results = benchmark.search().run(|point| RankStats {
        len: point.get_usize("len").unwrap_or(64),
        density: point.get("density").unwrap_or(0.5),
    });

    // Print results
    results.print();

    // Future: Save to JSON for CI
    // results.save("rank_thresholds.json").unwrap();

    // Future: Refine crossovers with binary search
    // let refined = benchmark.search().refine().run();
}

// ============================================================================
// STEP 8: CLI integration
// ============================================================================

/*
// In a binary crate (e.g., benches/rank_bench.rs):

use clap::{Parser, ValueEnum};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "bench")]
    mode: Mode,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long)]
    len: Option<usize>,

    #[arg(long)]
    density: Option<f64>,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    Bench,
    Search,
}

fn main() {
    let args = Args::parse();
    let benchmark = create_rank_benchmark();

    match args.mode {
        Mode::Bench => {
            let mut runner = benchmark.bench();

            if let (Some(len), Some(density)) = (args.len, args.density) {
                runner = runner.at(RankStats { len, density });
            }

            let results = runner.run();
            results.print();

            if let Some(path) = args.output {
                results.save(&path).unwrap();
            }
        }
        Mode::Search => {
            let results = benchmark.search().run();
            results.print();

            if let Some(path) = args.output {
                results.save(&path).unwrap();
            }
        }
    }
}

// Usage:
// $ cargo run --release -- --mode bench --len 1024 --density 0.5
// $ cargo run --release -- --mode search --output results.json
*/

// ============================================================================
// MAIN - Demonstrates the StatsBench API
// ============================================================================

fn main() {
    println!("StatsBench API Example");
    println!("======================");
    println!();

    // First, verify the algorithm is correct
    println!("1. Verifying algorithm correctness...");
    let data = RankStats {
        len: 100,
        density: 0.5,
    }
    .generate(42);

    let naive_result = rank_naive(&data);
    let chunked_result = rank_chunked(&data, &ChunkedParams { chunk_size: 4 });

    println!(
        "   Test data: {} words, position {}",
        data.bitmap.len(),
        data.position
    );
    println!("   naive result:   {}", naive_result);
    println!("   chunked result: {}", chunked_result);
    assert_eq!(naive_result, chunked_result, "Results should match!");
    println!("   Results match!");
    println!();

    // Now run the benchmark
    println!("2. Running bench mode (quick comparison at one point)...");
    println!();
    run_bench_mode();
    println!();

    // Optionally run search mode (takes longer)
    let run_search = std::env::args().any(|arg| arg == "--search");
    if run_search {
        println!("3. Running search mode (full grid search)...");
        println!();
        run_search_mode();
    } else {
        println!("3. Skipping search mode (run with --search to enable)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_correctness() {
        for seed in 0..10 {
            let stats = RankStats {
                len: 100,
                density: 0.5,
            };
            let data = stats.generate(seed);

            let naive = rank_naive(&data);
            let chunked = rank_chunked(&data, &ChunkedParams { chunk_size: 4 });

            assert_eq!(naive, chunked, "Mismatch at seed {}", seed);
        }
    }

    #[test]
    fn test_stats_computation() {
        let data = RankData {
            bitmap: vec![0xFFFF_FFFF_FFFF_FFFF; 10], // All ones
            position: 500,
        };

        let stats = RankStats::compute(&data);
        assert_eq!(stats.len, 10);
        assert!((stats.density - 1.0).abs() < 0.001);
    }
}
