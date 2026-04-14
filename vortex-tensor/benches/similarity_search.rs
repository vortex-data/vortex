// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end similarity-search execution benchmark.
//!
//! For each of three compression strategies (uncompressed, default BtrBlocks, TurboQuant), this
//! bench:
//!
//! 1. Generates a deterministic random `Vector<dim, f32>` batch.
//! 2. Applies the compression strategy *outside* the timed region.
//! 3. Builds the lazy
//!    `Binary(Gt, [CosineSimilarity(data, query), threshold])`
//!    tree *outside* the timed region.
//! 4. Times *only* `tree.execute::<BoolArray>(&mut ctx)`.
//!
//! Run with: `cargo bench -p vortex-tensor --bench similarity_search`

use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_error::VortexExpect;

#[path = "similarity_search_common/mod.rs"]
mod common;

use common::Variant;
use common::build_similarity_search_tree;
use common::extract_row_as_query;
use common::generate_random_vectors;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Number of vectors in the benchmark dataset.
const NUM_ROWS: usize = 10_000;

/// Dimensionality of each vector. Must be `>= vortex_tensor::encodings::turboquant::MIN_DIMENSION`
/// (128) for the TurboQuant variant to work.
const DIM: u32 = 768;

/// Deterministic PRNG seed for the generated dataset.
const SEED: u64 = 0xC0FFEE;

/// Cosine similarity threshold for the "greater than" filter. Random f32 vectors from N(0, 1) at
/// this dimension have near-zero pairwise similarity, so picking a row of the dataset as the
/// query guarantees at least that row matches.
const THRESHOLD: f32 = 0.8;

fn main() {
    divan::main();
}

/// Runs one end-to-end execution of the similarity-search tree for the given variant. All dataset
/// generation and tree construction happens in the bench setup closure so only the execution of
/// the lazy tree is timed.
fn bench_variant(bencher: Bencher<'_, '_>, variant: Variant) {
    bencher
        .with_inputs(|| {
            let mut ctx = common::SESSION.create_execution_ctx();

            // Use row 0 of the uncompressed data as the query so we always have at least one
            // match. Keeping the query extraction separate from the compressed-data build keeps
            // the query identical across all three variants.
            let raw = generate_random_vectors(NUM_ROWS, DIM, SEED);
            let query = extract_row_as_query(&raw, 0, DIM);
            let data = match variant {
                Variant::Uncompressed => raw,
                Variant::DefaultCompression => {
                    common::compress_default(raw).vortex_expect("default compression succeeds")
                }
                Variant::TurboQuant => common::compress_turboquant(raw, &mut ctx)
                    .vortex_expect("turboquant compression succeeds"),
            };

            // println!(
            //     "\n\n{}: {}\n\n",
            //     variant,
            //     data.display_tree_encodings_only()
            // );

            let tree = build_similarity_search_tree(data, &query, THRESHOLD)
                .vortex_expect("tree construction succeeds");

            (tree, ctx)
        })
        .bench_values(|(tree, mut ctx)| {
            // Hot path: only the .execute() call is timed. The result is a BoolArray of length
            // NUM_ROWS with true at positions where cosine_similarity > THRESHOLD.
            tree.execute::<BoolArray>(&mut ctx)
                .vortex_expect("similarity search tree executes to a BoolArray")
        });
}

#[divan::bench]
fn execute_uncompressed(bencher: Bencher) {
    bench_variant(bencher, Variant::Uncompressed);
}

#[divan::bench]
fn execute_default_compression(bencher: Bencher) {
    bench_variant(bencher, Variant::DefaultCompression);
}

#[divan::bench]
fn execute_turboquant(bencher: Bencher) {
    bench_variant(bencher, Variant::TurboQuant);
}
