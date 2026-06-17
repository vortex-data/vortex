// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FastLanes RLE roundtrip on real data.
//!
//! Loads the NYC yellow-taxi dataset, compresses every column with the default
//! `BtrBlocksCompressor`, and reports, per column:
//!  * the selected encoding tree,
//!  * whether the FastLanes RLE encoding (`fastlanes.rle`) was selected,
//!  * the compression ratio, and
//!  * decompression (canonicalization) throughput.
//!
//! Build with the `unstable_encodings` feature to exercise the RLE indices Delta
//! cascade that the default file writer uses:
//!
//! ```bash
//! cargo run -p compress-bench --release --features unstable_encodings \
//!     --example rle_roundtrip
//! ```

use std::hint::black_box;
use std::time::Instant;

use anyhow::Result;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::compressor::BtrBlocksCompressor;
use vortex::session::VortexSession;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_bench::datasets::taxi_data::taxi_data_parquet;

/// The FastLanes RLE encoding id.
const RLE_ID: &str = "fastlanes.rle";
/// Number of decompression iterations used to estimate throughput.
const DECOMPRESS_ITERS: u32 = 20;

#[tokio::main]
async fn main() -> Result<()> {
    let session = VortexSession::default();

    println!("Loading NYC taxi dataset (downloads ~56MB on first run)...");
    let parquet = taxi_data_parquet().await?;
    let chunked = parquet_to_vortex_chunks(parquet).await?;

    let mut ctx = session.create_execution_ctx();
    let struct_array = chunked
        .into_array()
        .execute::<Canonical>(&mut ctx)?
        .into_struct();
    let rows = struct_array.len();
    let names = struct_array.names().clone();

    println!("Rows: {rows}\n");
    println!(
        "{:<24} {:>12} {:>12} {:>8} {:>5} {:>14}",
        "column", "uncompressed", "compressed", "ratio", "rle?", "decompress"
    );
    println!("{}", "-".repeat(80));

    let mut rle_columns = Vec::new();

    for (idx, name) in names.iter().enumerate() {
        let field = struct_array.unmasked_field(idx).clone();
        let uncompressed = field.nbytes();

        let compressed = BtrBlocksCompressor::default().compress(&field, &mut ctx)?;
        let compressed_size = compressed.nbytes();
        let ratio = uncompressed as f64 / compressed_size.max(1) as f64;

        let uses_rle = tree_contains(&compressed, RLE_ID);

        // Decompression throughput: canonicalize the compressed column repeatedly.
        let start = Instant::now();
        for _ in 0..DECOMPRESS_ITERS {
            black_box(compressed.clone().execute::<Canonical>(&mut ctx)?);
        }
        let per_iter = start.elapsed() / DECOMPRESS_ITERS;
        let mb = uncompressed as f64 / (1024.0 * 1024.0);
        let throughput = mb / per_iter.as_secs_f64();

        println!(
            "{:<24} {:>12} {:>12} {:>7.2}x {:>5} {:>9.0} MB/s",
            truncate(name.as_ref(), 24),
            uncompressed,
            compressed_size,
            ratio,
            if uses_rle { "yes" } else { "" },
            throughput,
        );

        if uses_rle {
            rle_columns.push((name.to_string(), ratio, throughput, compressed.clone()));
        }
    }

    if rle_columns.is_empty() {
        println!("\nNo column selected FastLanes RLE for this dataset.");
        return Ok(());
    }

    println!("\n=== Columns that selected FastLanes RLE ===");
    for (name, ratio, throughput, compressed) in &rle_columns {
        println!(
            "\n{name}: column ratio {ratio:.2}x, column decompress {throughput:.0} MB/s\n  encoding tree:"
        );
        print_tree(compressed, 2);

        // Isolate every `fastlanes.rle` node and measure it on its own. This is the
        // compression ratio and decode throughput of the RLE encoding itself, decoupled
        // from the surrounding cascade.
        let mut rle_nodes = Vec::new();
        collect_nodes(compressed, RLE_ID, &mut rle_nodes);
        for (i, rle) in rle_nodes.iter().enumerate() {
            let canonical = rle.clone().execute::<Canonical>(&mut ctx)?.into_array();
            let decoded_bytes = canonical.nbytes();
            let node_ratio = decoded_bytes as f64 / rle.nbytes().max(1) as f64;

            let start = Instant::now();
            for _ in 0..DECOMPRESS_ITERS {
                black_box(rle.clone().execute::<Canonical>(&mut ctx)?);
            }
            let per_iter = start.elapsed() / DECOMPRESS_ITERS;
            let mb = decoded_bytes as f64 / (1024.0 * 1024.0);
            let throughput = mb / per_iter.as_secs_f64();
            let rows_per_s = rle.len() as f64 / per_iter.as_secs_f64();

            println!(
                "  RLE node #{i}: len {}, encoded {} -> decoded {} bytes ({:.2}x), \
                 decode {:.0} MB/s ({:.0} M rows/s)",
                rle.len(),
                rle.nbytes(),
                decoded_bytes,
                node_ratio,
                throughput,
                rows_per_s / 1e6,
            );
        }
    }

    Ok(())
}

/// Collects every node in the tree whose encoding id equals `id`.
fn collect_nodes(array: &ArrayRef, id: &str, out: &mut Vec<ArrayRef>) {
    if array.encoding_id().to_string() == id {
        out.push(array.clone());
    }
    for child in array.children() {
        collect_nodes(&child, id, out);
    }
}

/// Returns true if `id` appears anywhere in the array's encoding tree.
fn tree_contains(array: &ArrayRef, id: &str) -> bool {
    if array.encoding_id().to_string() == id {
        return true;
    }
    array.children().iter().any(|c| tree_contains(c, id))
}

/// Pretty-prints the encoding tree with sizes.
fn print_tree(array: &ArrayRef, indent: usize) {
    println!(
        "{}{} ({} bytes, len {})",
        " ".repeat(indent),
        array.encoding_id(),
        array.nbytes(),
        array.len(),
    );
    for (child_name, child) in array.named_children() {
        println!("{}- {child_name}:", " ".repeat(indent + 2));
        print_tree(&child, indent + 4);
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}
