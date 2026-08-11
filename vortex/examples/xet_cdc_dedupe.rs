// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measures how well Vortex files deduplicate under Hugging Face Xet's content-defined
//! chunking, comparing the default write strategy against the experimental
//! content-defined chunking (CDC) write mode.
//!
//! Xet-backed storage (e.g. the Hugging Face Hub) splits every uploaded file into ~64 KiB
//! chunks with a GEAR rolling hash and stores each distinct chunk once. This example writes a
//! dataset, derives edited versions of it (append / insert / delete / update), writes each
//! version with both strategies, and reports how many of the new file's Xet chunks already
//! exist in the previous version — i.e. how many bytes an upload would actually transfer.
//!
//! Run with: cargo run --release --example xet_cdc_dedupe
//!
//! Writes a JSON report with per-chunk maps to `target/xet_cdc_dedupe.json` (override with the
//! `XET_CDC_OUT` environment variable).

use std::fs;
use std::time::Instant;

use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::layout::layouts::cdc::ContentDefinedChunkingOptions;
use vortex::layout::layouts::cdc::xet::xet_chunks;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_set::HashSet;

const BASE_ROWS: usize = 1_500_000;
const CATEGORIES: [&str; 8] = [
    "checkout", "search", "browse", "login", "logout", "purchase", "refund", "support",
];

/// A tiny deterministic PRNG so the demo is reproducible without extra dependencies.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// One logical record. Edits keep untouched records bit-identical, the way dataset revisions
/// on a data hub usually do.
#[derive(Clone)]
struct Row {
    id: u64,
    timestamp: i64,
    category: &'static str,
    value: f64,
    payload: String,
}

fn make_rows(seed: u64, first_id: u64, count: usize) -> Vec<Row> {
    let mut rng = SplitMix64(seed);
    (0..count)
        .map(|i| {
            let r = rng.next();
            Row {
                id: first_id + i as u64,
                timestamp: 1_700_000_000_000
                    + (first_id + i as u64) as i64 * 1_000
                    + (r % 997) as i64,
                category: CATEGORIES[usize::try_from(r % CATEGORIES.len() as u64).unwrap_or(0)],
                // A float in [1, 2): high-entropy mantissa, realistic "measurement" column.
                value: f64::from_bits((r >> 12) | (1023u64 << 52)),
                payload: format!("evt-{:012x}", r & 0xFFFF_FFFF_FFFF),
            }
        })
        .collect()
}

fn to_struct_array(rows: &[Row]) -> VortexResult<StructArray> {
    StructArray::from_fields(&[
        (
            "id",
            PrimitiveArray::from_iter(rows.iter().map(|r| r.id)).into_array(),
        ),
        (
            "timestamp",
            PrimitiveArray::from_iter(rows.iter().map(|r| r.timestamp)).into_array(),
        ),
        (
            "category",
            VarBinViewArray::from_iter_str(rows.iter().map(|r| r.category)).into_array(),
        ),
        (
            "value",
            PrimitiveArray::from_iter(rows.iter().map(|r| r.value)).into_array(),
        ),
        (
            "payload",
            VarBinViewArray::from_iter_str(rows.iter().map(|r| r.payload.as_str())).into_array(),
        ),
    ])
}

async fn write_file(
    session: &VortexSession,
    cdc: bool,
    rows: &[Row],
) -> VortexResult<(Vec<u8>, Vec<u64>)> {
    let strategy = if cdc {
        WriteStrategyBuilder::default()
            .with_content_defined_chunking(ContentDefinedChunkingOptions::default())
            .build()
    } else {
        WriteStrategyBuilder::default().build()
    };
    let mut buf = ByteBufferMut::empty();
    let summary = session
        .write_options()
        .with_strategy(strategy)
        .write(
            &mut buf,
            to_struct_array(rows)?.into_array().to_array_stream(),
        )
        .await?;
    let column_sizes = summary.compressed_column_sizes()?;
    Ok((buf.freeze().as_slice().to_vec(), column_sizes))
}

struct ChunkMap {
    /// (start, len, shared-with-the-other-version) per Xet chunk.
    chunks: Vec<(usize, usize, bool)>,
    shared_bytes: usize,
    total_bytes: usize,
}

/// Split `data` into Xet chunks and mark each chunk that also occurs in `other`.
fn chunk_map(data: &[u8], other_chunk_set: &HashSet<&[u8]>) -> ChunkMap {
    let mut chunks = Vec::new();
    let mut shared_bytes = 0usize;
    for range in xet_chunks(data) {
        let shared = other_chunk_set.contains(&data[range.clone()]);
        if shared {
            shared_bytes += range.len();
        }
        chunks.push((range.start, range.len(), shared));
    }
    ChunkMap {
        chunks,
        shared_bytes,
        total_bytes: data.len(),
    }
}

fn chunk_set(data: &[u8]) -> HashSet<&[u8]> {
    xet_chunks(data).into_iter().map(|r| &data[r]).collect()
}

fn chunks_json(map: &ChunkMap) -> serde_json::Value {
    serde_json::Value::Array(
        map.chunks
            .iter()
            .map(|(start, len, shared)| serde_json::json!([start, len, u8::from(*shared)]))
            .collect(),
    )
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let session = VortexSession::default();

    println!("generating {BASE_ROWS} base rows...");
    let base = make_rows(42, 0, BASE_ROWS);
    let next_id = BASE_ROWS as u64;

    // Each scenario derives an edited second version of the base dataset. Untouched rows keep
    // their exact content: only positions shift.
    let scenarios: Vec<(&str, Vec<Row>)> = vec![
        ("rewrite", base.clone()),
        ("append 2%", {
            let mut rows = base.clone();
            rows.extend(make_rows(43, next_id, BASE_ROWS / 50));
            rows
        }),
        ("insert 10k @ 40%", {
            let mut rows = base.clone();
            let at = 2 * BASE_ROWS / 5;
            rows.splice(at..at, make_rows(44, next_id, 10_000));
            rows
        }),
        ("delete 10k @ 60%", {
            let mut rows = base.clone();
            let at = 3 * BASE_ROWS / 5;
            rows.drain(at..at + 10_000);
            rows
        }),
        ("update 10k @ 25%", {
            let mut rows = base.clone();
            let at = BASE_ROWS / 4;
            let mut rng = SplitMix64(45);
            for row in &mut rows[at..at + 10_000] {
                row.value = f64::from_bits((rng.next() >> 12) | (1023u64 << 52));
            }
            rows
        }),
    ];

    let mut report = Vec::new();
    println!(
        "\n{:<10} {:<18} {:>9} {:>9} {:>12} {:>10}",
        "writer", "scenario", "v1", "v2", "new bytes", "deduped"
    );

    for cdc in [false, true] {
        let writer_name = if cdc { "cdc" } else { "baseline" };
        let started = Instant::now();
        let (v1, v1_column_sizes) = write_file(&session, cdc, &base).await?;
        let write_secs = started.elapsed().as_secs_f64();

        // Sanity: the written file must read back with the right number of rows.
        let row_count = session
            .open_options()
            .open_buffer(vortex::buffer::ByteBuffer::from(v1.clone()))?
            .row_count();
        assert_eq!(row_count, BASE_ROWS as u64);

        println!(
            "[{writer_name}] v1: {:.1} MiB written in {write_secs:.1}s",
            v1.len() as f64 / (1 << 20) as f64,
        );

        for (name, rows) in &scenarios {
            let (v2, _) = write_file(&session, cdc, rows).await?;
            if let Ok(dir) = std::env::var("XET_CDC_DUMP") {
                fs::create_dir_all(&dir)?;
                let tag = name.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
                fs::write(format!("{dir}/{writer_name}_v1.vortex"), &v1)?;
                fs::write(format!("{dir}/{writer_name}_v2_{tag}.vortex"), &v2)?;
            }

            let v1_set = chunk_set(&v1);
            let v2_set = chunk_set(&v2);
            let v2_map = chunk_map(&v2, &v1_set);
            let v1_map = chunk_map(&v1, &v2_set);

            let new_bytes = v2_map.total_bytes - v2_map.shared_bytes;
            let dedup_pct = 100.0 * v2_map.shared_bytes as f64 / v2_map.total_bytes as f64;
            println!(
                "{:<10} {:<18} {:>7.1}MB {:>7.1}MB {:>10.2}MB {:>9.1}%",
                writer_name,
                name,
                v1.len() as f64 / 1e6,
                v2.len() as f64 / 1e6,
                new_bytes as f64 / 1e6,
                dedup_pct,
            );

            report.push(serde_json::json!({
                "writer": writer_name,
                "scenario": name,
                "v1_bytes": v1.len(),
                "v2_bytes": v2.len(),
                "shared_bytes": v2_map.shared_bytes,
                "new_bytes": new_bytes,
                "dedup_pct": dedup_pct,
                "v1_column_sizes": v1_column_sizes,
                "v1_chunks": chunks_json(&v1_map),
                "v2_chunks": chunks_json(&v2_map),
            }));
        }
    }

    let out_path =
        std::env::var("XET_CDC_OUT").unwrap_or_else(|_| "target/xet_cdc_dedupe.json".to_string());
    fs::write(&out_path, serde_json::to_vec_pretty(&report)?)?;
    println!("\nwrote chunk-level report to {out_path}");
    Ok(())
}
