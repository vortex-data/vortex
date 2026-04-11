// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Write a synthetic parquet file matching the VectorDBBench `emb: list<f32>` + `id: int64`
//! schema. Useful for local dev runs of `vector-search-bench` without needing network
//! access to `assets.zilliz.com`, and for sandbox / CI environments that block outbound
//! HTTPS.
//!
//! The generated file is bit-identical across runs for a given `(num_rows, dim, seed)`
//! triple so that downstream benchmark output is reproducible.
//!
//! Example:
//!
//! ```bash
//! cargo run -p vector-search-bench --bin gen_synthetic_dataset --release -- \
//!     --num-rows 5000 \
//!     --dim 768 \
//!     --out vortex-bench/data/cohere-small/cohere-small.parquet
//! ```
//!
//! After running this, `vector-search-bench --datasets cohere-small` will find the
//! cached parquet file and skip the HTTP download via `idempotent_async`. (Cargo's
//! default bin name is the filename minus extension — underscores, not hyphens.)

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use arrow_array::Int64Array;
use arrow_array::ListArray;
use arrow_array::RecordBatch;
use arrow_array::builder::Float32Builder;
use arrow_array::builder::Int32BufferBuilder;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use clap::Parser;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Generate a synthetic VectorDBBench-style parquet file"
)]
struct Args {
    /// Number of rows to generate.
    #[arg(long, default_value_t = 5000)]
    num_rows: usize,

    /// Vector dimensionality. Must be ≥ 128 to exercise TurboQuant.
    #[arg(long, default_value_t = 768)]
    dim: u32,

    /// Deterministic PRNG seed — changing this changes the generated vectors.
    #[arg(long, default_value_t = 0xC0FFEE)]
    seed: u64,

    /// Output parquet file path. Parent directory is created if missing.
    #[arg(long)]
    out: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Build an Arrow `ListArray<f32>` so the schema matches VectorDBBench's `emb:
    // list<float>` (note: NOT fixed_size_list — parquet has no FSL logical type so
    // arrow-rs writes lists). Every list has exactly `dim` elements.
    let dim_usize = args.dim as usize;
    let total_elements = args.num_rows * dim_usize;

    let mut float_values = Float32Builder::with_capacity(total_elements);
    let mut offsets = Int32BufferBuilder::new(args.num_rows + 1);
    offsets.append(0i32);

    let mut state = args.seed.wrapping_add(1);
    for row in 0..args.num_rows {
        for i in 0..dim_usize {
            // Deterministic xorshift mixed with position so every vector is distinct.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let scale = 1.0f32 / 32768.0;
            let v = ((state & 0xFFFF) as f32 * scale - 0.5)
                + ((row as f32 * 0.00013) + (i as f32 * 0.00007)).sin() * 0.25;
            float_values.append_value(v);
        }
        let written = i32::try_from((row + 1) * dim_usize)
            .context("offset overflows i32 — reduce num_rows or dim")?;
        offsets.append(written);
    }

    let values_array = float_values.finish();
    let offsets_buffer = offsets.finish();

    let field = Arc::new(Field::new("item", DataType::Float32, false));
    let list_dtype = DataType::List(Arc::clone(&field));
    let list_array = ListArray::try_new(
        Arc::clone(&field),
        arrow_buffer::OffsetBuffer::new(offsets_buffer.into()),
        Arc::new(values_array),
        None,
    )?;

    let ids: Int64Array = (0..args.num_rows as i64).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("emb", list_dtype, false),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(ids), Arc::new(list_array)],
    )?;

    let writer_props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let file = File::create(&args.out)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(writer_props))?;
    writer.write(&batch)?;
    writer.close()?;

    println!(
        "wrote {} rows × {} dims to {}",
        args.num_rows,
        args.dim,
        args.out.display()
    );
    Ok(())
}
