// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(
    clippy::cast_possible_truncation,
    clippy::clone_on_ref_ptr,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::unwrap_used
)]
//
// Generate a single-column parquet of the real TPC-H `lineitem.l_comment`
// values, sized to a target number of bytes. Used to feed `bench_tpch` real
// data instead of the synthetic corpus:
//
//   cargo run --release -p vortex-onpair-rs --example gen_l_comment
//   ONPAIR_BENCH_PARQUET=<out> ONPAIR_BENCH_COLUMN=l_comment \
//     cargo run --release -p vortex-onpair-rs --example bench_tpch
//
// Env:
//   * `OUT`            — output parquet path (default `target/l_comment.parquet`)
//   * `TARGET_BYTES`   — stop once this many comment bytes are written
//                        (default 1.1 GiB so a 1 GiB bench cap is fully filled)
//   * `SCALE_FACTOR`   — TPC-H scale factor to draw from (default 8.0)

use std::env;
use std::fs::File;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::StringViewArray;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tpchgen::generators::LineItemGenerator;
use tpchgen_arrow::LineItemArrow;
use tpchgen_arrow::RecordBatchIterator;

fn main() {
    let out = env::var("OUT").unwrap_or_else(|_| "target/l_comment.parquet".to_string());
    let target_bytes = env::var("TARGET_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or((1u64 << 30) as usize + (100 << 20));
    let sf = env::var("SCALE_FACTOR")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(8.0);

    let schema = Arc::new(Schema::new(vec![Field::new(
        "l_comment",
        DataType::Utf8View,
        false,
    )]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let file = File::create(&out).expect("create output");
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).expect("writer");

    let batches = LineItemArrow::new(LineItemGenerator::new(sf, 1, 1)).with_batch_size(8192 * 8);
    let comment_idx = batches
        .schema()
        .fields()
        .iter()
        .position(|f| f.name() == "l_comment")
        .expect("l_comment column");

    let t = Instant::now();
    let mut written_bytes = 0usize;
    let mut rows = 0usize;
    for batch in batches {
        let col = batch.column(comment_idx).as_string_view();
        // Project just l_comment, truncating the final batch at the target.
        let mut values: Vec<&str> = Vec::with_capacity(col.len());
        let mut hit_target = false;
        for v in col.iter() {
            let s = v.unwrap_or("");
            written_bytes += s.len();
            rows += 1;
            values.push(s);
            if written_bytes >= target_bytes {
                hit_target = true;
                break;
            }
        }
        let arr = StringViewArray::from_iter_values(values.iter().copied());
        let out_batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).expect("batch");
        writer.write(&out_batch).expect("write");
        if hit_target {
            break;
        }
    }
    writer.close().expect("close");

    println!(
        "wrote {out}: rows = {rows}, comment bytes = {:.2} MiB ({:.1} MiB/s gen)",
        written_bytes as f64 / (1024.0 * 1024.0),
        written_bytes as f64 / (1024.0 * 1024.0) / t.elapsed().as_secs_f64(),
    );
}
