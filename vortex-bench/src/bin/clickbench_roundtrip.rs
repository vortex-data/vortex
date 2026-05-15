// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Roundtrip-and-size harness for Parquet shards.
//!
//! For each input parquet, writes it out as a Vortex file in one or more
//! modes (`--default`, `--compact`, `--fsst-only`), reads it back through
//! the original parquet arrow schema, and asserts every column matches the
//! source parquet row-for-row. Prints a summary of on-disk sizes with each
//! non-FSST mode's delta vs the FSST-only baseline, so you can quickly
//! check claims like "default cascade is X% smaller than FSST".

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Date64Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use arrow_schema::{DataType, SchemaRef};
use arrow_select::concat::concat;
use futures::StreamExt;
use futures::pin_mut;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use tokio::fs::File as TokioFile;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex_btrblocks::schemes::string::FSSTScheme;
use vortex_bench::SESSION;
use vortex_bench::conversions::parquet_to_vortex_stream;

fn human(b: u64) -> String {
    let kb = 1024f64;
    let mb = kb * 1024.0;
    let gb = mb * 1024.0;
    let bf = b as f64;
    if bf >= gb {
        format!("{:.2} GiB", bf / gb)
    } else if bf >= mb {
        format!("{:.2} MiB", bf / mb)
    } else if bf >= kb {
        format!("{:.2} KiB", bf / kb)
    } else {
        format!("{b} B")
    }
}

#[derive(Copy, Clone, Debug)]
enum Mode {
    Default,
    Compact,
    FsstOnly,
}

async fn write_vortex(parquet: &Path, out: &Path, mode: Mode) -> Result<u64> {
    let f = TokioFile::open(parquet).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(f).await?;
    let dtype = DType::from_arrow(builder.schema().as_ref());
    let stream = parquet_to_vortex_stream(builder.build()?);

    let mut out_f = tokio::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(out)
        .await?;

    let strategy = match mode {
        Mode::Default => WriteStrategyBuilder::default().build(),
        Mode::Compact => WriteStrategyBuilder::default()
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::default().with_compact())
            .build(),
        Mode::FsstOnly => WriteStrategyBuilder::default()
            .with_btrblocks_builder(BtrBlocksCompressorBuilder::empty().with_new_scheme(&FSSTScheme))
            .build(),
    };

    SESSION
        .write_options()
        .with_strategy(strategy)
        .write(&mut out_f, ArrayStreamAdapter::new(dtype, stream))
        .await?;
    drop(out_f);
    Ok(tokio::fs::metadata(out).await?.len())
}

fn mode_label(mode: Mode) -> &'static str {
    match mode {
        Mode::Default => "default",
        Mode::Compact => "compact",
        Mode::FsstOnly => "fsst-only",
    }
}

async fn read_parquet(path: &Path) -> Result<(SchemaRef, Vec<RecordBatch>)> {
    let f = TokioFile::open(path).await?;
    let b = ParquetRecordBatchStreamBuilder::new(f).await?;
    let schema = b.schema().clone();
    let stream = b.build()?;
    pin_mut!(stream);
    let mut out = Vec::new();
    while let Some(rb) = stream.next().await {
        out.push(rb.context("parquet decode")?);
    }
    Ok((schema, out))
}

async fn read_vortex(path: &Path, schema: SchemaRef) -> Result<Vec<RecordBatch>> {
    let file = SESSION.open_options().open_path(path).await?;
    let scan = file.scan()?;
    let stream = scan.into_record_batch_stream(schema)?;
    pin_mut!(stream);
    let mut out = Vec::new();
    while let Some(rb) = stream.next().await {
        out.push(rb?);
    }
    Ok(out)
}

fn concat_col(batches: &[RecordBatch], col: usize) -> Result<ArrayRef> {
    let arrs: Vec<&dyn Array> = batches.iter().map(|b| b.column(col).as_ref()).collect();
    Ok(concat(&arrs)?)
}

fn first_mismatch(name: &str, e: &ArrayRef, a: &ArrayRef) -> Result<Option<(usize, String, String)>> {
    if e.len() != a.len() {
        anyhow::bail!("column {name}: length differs - {} vs {}", e.len(), a.len());
    }
    if e.data_type() != a.data_type() {
        anyhow::bail!(
            "column {name}: dtype differs - {:?} vs {:?}",
            e.data_type(),
            a.data_type()
        );
    }
    let len = e.len();
    macro_rules! prim {
        ($t:ty) => {{
            let e = e.as_primitive::<$t>();
            let a = a.as_primitive::<$t>();
            for i in 0..len {
                if e.is_null(i) != a.is_null(i) {
                    return Ok(Some((i, format!("null={}", e.is_null(i)), format!("null={}", a.is_null(i)))));
                }
                if !e.is_null(i) && e.value(i) != a.value(i) {
                    return Ok(Some((i, format!("{:?}", e.value(i)), format!("{:?}", a.value(i)))));
                }
            }
            return Ok(None);
        }};
    }
    match e.data_type() {
        DataType::Int8 => prim!(Int8Type),
        DataType::Int16 => prim!(Int16Type),
        DataType::Int32 => prim!(Int32Type),
        DataType::Int64 => prim!(Int64Type),
        DataType::UInt8 => prim!(UInt8Type),
        DataType::UInt16 => prim!(UInt16Type),
        DataType::UInt32 => prim!(UInt32Type),
        DataType::UInt64 => prim!(UInt64Type),
        DataType::Float32 => prim!(Float32Type),
        DataType::Float64 => prim!(Float64Type),
        DataType::Date32 => prim!(Date32Type),
        DataType::Date64 => prim!(Date64Type),
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => prim!(TimestampMicrosecondType),
        DataType::Timestamp(arrow_schema::TimeUnit::Millisecond, _) => prim!(TimestampMillisecondType),
        DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => prim!(TimestampNanosecondType),
        DataType::Timestamp(arrow_schema::TimeUnit::Second, _) => prim!(TimestampSecondType),
        DataType::Boolean => {
            let e = e.as_boolean();
            let a = a.as_boolean();
            for i in 0..len {
                if e.is_null(i) != a.is_null(i) {
                    return Ok(Some((i, format!("null={}", e.is_null(i)), format!("null={}", a.is_null(i)))));
                }
                if !e.is_null(i) && e.value(i) != a.value(i) {
                    return Ok(Some((i, format!("{:?}", e.value(i)), format!("{:?}", a.value(i)))));
                }
            }
            Ok(None)
        }
        DataType::Utf8 => {
            let e = e.as_string::<i32>();
            let a = a.as_string::<i32>();
            for i in 0..len {
                if e.is_null(i) != a.is_null(i) {
                    return Ok(Some((i, format!("null={}", e.is_null(i)), format!("null={}", a.is_null(i)))));
                }
                if !e.is_null(i) && e.value(i) != a.value(i) {
                    let ev = e.value(i);
                    let av = a.value(i);
                    return Ok(Some((
                        i,
                        format!("{:?} (len {})", ev, ev.len()),
                        format!("{:?} (len {})", av, av.len()),
                    )));
                }
            }
            Ok(None)
        }
        DataType::LargeUtf8 => {
            let e = e.as_string::<i64>();
            let a = a.as_string::<i64>();
            for i in 0..len {
                if e.is_null(i) != a.is_null(i) {
                    return Ok(Some((i, format!("null={}", e.is_null(i)), format!("null={}", a.is_null(i)))));
                }
                if !e.is_null(i) && e.value(i) != a.value(i) {
                    return Ok(Some((i, format!("{:?}", e.value(i)), format!("{:?}", a.value(i)))));
                }
            }
            Ok(None)
        }
        DataType::Utf8View => {
            let e = e.as_string_view();
            let a = a.as_string_view();
            for i in 0..len {
                if e.is_null(i) != a.is_null(i) {
                    return Ok(Some((i, format!("null={}", e.is_null(i)), format!("null={}", a.is_null(i)))));
                }
                if !e.is_null(i) && e.value(i) != a.value(i) {
                    return Ok(Some((i, format!("{:?}", e.value(i)), format!("{:?}", a.value(i)))));
                }
            }
            Ok(None)
        }
        DataType::Binary => {
            let e = e.as_binary::<i32>();
            let a = a.as_binary::<i32>();
            for i in 0..len {
                if e.is_null(i) != a.is_null(i) {
                    return Ok(Some((i, format!("null={}", e.is_null(i)), format!("null={}", a.is_null(i)))));
                }
                if !e.is_null(i) && e.value(i) != a.value(i) {
                    return Ok(Some((
                        i,
                        format!("bytes len={}", e.value(i).len()),
                        format!("bytes len={}", a.value(i).len()),
                    )));
                }
            }
            Ok(None)
        }
        other => anyhow::bail!("column {name}: unsupported arrow type for comparison: {other:?}"),
    }
}

async fn roundtrip_one(parquet: &Path, mode: Mode) -> Result<u64> {
    let t0 = Instant::now();
    let parquet_size = tokio::fs::metadata(parquet).await?.len();
    eprintln!(
        "\n=== {} ({}) [{}] ===",
        parquet.display(),
        human(parquet_size),
        mode_label(mode),
    );

    let vortex_path = parquet.with_extension(format!("{}.vortex", mode_label(mode)));

    let t = Instant::now();
    let vortex_size = write_vortex(parquet, &vortex_path, mode).await?;
    eprintln!(
        "  write vortex: {} -> {} ({:.2}%) in {:?}",
        human(parquet_size),
        human(vortex_size),
        100.0 * vortex_size as f64 / parquet_size as f64,
        t.elapsed()
    );

    let t = Instant::now();
    let (pq_schema, pq_batches) = read_parquet(parquet).await?;
    let pq_rows: usize = pq_batches.iter().map(|b| b.num_rows()).sum();
    eprintln!(
        "  read parquet:  {} batches, {} rows in {:?}",
        pq_batches.len(),
        pq_rows,
        t.elapsed()
    );

    let t = Instant::now();
    let vx_batches = read_vortex(&vortex_path, Arc::clone(&pq_schema)).await?;
    let vx_rows: usize = vx_batches.iter().map(|b| b.num_rows()).sum();
    eprintln!(
        "  read vortex:   {} batches, {} rows in {:?}",
        vx_batches.len(),
        vx_rows,
        t.elapsed()
    );

    if pq_rows != vx_rows {
        anyhow::bail!("row count mismatch: parquet={pq_rows} vortex={vx_rows}");
    }

    let t = Instant::now();
    let ncols = pq_schema.fields().len();
    let mut bad = Vec::new();
    for col in 0..ncols {
        let name = pq_schema.field(col).name().clone();
        let e = concat_col(&pq_batches, col)?;
        let a = concat_col(&vx_batches, col)?;
        if let Some((idx, exp, got)) = first_mismatch(&name, &e, &a)? {
            eprintln!(
                "  MISMATCH col {col:>3} {name:<28} row {idx}: expected={exp} got={got}"
            );
            bad.push(name);
        }
    }
    eprintln!(
        "  compared {ncols} columns, {} mismatched, in {:?}",
        bad.len(),
        t.elapsed()
    );
    eprintln!("  total: {:?}", t0.elapsed());

    if !bad.is_empty() {
        anyhow::bail!("ROUNDTRIP FAILED for {}: {} column(s)", parquet.display(), bad.len());
    }
    Ok(vortex_size)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let mut modes: Vec<Mode> = Vec::new();
    let mut paths: Vec<String> = Vec::new();
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "--compact" => modes.push(Mode::Compact),
            "--fsst-only" => modes.push(Mode::FsstOnly),
            "--default" => modes.push(Mode::Default),
            "-h" | "--help" => {
                eprintln!(
                    "usage: clickbench-roundtrip [--default] [--compact] [--fsst-only] <parquet> [<parquet> ...]\n\
                     Multiple mode flags may be passed; each is run independently and a summary is printed.\n\
                     Defaults to --default if no mode flag is given."
                );
                return Ok(());
            }
            _ => paths.push(a),
        }
    }
    if paths.is_empty() {
        anyhow::bail!("see --help");
    }
    if modes.is_empty() {
        modes.push(Mode::Default);
    }

    let mut summary: Vec<(String, Mode, u64, u64)> = Vec::new();
    let mut failed = 0usize;
    for p in &paths {
        let path = PathBuf::from(p);
        let parquet_size = tokio::fs::metadata(&path).await.ok().map(|m| m.len()).unwrap_or(0);
        for mode in &modes {
            match roundtrip_one(&path, *mode).await {
                Ok(vx_size) => {
                    summary.push((p.clone(), *mode, parquet_size, vx_size));
                }
                Err(e) => {
                    eprintln!("ERROR ({}): {e:?}", mode_label(*mode));
                    failed += 1;
                }
            }
        }
    }

    eprintln!("\n=== SUMMARY ===");
    eprintln!(
        "{:<48} {:>14} {:>14} {:>14} {:>10}",
        "file [mode]", "parquet", "vortex", "ratio", "vs FSST"
    );
    // Look up FSST sizes per file for comparison.
    let fsst_size_for = |file: &str| -> Option<u64> {
        summary
            .iter()
            .find(|(p, m, _, _)| p == file && matches!(m, Mode::FsstOnly))
            .map(|(_, _, _, s)| *s)
    };
    for (file, mode, pq, vx) in &summary {
        let ratio = *vx as f64 / *pq as f64;
        let vs_fsst = if !matches!(mode, Mode::FsstOnly) {
            fsst_size_for(file)
                .map(|f| format!("{:+.1}%", 100.0 * (*vx as f64 / f as f64 - 1.0)))
                .unwrap_or_else(|| "-".into())
        } else {
            "-".into()
        };
        eprintln!(
            "{:<48} {:>14} {:>14} {:>13.2}x {:>10}",
            format!("{} [{}]", file, mode_label(*mode)),
            human(*pq),
            human(*vx),
            ratio,
            vs_fsst,
        );
    }

    if failed > 0 {
        anyhow::bail!("{failed} run(s) failed");
    }
    Ok(())
}
