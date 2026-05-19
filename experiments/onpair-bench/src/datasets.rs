// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use arrow_array::Array;
use arrow_array::StringArray;
use arrow_array::StringViewArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use tpchgen::generators::LineItemGenerator;

const DATA_DIR: &str = "/tmp/onpair-bench-data";

fn ensure_dir() -> Result<PathBuf> {
    let p = PathBuf::from(DATA_DIR);
    fs::create_dir_all(&p)?;
    Ok(p)
}

/// Load up to `max_rows` of TPC-H l_comment at SF=1. Generated in-process via
/// `tpchgen` so no network required. `skip` rows are discarded from the head
/// of the iterator so different slices can be sampled.
pub fn tpch_l_comment(max_rows: usize, skip: usize) -> Result<Vec<Vec<u8>>> {
    let generator = LineItemGenerator::new(1.0, 1, 1);
    let mut out = Vec::with_capacity(max_rows);
    for item in generator.iter().skip(skip) {
        if out.len() >= max_rows {
            break;
        }
        out.push(item.l_comment.as_bytes().to_vec());
    }
    Ok(out)
}

fn download_clickbench_partition(idx: usize) -> Result<PathBuf> {
    let dir = ensure_dir()?;
    let path = dir.join(format!("hits_{idx}.parquet"));
    if path.exists() {
        return Ok(path);
    }
    let url = format!("https://pub-3ba949c0f0354ac18db1f0f14f0a2c52.r2.dev/clickbench/parquet_many/hits_{idx}.parquet");
    eprintln!("Downloading {url} → {}", path.display());
    let resp = reqwest::blocking::get(&url)?.error_for_status()?;
    let bytes = resp.bytes()?;
    let tmp = path.with_extension("parquet.partial");
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
    }
    fs::rename(&tmp, &path)?;
    eprintln!("Downloaded {} bytes", bytes.len());
    Ok(path)
}

pub fn clickbench_column(col_name: &str, max_rows: usize, partition: usize) -> Result<Vec<Vec<u8>>> {
    let path = download_clickbench_partition(partition)?;
    let file = fs::File::open(&path).context("open hits parquet")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    // Find the column index.
    let schema = builder.schema();
    let (idx, _) = schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_, f)| f.name() == col_name)
        .ok_or_else(|| anyhow::anyhow!("column '{col_name}' not found"))?;

    let mask = ProjectionMask::leaves(builder.parquet_schema(), [idx]);
    let reader = builder.with_projection(mask).build()?;

    let mut out = Vec::with_capacity(max_rows);
    for batch in reader {
        let batch = batch?;
        let col = batch.column(0);
        if let Some(arr) = col.as_any().downcast_ref::<StringViewArray>() {
            for v in arr.iter() {
                if out.len() >= max_rows {
                    return Ok(out);
                }
                if let Some(s) = v {
                    out.push(s.as_bytes().to_vec());
                }
            }
        } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
            for v in arr.iter() {
                if out.len() >= max_rows {
                    return Ok(out);
                }
                if let Some(s) = v {
                    out.push(s.as_bytes().to_vec());
                }
            }
        } else {
            bail!("unexpected array type for '{col_name}'");
        }
    }
    Ok(out)
}

#[allow(dead_code)]
pub fn rows_bytes(rows: &[Vec<u8>]) -> usize {
    rows.iter().map(|r| r.len()).sum()
}

#[allow(dead_code)]
fn _silence(_: Arc<dyn Array>, _: &Path) {} // keep `Path` import used if datasets stop using it
