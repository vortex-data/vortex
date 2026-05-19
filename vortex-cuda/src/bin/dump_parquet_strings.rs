// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dump each string column of a parquet file to a raw byte file, padded
//! to a multiple of `BLOCK_BYTES` so that the GPU-FSST (gtsst) bench can
//! consume it. Files go to `<out_dir>/<column_name>/<column_name>.bin`.
//!
//! Usage:
//!   dump_parquet_strings <input.parquet> <output_dir> [BLOCK_BYTES]
//!
//! Default BLOCK_BYTES = 1310720 (matches gtsst).

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use arrow_array::Array as ArrowArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const DEFAULT_BLOCK_BYTES: usize = 1_310_720;
// Cap per-column bytes well below 4 GiB (VarBinArray u32 offsets).
const MAX_BYTES: usize = 3_500_000_000;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dump_parquet_strings <input.parquet> <output_dir> [BLOCK_BYTES]");
        std::process::exit(2);
    }
    let input = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    let block_bytes: usize = if args.len() >= 4 {
        args[3].parse()?
    } else {
        DEFAULT_BLOCK_BYTES
    };

    fs::create_dir_all(&out_dir)?;

    println!("reading {}", input.display());
    let file = fs::File::open(&input)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut batches = Vec::new();
    for b in reader {
        batches.push(b?);
    }
    if batches.is_empty() {
        anyhow::bail!("no batches");
    }
    let schema = batches[0].schema();

    for (col_idx, field) in schema.fields().iter().enumerate() {
        let dt = field.data_type();
        let is_str = matches!(
            dt,
            arrow_schema::DataType::Utf8
                | arrow_schema::DataType::LargeUtf8
                | arrow_schema::DataType::Utf8View
        );
        if !is_str {
            continue;
        }
        let col_dir = out_dir.join(field.name());
        fs::create_dir_all(&col_dir)?;
        let bin_path = col_dir.join(format!("{}.bin", field.name()));
        let mut out = fs::File::create(&bin_path)?;
        let mut written: usize = 0;

        for b in &batches {
            let col = b.column(col_idx);
            macro_rules! handle {
                ($t:ty) => {{
                    let s = col.as_any().downcast_ref::<$t>().unwrap();
                    for i in 0..s.len() {
                        let bytes = s.value(i).as_bytes();
                        if written + bytes.len() > MAX_BYTES {
                            break;
                        }
                        out.write_all(bytes)?;
                        written += bytes.len();
                    }
                }};
            }
            if col.as_any().is::<arrow_array::StringArray>() {
                handle!(arrow_array::StringArray);
            } else if col.as_any().is::<arrow_array::LargeStringArray>() {
                handle!(arrow_array::LargeStringArray);
            } else if col.as_any().is::<arrow_array::StringViewArray>() {
                handle!(arrow_array::StringViewArray);
            }
            if written >= MAX_BYTES {
                break;
            }
        }

        // Pad up to next multiple of block_bytes.
        let pad = (block_bytes - (written % block_bytes)) % block_bytes;
        if pad > 0 {
            let zeros = vec![0u8; pad];
            out.write_all(&zeros)?;
        }
        out.flush()?;
        let total = written + pad;
        println!(
            "  {} : {} ({} raw + {} pad, block={})",
            field.name(),
            bin_path.display(),
            written,
            pad,
            block_bytes
        );
        // Skip tiny columns (<100KB raw).
        if written < 100_000 {
            // leave the file; gtsst will still attempt it
        }
        // Skip column directories under the bench's filter? Leave for user.
        let _ = total;
    }
    Ok(())
}
