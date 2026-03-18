// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark BtrBlocksCompressor on TPC-H SF1 data.
//!
//! Generates each TPC-H table, converts to Vortex arrays, and measures
//! compression time and ratio per table.

use std::time::Instant;

use anyhow::Result;
use tpchgen::generators::CustomerGenerator;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::NationGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen::generators::PartGenerator;
use tpchgen::generators::PartSuppGenerator;
use tpchgen::generators::RegionGenerator;
use tpchgen::generators::SupplierGenerator;
use tpchgen_arrow::RecordBatchIterator;
use vortex::array::ArrayRef;
use vortex::array::arrow::FromArrowArray;
use vortex::compressor::BtrBlocksCompressor;

const SCALE_FACTOR: f64 = 1.0;
const BATCH_SIZE: usize = 65536;

fn compress_table(
    name: &str,
    iter: Box<dyn RecordBatchIterator>,
    compressor: &BtrBlocksCompressor,
) -> Result<()> {
    let mut total_uncompressed: u64 = 0;
    let mut total_compressed: u64 = 0;
    let mut total_rows: usize = 0;
    let mut total_duration = std::time::Duration::ZERO;

    for batch in iter {
        let array = ArrayRef::from_arrow(&batch, false)?;
        let uncompressed_size = array.nbytes();

        let start = Instant::now();
        let compressed = compressor.compress(&array)?;
        let elapsed = start.elapsed();

        let compressed_size = compressed.nbytes();
        total_uncompressed += uncompressed_size;
        total_compressed += compressed_size;
        total_rows += batch.num_rows();
        total_duration += elapsed;
    }

    let ratio = total_uncompressed as f64 / total_compressed as f64;
    let throughput_mb = total_uncompressed as f64 / 1024.0 / 1024.0 / total_duration.as_secs_f64();

    println!(
        "{:<12} {:>10} rows  {:>8.1} MB -> {:>8.1} MB  ratio={:.2}x  {:.0}ms  {:.0} MB/s",
        name,
        total_rows,
        total_uncompressed as f64 / 1024.0 / 1024.0,
        total_compressed as f64 / 1024.0 / 1024.0,
        ratio,
        total_duration.as_millis(),
        throughput_mb,
    );

    Ok(())
}

fn main() -> Result<()> {
    let compressor = BtrBlocksCompressor::default();

    println!("TPC-H SF1 Compression Benchmark");
    println!("================================");
    println!(
        "{:<12} {:>14}  {:>21}  {:>10}  {:>6}  {:>8}",
        "Table", "Rows", "Size", "Ratio", "Time", "Throughput"
    );
    println!("{}", "-".repeat(85));

    // Generate and compress each table
    let tables: Vec<(&str, Box<dyn RecordBatchIterator>)> = vec![
        (
            "nation",
            Box::new(
                tpchgen_arrow::NationArrow::new(NationGenerator::new(SCALE_FACTOR, 1, 1))
                    .with_batch_size(BATCH_SIZE),
            ),
        ),
        (
            "region",
            Box::new(
                tpchgen_arrow::RegionArrow::new(RegionGenerator::new(SCALE_FACTOR, 1, 1))
                    .with_batch_size(BATCH_SIZE),
            ),
        ),
        (
            "part",
            Box::new(
                tpchgen_arrow::PartArrow::new(PartGenerator::new(SCALE_FACTOR, 1, 1))
                    .with_batch_size(BATCH_SIZE),
            ),
        ),
        (
            "supplier",
            Box::new(
                tpchgen_arrow::SupplierArrow::new(SupplierGenerator::new(SCALE_FACTOR, 1, 1))
                    .with_batch_size(BATCH_SIZE),
            ),
        ),
        (
            "customer",
            Box::new(
                tpchgen_arrow::CustomerArrow::new(CustomerGenerator::new(SCALE_FACTOR, 1, 1))
                    .with_batch_size(BATCH_SIZE),
            ),
        ),
        (
            "partsupp",
            Box::new(
                tpchgen_arrow::PartSuppArrow::new(PartSuppGenerator::new(SCALE_FACTOR, 1, 1))
                    .with_batch_size(BATCH_SIZE),
            ),
        ),
        (
            "orders",
            Box::new(
                tpchgen_arrow::OrderArrow::new(OrderGenerator::new(SCALE_FACTOR, 1, 1))
                    .with_batch_size(BATCH_SIZE),
            ),
        ),
        (
            "lineitem",
            Box::new(
                tpchgen_arrow::LineItemArrow::new(LineItemGenerator::new(SCALE_FACTOR, 1, 1))
                    .with_batch_size(BATCH_SIZE),
            ),
        ),
    ];

    for (name, iter) in tables {
        compress_table(name, iter, &compressor)?;
    }

    Ok(())
}
