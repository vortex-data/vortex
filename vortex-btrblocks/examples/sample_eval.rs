// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! EXPERIMENT ONLY: compress real columns block by block and report sizes and timings.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss, clippy::use_debug)]

use std::fs::File;
use std::time::Duration;
use std::time::Instant;

use arrow_array::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::OrderGenerator;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::string::OnPairScheme;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_session::VortexSession;

const ONE_MEG: usize = 1 << 20;
const ROW_BLOCK: usize = 8192;

fn session() -> VortexSession {
    let session = VortexSession::empty().with::<EditionSession>();
    vortex_alp::initialize(&session);
    vortex_datetime_parts::initialize(&session);
    vortex_decimal_byte_parts::initialize(&session);
    vortex_fastlanes::initialize(&session);
    vortex_fsst::initialize(&session);
    vortex_onpair::initialize(&session);
    vortex_sequence::initialize(&session);
    vortex_sparse::initialize(&session);
    vortex_zigzag::initialize(&session);
    for family in EDITION_FAMILIES {
        session.editions().declare_family(family).unwrap();
    }
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration).unwrap();
    }
    session.enable_edition(CORE_2026_08_3).unwrap();
    session
}

fn columns(batches: Vec<RecordBatch>) -> Vec<(String, ArrayRef)> {
    let schema = batches[0].schema();
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let chunks = batches
                .iter()
                .map(|b| vortex_arrow::from_arrow_dyn(b.column(i).as_ref(), f.is_nullable()))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let dtype = chunks[0].dtype().clone();
            (
                f.name().clone(),
                ChunkedArray::try_new(chunks, dtype).unwrap().into_array(),
            )
        })
        .collect()
}

fn parquet(path: &str, max_rows: usize) -> Vec<(String, ArrayRef)> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
        .unwrap()
        .with_batch_size(65536)
        .with_limit(max_rows)
        .build()
        .unwrap();
    columns(reader.map(|b| b.unwrap()).collect())
}

fn main() {
    let dir = std::env::args().nth(1).expect("data dir");
    let session = session();
    let compressor: BtrBlocksCompressor = BtrBlocksCompressorBuilder::from_session(&session)
        .exclude_schemes([OnPairScheme::default().id()])
        .build();

    let mut datasets: Vec<(&str, Vec<(String, ArrayRef)>)> = vec![
        ("airquality", parquet(&format!("{dir}/airquality.parquet"), usize::MAX)),
        ("rplace", parquet(&format!("{dir}/rplace.parquet"), 8_000_000)),
    ];
    datasets.push((
        "lineitem",
        columns(
            tpchgen_arrow::LineItemArrow::new(LineItemGenerator::new(0.5, 1, 1))
                .with_batch_size(65536)
                .collect(),
        ),
    ));
    datasets.push((
        "orders",
        columns(
            tpchgen_arrow::OrderArrow::new(OrderGenerator::new(0.5, 1, 1))
                .with_batch_size(65536)
                .collect(),
        ),
    ));

    let (mut all_in, mut all_out, mut all_time) = (0usize, 0usize, Duration::ZERO);
    for (dataset, cols) in datasets {
        for (name, col) in cols {
            let mut ctx = session.create_execution_ctx();
            let bytes_per_row = (col.nbytes() / col.len().max(1)).max(1);
            let block = (ONE_MEG / bytes_per_row).next_multiple_of(ROW_BLOCK);
            let (mut nin, mut nout, mut time) = (0usize, 0usize, Duration::ZERO);
            let mut start = 0;
            while start < col.len() {
                let end = (start + block).min(col.len());
                let slice = col.slice(start..end).unwrap();
                let canonical = slice.to_canonical().unwrap().into_array();
                let t = Instant::now();
                let out = compressor.compress(&canonical, &mut ctx).unwrap();
                time += t.elapsed();
                nin += canonical.nbytes() as usize;
                nout += out.nbytes() as usize;
                start = end;
            }
            println!(
                "{dataset}\t{name}\t{}\t{block}\t{nin}\t{nout}\t{:.3}",
                col.dtype(),
                time.as_secs_f64() * 1e3
            );
            all_in += nin;
            all_out += nout;
            all_time += time;
        }
    }
    println!(
        "TOTAL\t-\t-\t-\t{all_in}\t{all_out}\t{:.3}",
        all_time.as_secs_f64() * 1e3
    );
}
