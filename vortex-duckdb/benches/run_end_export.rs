// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run-end exporter setup and batch export, including ranges across shared dictionaries.

#![expect(clippy::unwrap_used)]

use divan::AllocProfiler;
use divan::Bencher;
use mimalloc::MiMalloc;
use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::buffer::Buffer;
use vortex::encodings::runend::RunEnd;
use vortex::session::VortexSession;
use vortex_duckdb::_benchmarking::ArrayExporter;
use vortex_duckdb::_benchmarking::ConversionCache;
use vortex_duckdb::duckdb::DataChunk;
use vortex_duckdb::duckdb::LogicalType;

#[global_allocator]
static GLOBAL: AllocProfiler<MiMalloc> = AllocProfiler::new(MiMalloc);

fn main() {
    divan::main();
}

fn input(session: &VortexSession, run_len: usize, chunked: bool) -> StructArray {
    let mut ctx = session.create_execution_ctx();
    let runs = if run_len == 1 { 16383 } else { 8 };
    let values =
        PrimitiveArray::from_iter((0..runs).map(|i| i64::try_from(i).unwrap())).into_array();
    let ends: Buffer<u64> = (1..=runs)
        .map(|i| u64::try_from(i * run_len).unwrap())
        .collect();
    let array = RunEnd::try_new(ends.into_array(), values, &mut ctx)
        .unwrap()
        .into_array();
    let array = if chunked {
        // Reusing the exact array preserves the values identity. The sliced request puts chunk
        // boundaries inside output batches even when the source length is a multiple of 2048.
        ChunkedArray::try_new(vec![array.clone(); 4], array.dtype().clone())
            .unwrap()
            .into_array()
    } else {
        array
    };
    let sliced = array.slice(17..array.len() - 13).unwrap();
    let array = if chunked {
        sliced.execute_until::<Chunked>(&mut ctx).unwrap()
    } else {
        sliced.execute_until::<RunEnd>(&mut ctx).unwrap()
    };

    StructArray::from_fields(&[("value", array)]).unwrap()
}

#[divan::bench(args = [1, 8192], consts = [false, true])]
fn construct<const CHUNKED: bool>(bencher: Bencher, run_len: usize) {
    let session = VortexSession::default();
    let array = input(&session, run_len, CHUNKED);
    bencher
        .with_inputs(|| (ConversionCache::default(), session.create_execution_ctx()))
        .bench_values(|(cache, ctx)| ArrayExporter::try_new(&array, &cache, ctx).unwrap());
}

#[divan::bench(args = [1, 8192], consts = [false, true])]
fn batches<const CHUNKED: bool>(bencher: Bencher, run_len: usize) {
    let session = VortexSession::default();
    let array = input(&session, run_len, CHUNKED);
    let cache = ConversionCache::default();
    bencher
        .with_inputs(|| {
            (
                ArrayExporter::try_new(&array, &cache, session.create_execution_ctx()).unwrap(),
                DataChunk::new([LogicalType::int64()]),
            )
        })
        .bench_values(|(mut exporter, mut chunk)| {
            while exporter.export(&mut chunk, None).unwrap() {}
            (exporter, chunk)
        });
}
