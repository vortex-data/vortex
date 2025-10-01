// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use vortex::arrays::PrimitiveArray;
use vortex_duckdb::cpp;
use vortex_duckdb::duckdb::{DataChunk, LogicalType};
use vortex_duckdb::exporter::primitive::{new_copy_exporter, new_exporter};

fn main() {
    divan::main();
}

// #[divan::bench(args = [100, 1_000, 10_000, 100_000])]
// fn primitive_exporter_zero_copy(bencher: Bencher, size: i32) {
//     let array = PrimitiveArray::from_iter(0..size);
//     let exporter = new_exporter(&array).unwrap();
//
//     bencher
//         .with_inputs(|| DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]))
//         .bench_values(|mut chunk| {
//             let mut vector = chunk.get_vector(0);
//             exporter.export(0, size as usize, &mut vector).unwrap();
//             chunk.set_len(size as usize);
//             chunk
//         })
// }
//
// #[divan::bench(args = [100, 1_000, 10_000, 100_000])]
// fn primitive_exporter_copy(bencher: Bencher, size: i32) {
//     let array = PrimitiveArray::from_iter(0..size);
//     let exporter = new_copy_exporter(&array).unwrap();
//
//     bencher
//         .with_inputs(|| DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]))
//         .bench_values(|mut chunk| {
//             let mut vector = chunk.get_vector(0);
//             exporter.export(0, size as usize, &mut vector).unwrap();
//             chunk.set_len(size as usize);
//             chunk
//         })
// }
//
// #[divan::bench(args = [100, 1_000, 10_000])]
// fn partial_export_zero_copy(bencher: Bencher, export_size: usize) {
//     const ARRAY_SIZE: i32 = 100_000;
//     let array = PrimitiveArray::from_iter(0..ARRAY_SIZE);
//     let exporter = new_exporter(&array).unwrap();
//
//     bencher
//         .with_inputs(|| DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]))
//         .bench_values(|mut chunk| {
//             let mut vector = chunk.get_vector(0);
//             exporter.export(0, export_size, &mut vector).unwrap();
//             chunk.set_len(export_size);
//             chunk
//         })
// }
//
// #[divan::bench(args = [100, 1_000, 10_000])]
// fn partial_export_copy(bencher: Bencher, export_size: usize) {
//     const ARRAY_SIZE: i32 = 100_000;
//     let array = PrimitiveArray::from_iter(0..ARRAY_SIZE);
//     let exporter = new_copy_exporter(&array).unwrap();
//
//     bencher
//         .with_inputs(|| DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]))
//         .bench_values(|mut chunk| {
//             let mut vector = chunk.get_vector(0);
//             exporter.export(0, export_size, &mut vector).unwrap();
//             chunk.set_len(export_size);
//             chunk
//         })
// }
//
// #[divan::bench(args = [0, 1_000, 10_000, 50_000])]
// fn offset_export_zero_copy(bencher: Bencher, offset: usize) {
//     const ARRAY_SIZE: i32 = 100_000;
//     const EXPORT_SIZE: usize = 1_000;
//     let array = PrimitiveArray::from_iter(0..ARRAY_SIZE);
//
//
//     bencher
//         .with_inputs(|| (&array, DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)])))
//         .bench_values(|(array, mut chunk)| {
//             let exporter = new_exporter(&array).unwrap();
//             let mut vector = chunk.get_vector(0);
//             exporter.export(offset, EXPORT_SIZE, &mut vector).unwrap();
//             chunk.set_len(EXPORT_SIZE);
//             chunk
//         })
// }

#[divan::bench(args = [2048, 20_480, 204_800])]
fn offset_export_copy(bencher: Bencher, size: usize) {
    const EXPORT_SIZE: usize = 2048;
    let array = PrimitiveArray::from_iter(0..size as u64);

    bencher
        .with_inputs(|| (&array, DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)])))
        .bench_values(|(array, mut chunk)| {
            let exporter = new_copy_exporter(array).unwrap();
            let mut vector = chunk.get_vector(0);
            for i in (0..size).step_by(EXPORT_SIZE) {
                exporter.export(i, EXPORT_SIZE, &mut vector).unwrap();
            }
            chunk
        })
}
//
// Benchmark specifically for memory allocation patterns
// #[divan::bench(args = [2048, 20_480, 204_800])]
// fn memory_pattern_zero_copy(bencher: Bencher, size: usize) {
//     const EXPORT_SIZE: usize = 2048;
//     let array = PrimitiveArray::from_iter(0..size as u64);
//
//
//     bencher
//         .with_inputs(|| (&array, DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)])))
//         .bench_values(|(array, mut chunk)| {
//             let exporter = new_exporter(array).unwrap();
//             let mut vector = chunk.get_vector(0);
//             for i in (0..size).step_by(EXPORT_SIZE) {
//                 exporter.export(i, EXPORT_SIZE, &mut vector).unwrap();
//             }
//             chunk
//         })
// }
//
// #[divan::bench(args = [1_000, 10_000, 100_000])]
// fn memory_pattern_copy(bencher: Bencher, size: i32) {
//     bencher
//         .with_inputs(|| PrimitiveArray::from_iter(0..size))
//         .bench_values(|array| {
//             let exporter = new_copy_exporter(&array).unwrap();
//             let mut chunk =
//                 DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);
//             let mut vector = chunk.get_vector(0);
//             exporter.export(0, size as usize, &mut vector).unwrap();
//             chunk.set_len(size as usize);
//             chunk
//         })
// }
