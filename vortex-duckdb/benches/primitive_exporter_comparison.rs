// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors


use divan::Bencher;
use itertools::Itertools;
use vortex::arrays::PrimitiveArray;
use vortex_duckdb::cpp;
use vortex_duckdb::duckdb::{DUCKDB_STANDARD_VECTOR_SIZE, LogicalType, Vector};
use vortex_duckdb::exporter::primitive::{new_copy_exporter, new_exporter};

fn main() {
    divan::main();
}

#[divan::bench(args = [1, 10, 100])]
fn offset_export_copy(bencher: Bencher, size: usize) {
    let elem_size = size * DUCKDB_STANDARD_VECTOR_SIZE;

    bencher
        .with_inputs(|| {
            (
                PrimitiveArray::from_iter(0..=elem_size as u64),
                (0..size)
                    .map(|_| {
                        Vector::with_capacity(
                            LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER),
                            DUCKDB_STANDARD_VECTOR_SIZE,
                        )
                    })
                    .collect_vec(),
            )
        })
        .bench_local_values(|(array, mut vector)| {
            let exporter = new_copy_exporter(&array).unwrap();
            for i in 0..size {
                exporter
                    .export(
                        i * DUCKDB_STANDARD_VECTOR_SIZE,
                        DUCKDB_STANDARD_VECTOR_SIZE,
                        &mut vector[i],
                    )
                    .unwrap();
            }
            vector
        })
}

// #[divan::bench(args = [1, 10, 100])]
fn memory_pattern_zero_copy(bencher: Bencher, size: usize) {
    const EXPORT_SIZE: usize = 2048;
    let array = PrimitiveArray::from_iter(0..(size * EXPORT_SIZE) as u64);

    bencher
        .with_inputs(|| {
            (
                &array,
                (0..size)
                    .map(|_| {
                        Vector::with_capacity(
                            LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER),
                            EXPORT_SIZE,
                        )
                    })
                    .collect_vec(),
            )
        })
        .bench_local_values(|(array, mut vector)| {
            let exporter = new_exporter(array).unwrap();
            for i in 0..size {
                exporter
                    .export(i * EXPORT_SIZE, EXPORT_SIZE, &mut vector[i])
                    .unwrap();
            }
            vector
        })
}
