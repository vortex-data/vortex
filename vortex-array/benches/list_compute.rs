// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::builders::{ArrayBuilder, ListBuilder};
use vortex_array::compute::is_sorted;
use vortex_array::compute::min_max;
use vortex_dtype::{DType, Nullability, PType};

fn main() {
    divan::main();
}

const ARRAY_SIZE: usize = 1_000;
const LIST_SIZE: usize = 10;

fn create_sorted_list_array() -> ArrayRef {
    let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
    let nullability = Nullability::NonNullable;
    let mut builder = ListBuilder::<u32>::new(element_dtype.clone(), nullability);

    for i in 0..ARRAY_SIZE {
        let list_elements: Vec<i32> = (0..LIST_SIZE).map(|j| (i * LIST_SIZE + j) as i32).collect();
        let list_scalar = list_elements
            .into_iter()
            .map(|x| vortex_scalar::Scalar::primitive(x, Nullability::NonNullable))
            .collect::<Vec<_>>();
        let list = vortex_scalar::Scalar::list(
            element_dtype.clone(),
            list_scalar,
            Nullability::NonNullable,
        );
        builder.append_value(list.as_list()).unwrap();
    }

    builder.finish()
}

fn create_almost_sorted_list_array() -> ArrayRef {
    // Create an array where the last two elements are swapped
    // For simplicity, we'll create a new array with the swap
    let element_dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
    let nullability = Nullability::NonNullable;
    let mut builder = ListBuilder::<u32>::new(element_dtype.clone(), nullability);

    for i in 0..ARRAY_SIZE {
        let list_elements: Vec<i32> = if i == ARRAY_SIZE - 2 {
            // Second to last: use last elements
            ((ARRAY_SIZE - 1) * LIST_SIZE..ARRAY_SIZE * LIST_SIZE)
                .map(|x| x as i32)
                .collect()
        } else if i == ARRAY_SIZE - 1 {
            // Last: use second to last elements
            ((ARRAY_SIZE - 2) * LIST_SIZE..(ARRAY_SIZE - 1) * LIST_SIZE)
                .map(|x| x as i32)
                .collect()
        } else {
            (i * LIST_SIZE..(i + 1) * LIST_SIZE).map(|x| x as i32).collect()
        };

        let list_scalar = list_elements
            .into_iter()
            .map(|x| vortex_scalar::Scalar::primitive(x, Nullability::NonNullable))
            .collect::<Vec<_>>();
        let list = vortex_scalar::Scalar::list(
            element_dtype.clone(),
            list_scalar,
            Nullability::NonNullable,
        );
        builder.append_value(list.as_list()).unwrap();
    }

    builder.finish()
}

#[divan::bench]
fn is_sorted_list_sorted(bencher: Bencher) {
    let arr = create_sorted_list_array();

    bencher
        .with_inputs(|| &arr)
        .bench_refs(|arr| is_sorted(*arr).unwrap());
}

#[divan::bench]
fn is_sorted_list_almost_sorted(bencher: Bencher) {
    let arr = create_almost_sorted_list_array();

    bencher
        .with_inputs(|| &arr)
        .bench_refs(|arr| is_sorted(*arr).unwrap());
}

#[divan::bench]
fn min_max_list_sorted(bencher: Bencher) {
    let arr = create_sorted_list_array();

    bencher
        .with_inputs(|| &arr)
        .bench_refs(|arr| min_max(*arr).unwrap());
}

#[divan::bench]
fn min_max_list_almost_sorted(bencher: Bencher) {
    let arr = create_almost_sorted_list_array();

    bencher
        .with_inputs(|| &arr)
        .bench_refs(|arr| min_max(*arr).unwrap());
}