// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::slice::SliceReduce;

fn main() {
    divan::main();
}

const ARRAY_LENGTHS: &[usize] = &[10_000];

fn build_dict(len: usize) -> DictArray {
    let num_unique = 256;
    let codes = PrimitiveArray::from_iter((0..len).map(|i| (i % num_unique) as u32));
    let values = PrimitiveArray::from_iter(0..num_unique as u32);
    DictArray::try_new(codes.into_array(), values.into_array()).unwrap()
}

fn build_primitive(len: usize) -> PrimitiveArray {
    PrimitiveArray::from_iter(0..len as u32)
}

#[divan::bench(args = ARRAY_LENGTHS)]
fn slice_primitive_tight_loop(bencher: Bencher, len: usize) {
    let arr = build_primitive(len).into_array();
    let slice_len = 64;

    let num_slices = len / slice_len;

    bencher
        .with_inputs(|| (&arr, Vec::<ArrayRef>::with_capacity(num_slices)))
        .bench_refs(|(arr, out)| {
            out.clear();
            let mut offset = 0;
            while offset + slice_len <= len {
                out.push(arr.slice(offset..offset + slice_len).unwrap());
                offset += slice_len;
            }
        });
}

#[divan::bench(args = ARRAY_LENGTHS)]
fn slice_primitive_reduce_tight_loop(bencher: Bencher, len: usize) {
    let arr = build_primitive(len);
    let slice_len = 64;

    let num_slices = len / slice_len;

    bencher
        .with_inputs(|| (&arr, Vec::<ArrayRef>::with_capacity(num_slices)))
        .bench_refs(|(arr, out)| {
            out.clear();
            let mut offset = 0;
            while offset + slice_len <= len {
                out.push(
                    <Primitive as SliceReduce>::slice(arr.as_view(), offset..offset + slice_len)
                        .unwrap()
                        .unwrap(),
                );
                offset += slice_len;
            }
        });
}

#[divan::bench(args = ARRAY_LENGTHS)]
fn slice_dict_tight_loop(bencher: Bencher, len: usize) {
    let dict = build_dict(len).into_array();
    let slice_len = 64;
    let num_slices = len / slice_len;

    bencher
        .with_inputs(|| (&dict, Vec::<ArrayRef>::with_capacity(num_slices)))
        .bench_refs(|(dict, out)| {
            out.clear();
            let mut offset = 0;
            while offset + slice_len <= len {
                out.push(dict.slice(offset..offset + slice_len).unwrap());
                offset += slice_len;
            }
        });
}
