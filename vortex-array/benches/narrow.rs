// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selection, materialization, and encoding costs for logical i64 arrays with narrow storage.

#![expect(
    clippy::unwrap_used,
    reason = "benchmark fixtures satisfy encoding invariants"
)]

use std::hint::black_box;
use std::time::Instant;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::NarrowArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_mask::Mask;

const WIDTHS: &[PType] = &[PType::I64, PType::I32, PType::I16, PType::I8];
const OPERATIONS: &[Operation] = &[
    Operation::FilterToI64,
    Operation::TakeToI64,
    Operation::CompareFilterToI64,
    Operation::FilterCompare,
    Operation::TakeCompare,
    Operation::MaterializeI64,
    Operation::Encode,
];

#[derive(Clone, Copy, Debug)]
enum Operation {
    FilterToI64,
    TakeToI64,
    CompareFilterToI64,
    FilterCompare,
    TakeCompare,
    MaterializeI64,
    Encode,
}

struct Input {
    values: ArrayRef,
    mask: Mask,
    indices: ArrayRef,
    constant: ArrayRef,
    encode_values: Buffer<i64>,
}

fn value(i: usize) -> i64 {
    (i.wrapping_mul(2_654_435_761).rotate_right(13) & 31) as i64
}

fn input(len: usize, storage: PType, ctx: &mut ExecutionCtx) -> Input {
    let wide = PrimitiveArray::from_iter((0..len).map(value)).into_array();
    let values = if storage == PType::I64 {
        wide
    } else {
        let child = wide
            .cast(storage.into())
            .unwrap()
            .execute::<PrimitiveArray>(ctx)
            .unwrap();
        NarrowArray::try_new(child.into_array(), PType::I64.into())
            .unwrap()
            .into_array()
    };
    // Force encode to choose the requested width rather than always choosing i8.
    let offset = match storage {
        PType::I8 => 0,
        PType::I16 => 128,
        PType::I32 => 32_768,
        PType::I64 => 2_147_483_648,
        _ => unreachable!(),
    };
    Input {
        values,
        mask: Mask::from_iter((0..len).map(|i| value(i) < 16)),
        indices: PrimitiveArray::from_iter(
            (0..len / 8).map(|i| {
                u32::try_from(i.wrapping_mul(2_654_435_761).rotate_right(13) % len).unwrap()
            }),
        )
        .into_array(),
        constant: ConstantArray::new(16i64, len).into_array(),
        encode_values: Buffer::from_iter((0..len).map(|i| offset + value(i))),
    }
}

fn run(
    op: Operation,
    input: &Input,
    encode_source: PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> ArrayRef {
    match op {
        Operation::Encode => NarrowArray::encode(encode_source, ctx).unwrap(),
        Operation::MaterializeI64 => input
            .values
            .clone()
            .execute::<PrimitiveArray>(ctx)
            .unwrap()
            .into_array(),
        Operation::FilterToI64 => input
            .values
            .filter(input.mask.clone())
            .unwrap()
            .execute::<PrimitiveArray>(ctx)
            .unwrap()
            .into_array(),
        Operation::TakeToI64 => input
            .values
            .take(input.indices.clone())
            .unwrap()
            .execute::<PrimitiveArray>(ctx)
            .unwrap()
            .into_array(),
        Operation::CompareFilterToI64 => {
            let selected = input
                .values
                .binary(input.constant.clone(), Operator::Lt)
                .unwrap()
                .execute::<BoolArray>(ctx)
                .unwrap()
                .to_mask_fill_null_false(ctx);
            input
                .values
                .filter(selected)
                .unwrap()
                .execute::<PrimitiveArray>(ctx)
                .unwrap()
                .into_array()
        }
        Operation::FilterCompare | Operation::TakeCompare => {
            let selected = if matches!(op, Operation::FilterCompare) {
                input.values.filter(input.mask.clone()).unwrap()
            } else {
                input.values.take(input.indices.clone()).unwrap()
            };
            let constant = ConstantArray::new(16i64, selected.len()).into_array();
            selected
                .binary(constant, Operator::Lt)
                .unwrap()
                .execute::<BoolArray>(ctx)
                .unwrap()
                .into_array()
        }
    }
}

#[divan::bench(args = [8_192, 1_048_576], types = [i64, i32, i16, i8])]
fn compare_filter<T: NativePType>(bencher: Bencher, len: usize) {
    bench(bencher, len, T::PTYPE, Operation::CompareFilterToI64);
}

#[divan::bench(args = [8_192, 1_048_576], types = [i64, i32, i16, i8])]
fn filter<T: NativePType>(bencher: Bencher, len: usize) {
    bench(bencher, len, T::PTYPE, Operation::FilterToI64);
}

#[divan::bench(args = [8_192, 1_048_576], types = [i64, i32, i16, i8])]
fn take<T: NativePType>(bencher: Bencher, len: usize) {
    bench(bencher, len, T::PTYPE, Operation::TakeToI64);
}

#[divan::bench(args = [8_192, 1_048_576], types = [i64, i32, i16, i8])]
fn filter_compare<T: NativePType>(bencher: Bencher, len: usize) {
    bench(bencher, len, T::PTYPE, Operation::FilterCompare);
}

#[divan::bench(args = [8_192, 1_048_576], types = [i64, i32, i16, i8])]
fn take_compare<T: NativePType>(bencher: Bencher, len: usize) {
    bench(bencher, len, T::PTYPE, Operation::TakeCompare);
}

#[divan::bench(args = [8_192, 1_048_576], types = [i64, i32, i16, i8])]
fn materialize<T: NativePType>(bencher: Bencher, len: usize) {
    bench(bencher, len, T::PTYPE, Operation::MaterializeI64);
}

#[divan::bench(args = [8_192, 1_048_576], types = [i64, i32, i16, i8])]
fn encode<T: NativePType>(bencher: Bencher, len: usize) {
    bench(bencher, len, T::PTYPE, Operation::Encode);
}

fn bench(bencher: Bencher, len: usize, storage: PType, op: Operation) {
    let session = array_session();
    let input = input(len, storage, &mut session.create_execution_ctx());
    bencher
        .counter(ItemsCount::new(len))
        .with_inputs(|| {
            // Fresh array statistics include the bounds scan in every encoding sample.
            (
                PrimitiveArray::new(input.encode_values.clone(), Validity::NonNullable),
                session.create_execution_ctx(),
            )
        })
        .bench_values(|(source, mut ctx)| run(op, &input, source, &mut ctx));
}

#[expect(clippy::use_debug, reason = "enum variants name the benchmark cases")]
fn interleaved() {
    const SAMPLES: usize = 140;
    let session = array_session();
    println!("rows,operation,representation,sample,nanoseconds");
    for len in [8_192, 1_048_576, 16_777_216] {
        let inputs = WIDTHS
            .iter()
            .map(|&width| input(len, width, &mut session.create_execution_ctx()))
            .collect::<Vec<_>>();
        let names = ["primitive_i64", "narrow_i32", "narrow_i16", "narrow_i8"];
        for (name, input) in names.iter().zip(&inputs) {
            eprintln!("size,{len},{name},{}", input.values.nbytes());
        }
        for &op in OPERATIONS {
            let mut expected: Option<ArrayRef> = None;
            for (input, &width) in inputs.iter().zip(WIDTHS) {
                let source =
                    PrimitiveArray::new(input.encode_values.clone(), Validity::NonNullable);
                let result = run(
                    op,
                    input,
                    source.clone(),
                    &mut session.create_execution_ctx(),
                );
                if matches!(op, Operation::Encode) {
                    assert_eq!(result.nbytes(), (len * width.byte_width()) as u64);
                    assert_arrays_eq!(result, source, &mut session.create_execution_ctx());
                } else if let Some(expected) = &expected {
                    assert_arrays_eq!(result, expected, &mut session.create_execution_ctx());
                } else {
                    expected = Some(result);
                }
            }
            let mut samples = vec![Vec::with_capacity(SAMPLES); inputs.len()];
            for sample in 0..SAMPLES {
                let reverse = !(sample / inputs.len()).is_multiple_of(2);
                for step in 0..inputs.len() {
                    let offset = if reverse { inputs.len() - step } else { step };
                    let index = (sample + offset) % inputs.len();
                    let input = &inputs[index];
                    let source =
                        PrimitiveArray::new(input.encode_values.clone(), Validity::NonNullable);
                    let mut ctx = session.create_execution_ctx();
                    let start = Instant::now();
                    let result = run(op, input, source, &mut ctx);
                    let elapsed = start.elapsed();
                    black_box(result);
                    samples[index].push(elapsed);
                }
            }
            for (name, samples) in names.iter().zip(samples) {
                for (sample, duration) in samples.into_iter().enumerate() {
                    println!("{len},{op:?},{name},{sample},{}", duration.as_nanos());
                }
            }
        }
    }
}

fn main() {
    if std::env::var_os("VORTEX_NARROW_INTERLEAVED").is_some() {
        interleaved();
    } else {
        divan::main();
    }
}
