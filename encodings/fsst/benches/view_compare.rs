// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(clippy::unwrap_used)]

use divan::Bencher;
use vortex_array::Array;
use vortex_array::arrays::{ConstantArray, VarBinViewArray};
use vortex_array::compute::{Operator, compare};
use vortex_dtype::Nullability;
use vortex_fsst::{FSSTEncoding, FSSTViewEncoding};
use vortex_scalar::Scalar;

#[divan::bench(args = [Operator::Eq, Operator::NotEq])]
fn fsst_view(bencher: Bencher, operator: Operator) {
    let array = VarBinViewArray::from_iter_str([
        "short",
        "short",
        "looooooong",
        "short",
        "looooooong",
        "super duper looong",
    ])
    .to_canonical();

    let array = FSSTViewEncoding.encode(&array, None).unwrap().unwrap();
    let target = ConstantArray::new(
        Scalar::utf8("looooooong", Nullability::NonNullable),
        array.len(),
    );

    bencher.bench_local(|| compare(array.as_ref(), target.as_ref(), operator).unwrap())
}

#[divan::bench(args = [Operator::Eq, Operator::NotEq])]
fn fsst(bencher: Bencher, operator: Operator) {
    let array = VarBinViewArray::from_iter_str([
        "short",
        "short",
        "looooooong",
        "short",
        "looooooong",
        "super duper looong",
    ])
    .to_canonical();

    let array = FSSTEncoding.encode(&array, None).unwrap().unwrap();
    let target = ConstantArray::new(
        Scalar::utf8("looooooong", Nullability::NonNullable),
        array.len(),
    );

    bencher.bench_local(|| compare(array.as_ref(), target.as_ref(), operator).unwrap())
}

#[divan::bench(args = [Operator::Eq, Operator::NotEq])]
fn raw(bencher: Bencher, operator: Operator) {
    let array = VarBinViewArray::from_iter_str([
        "short",
        "short",
        "looooooong",
        "short",
        "looooooong",
        "super duper looong",
    ]);

    let target = ConstantArray::new(
        Scalar::utf8("looooooong", Nullability::NonNullable),
        array.len(),
    );

    bencher.bench_local(|| compare(array.as_ref(), target.as_ref(), operator).unwrap())
}

fn main() {
    divan::main()
}
