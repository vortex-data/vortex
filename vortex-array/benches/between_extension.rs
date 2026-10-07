// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension BETWEEN delegation against the two-comparison fallback.

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

fn predicate<const NULLABLE: bool>(len: usize, fused: bool) -> ArrayRef {
    let storage = PrimitiveArray::new(
        (0..len)
            .map(|index| i32::try_from(index % 4096).unwrap())
            .collect::<Buffer<_>>(),
        if NULLABLE {
            Validity::from_iter((0..len).map(|index| index % 11 != 0))
        } else {
            Validity::NonNullable
        },
    )
    .into_array();
    let dtype = Date::new(TimeUnit::Days, storage.dtype().nullability()).erased();
    let array = ExtensionArray::new(dtype.clone(), storage).into_array();
    let bound = |value| {
        ConstantArray::new(
            Scalar::extension_ref(
                dtype.with_nullability(Nullability::NonNullable),
                Scalar::from(value),
            ),
            len,
        )
        .into_array()
    };
    let lower = bound(1024i32);
    let upper = bound(2048i32);
    if fused {
        array
            .between(
                lower,
                upper,
                BetweenOptions {
                    lower_strict: StrictComparison::NonStrict,
                    upper_strict: StrictComparison::Strict,
                },
            )
            .unwrap()
    } else {
        lower
            .binary(array.clone(), Operator::Lte)
            .unwrap()
            .binary(array.binary(upper, Operator::Lt).unwrap(), Operator::And)
            .unwrap()
    }
}

fn bench_predicate(bencher: Bencher, array: ArrayRef) {
    let session = array_session();
    bencher
        .counter(ItemsCount::new(array.len()))
        .with_inputs(|| session.create_execution_ctx())
        .bench_refs(|ctx| array.clone().execute::<BoolArray>(ctx).unwrap());
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = [8192, 262144], consts = [false, true])]
fn between<const NULLABLE: bool>(bencher: Bencher, len: usize) {
    bench_predicate(bencher, predicate::<NULLABLE>(len, true));
}

#[vortex_bench_support::cpu_features]
#[divan::bench(args = [8192, 262144], consts = [false, true])]
fn two_compares<const NULLABLE: bool>(bencher: Bencher, len: usize) {
    bench_predicate(bencher, predicate::<NULLABLE>(len, false));
}
