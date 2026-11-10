// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare ascending and descending sortedness scans.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex_array::aggregate_fn::fns::is_sorted::SortDirection;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted_with_options;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);
const LEN: usize = 16_384;
const CASES: &[(SortDirection, bool)] = &[
    (SortDirection::Asc, false),
    (SortDirection::Desc, false),
    (SortDirection::Asc, true),
    (SortDirection::Desc, true),
];

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn values(direction: SortDirection) -> impl Iterator<Item = i64> {
    (0..LEN).map(move |i| match direction {
        SortDirection::Asc => i as i64,
        SortDirection::Desc => (LEN - i) as i64,
    })
}

fn scan(bencher: Bencher, array: ArrayRef, options: IsSortedOptions) {
    // Direct accumulation bypasses sortedness statistics, so each iteration scans the values.
    bencher
        .counter(ItemsCount::new(array.len()))
        .with_inputs(|| {
            (
                Accumulator::try_new(IsSorted, options, array.dtype().clone()).unwrap(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(acc, ctx)| {
            acc.accumulate(&array, ctx).unwrap();
            acc.finish().unwrap()
        });
}

#[divan::bench(args = CASES)]
fn primitive_scan(bencher: Bencher, &(direction, strict): &(SortDirection, bool)) {
    let array = PrimitiveArray::from_iter(values(direction)).into_array();
    scan(bencher, array, IsSortedOptions { strict, direction });
}

#[divan::bench(args = CASES)]
fn nullable_primitive_scan(bencher: Bencher, &(direction, strict): &(SortDirection, bool)) {
    let array = PrimitiveArray::from_option_iter(
        std::iter::once(None).chain(values(direction).skip(1).map(Some)),
    )
    .into_array();
    scan(bencher, array, IsSortedOptions { strict, direction });
}

#[divan::bench(args = [SortDirection::Asc, SortDirection::Desc])]
fn nullable_bool_scan(bencher: Bencher, direction: SortDirection) {
    let values = (1..LEN).map(|i| {
        Some(match direction {
            SortDirection::Asc => i >= LEN / 2,
            SortDirection::Desc => i < LEN / 2,
        })
    });
    let array = BoolArray::from_iter(std::iter::once(None).chain(values)).into_array();
    scan(
        bencher,
        array,
        IsSortedOptions {
            strict: false,
            direction,
        },
    );
}

#[divan::bench(args = CASES)]
fn utf8_scan(bencher: Bencher, &(direction, strict): &(SortDirection, bool)) {
    let array =
        VarBinViewArray::from_iter_str(values(direction).map(|v| format!("{v:016}"))).into_array();
    scan(bencher, array, IsSortedOptions { strict, direction });
}

#[divan::bench(args = CASES)]
fn dictionary(bencher: Bencher, &(direction, strict): &(SortDirection, bool)) {
    let dictionary: Buffer<i64> = values(SortDirection::Asc).collect();
    let len = u32::try_from(LEN).vortex_expect("valid len");
    let codes: Buffer<u32> = (0..len)
        .map(|i| match direction {
            SortDirection::Asc => i,
            SortDirection::Desc => len - 1 - i,
        })
        .collect();
    bencher
        .counter(ItemsCount::new(LEN))
        .with_inputs(|| {
            // Fresh children prevent their sortedness caches from replacing the scans.
            let codes = PrimitiveArray::new(codes.clone(), Validity::NonNullable).into_array();
            let values =
                PrimitiveArray::new(dictionary.clone(), Validity::NonNullable).into_array();
            (
                DictArray::try_new(codes, values).unwrap().into_array(),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(array, ctx)| {
            is_sorted_with_options(array, IsSortedOptions { strict, direction }, ctx).unwrap()
        });
}
