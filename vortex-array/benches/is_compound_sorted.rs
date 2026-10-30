// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sort order benchmark.
//!
//! Covers order and line-item keys with dense and sparse ties, batch boundaries,
//! nullable and UTF-8 fields, and constant and dictionary encodings. Also compares
//! single-field `IsCompoundSorted` with `IsSorted` on the same ascending order IDs.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use mimalloc::MiMalloc;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::Accumulator;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynAccumulator;
use vortex_array::aggregate_fn::fns::is_compound_sorted::IsCompoundSorted;
use vortex_array::aggregate_fn::fns::is_compound_sorted::IsCompoundSortedOptions;
use vortex_array::aggregate_fn::fns::is_compound_sorted::SortDirection;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);
const ROWS: u32 = 65_536;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

fn order_line_options() -> IsCompoundSortedOptions {
    IsCompoundSortedOptions::new([
        ("order_id", SortDirection::Asc),
        ("line_item_id", SortDirection::Asc),
    ])
}

fn measure<V: AggregateFnVTable>(
    bencher: Bencher,
    vtable: V,
    options: V::Options,
    make_array: impl Fn() -> ArrayRef + Sync,
) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| {
            let array = make_array();
            let accumulator =
                Accumulator::try_new(vtable.clone(), options.clone(), array.dtype().clone())
                    .unwrap();
            (array, accumulator, SESSION.create_execution_ctx())
        })
        .bench_values(|(array, mut accumulator, mut ctx)| {
            accumulator.accumulate(&array, &mut ctx).unwrap();
            let result = accumulator.finish().unwrap();
            assert_eq!(result.as_bool().value(), Some(true));
            divan::black_box(result)
        });
}

// Slice during setup so the measurement covers accumulation and boundary comparisons.
fn measure_batches(bencher: Bencher, batch_rows: usize, make_array: impl Fn() -> ArrayRef + Sync) {
    bencher
        .counter(ItemsCount::new(ROWS))
        .with_inputs(|| {
            let array = make_array();
            let accumulator =
                Accumulator::try_new(IsCompoundSorted, order_line_options(), array.dtype().clone())
                    .unwrap();
            let batches = (0..array.len())
                .step_by(batch_rows)
                .map(|start| {
                    let end = (start + batch_rows).min(array.len());
                    array.slice(start..end).unwrap()
                })
                .collect::<Vec<_>>();
            (batches, accumulator, SESSION.create_execution_ctx())
        })
        .bench_values(|(batches, mut accumulator, mut ctx)| {
            for batch in &batches {
                accumulator.accumulate(batch, &mut ctx).unwrap();
            }
            let result = accumulator.finish().unwrap();
            assert_eq!(result.as_bool().value(), Some(true));
            divan::black_box(result)
        });
}

#[divan::bench(args = [1, 2, 16, 65_536])]
fn order_lines_by_order_size(bencher: Bencher, items_per_order: u32) {
    let order_ids = (0..ROWS)
        .map(|index| u64::from(index / items_per_order))
        .collect::<Buffer<_>>();
    let line_item_ids = (0..ROWS)
        .map(|index| u64::from(index % items_per_order))
        .collect::<Buffer<_>>();
    measure(bencher, IsCompoundSorted, order_line_options(), || {
        StructArray::from_fields(&[
            ("order_id", order_ids.clone().into_array()),
            ("line_item_id", line_item_ids.clone().into_array()),
        ])
        .unwrap()
        .into_array()
    });
}

#[divan::bench]
fn repeated_order_line_key(bencher: Bencher) {
    measure(bencher, IsCompoundSorted, order_line_options(), || {
        StructArray::from_fields(&[
            (
                "order_id",
                ConstantArray::new(1u64, ROWS as usize).into_array(),
            ),
            (
                "line_item_id",
                ConstantArray::new(2u64, ROWS as usize).into_array(),
            ),
        ])
        .unwrap()
        .into_array()
    });
}

// Customer IDs repeat every 1024 rows. Order IDs preserve ties every 2048 rows.
// This leaves sparse tied pairs for the line-item IDs to compare.
#[divan::bench]
fn customer_order_lines_sparse_ties(bencher: Bencher) {
    let customer_ids = (0..ROWS)
        .map(|index| u64::from(index - index / 1024))
        .collect::<Buffer<_>>();
    let order_ids = (0..ROWS)
        .map(|index| u64::from(index - index / 2048))
        .collect::<Buffer<_>>();
    let line_item_ids = (0..ROWS).map(u64::from).collect::<Buffer<_>>();
    measure(
        bencher,
        IsCompoundSorted,
        IsCompoundSortedOptions::new([
            ("customer_id", SortDirection::Asc),
            ("order_id", SortDirection::Asc),
            ("line_item_id", SortDirection::Asc),
        ]),
        || {
            StructArray::from_fields(&[
                ("customer_id", customer_ids.clone().into_array()),
                ("order_id", order_ids.clone().into_array()),
                ("line_item_id", line_item_ids.clone().into_array()),
            ])
            .unwrap()
            .into_array()
        },
    );
}

// A batch size of 63 splits orders, so some boundary comparisons reach line-item IDs.
#[divan::bench(args = [63, 64, 1024, 16_384])]
fn order_lines_across_batches(bencher: Bencher, batch_rows: usize) {
    let order_ids = (0..ROWS)
        .map(|index| u64::from(index / 16))
        .collect::<Buffer<_>>();
    let line_item_ids = (0..ROWS)
        .map(|index| u64::from(index % 16))
        .collect::<Buffer<_>>();
    measure_batches(bencher, batch_rows, || {
        StructArray::from_fields(&[
            ("order_id", order_ids.clone().into_array()),
            ("line_item_id", line_item_ids.clone().into_array()),
        ])
        .unwrap()
        .into_array()
    });
}

// Rows without an order ID are sorted by line-item ID. Each order starts with a null item ID.
#[divan::bench]
fn nullable_order_line_keys(bencher: Bencher) {
    let order_ids = (0..ROWS)
        .map(|index| (index >= 1024).then(|| u64::from(index / 16)))
        .collect::<Vec<_>>();
    let line_item_ids = (0..ROWS)
        .map(|index| {
            if index < 1024 {
                Some(u64::from(index))
            } else {
                (index % 16 != 0).then(|| u64::from(index % 16))
            }
        })
        .collect::<Vec<_>>();
    measure(bencher, IsCompoundSorted, order_line_options(), || {
        StructArray::from_fields(&[
            (
                "order_id",
                PrimitiveArray::from_option_iter(order_ids.clone()).into_array(),
            ),
            (
                "line_item_id",
                PrimitiveArray::from_option_iter(line_item_ids.clone()).into_array(),
            ),
        ])
        .unwrap()
        .into_array()
    });
}

// Zero-padding keeps lexicographic and numeric ordering consistent.
#[divan::bench]
fn customer_order_string_keys(bencher: Bencher) {
    let customer_ids = (0..ROWS)
        .map(|index| format!("customer-{:05}", index / 16))
        .collect::<Vec<_>>();
    let order_ids = (0..ROWS)
        .map(|index| format!("order-{:05}", index % 16))
        .collect::<Vec<_>>();
    measure(
        bencher,
        IsCompoundSorted,
        IsCompoundSortedOptions::new([
            ("customer_id", SortDirection::Asc),
            ("order_id", SortDirection::Asc),
        ]),
        || {
            StructArray::from_fields(&[
                (
                    "customer_id",
                    VarBinViewArray::from_iter_str(customer_ids.iter()).into_array(),
                ),
                (
                    "order_id",
                    VarBinViewArray::from_iter_str(order_ids.iter()).into_array(),
                ),
            ])
            .unwrap()
            .into_array()
        },
    );
}

#[divan::bench]
fn order_lines_with_dictionary_item_ids(bencher: Bencher) {
    let order_ids = (0..ROWS)
        .map(|index| u64::from(index / 16))
        .collect::<Buffer<_>>();
    let line_item_codes = (0..ROWS).map(|index| index % 16).collect::<Buffer<u32>>();
    let line_item_dictionary = (0..16u64).collect::<Buffer<_>>();
    measure(bencher, IsCompoundSorted, order_line_options(), || {
        StructArray::from_fields(&[
            ("order_id", order_ids.clone().into_array()),
            (
                "line_item_id",
                DictArray::try_new(
                    line_item_codes.clone().into_array(),
                    line_item_dictionary.clone().into_array(),
                )
                .unwrap()
                .into_array(),
            ),
        ])
        .unwrap()
        .into_array()
    });
}

/// The following bench is to compare against the `IsSorted`
/// aggregate fn. It shouldn't deviate much from it, given
/// that it is a single column/field.
mod single_order_id {
    use super::*;

    fn order_ids() -> Buffer<u64> {
        (0..ROWS).map(u64::from).collect()
    }

    #[divan::bench]
    fn is_compound_sorted(bencher: Bencher) {
        let order_ids = order_ids();
        measure(
            bencher,
            IsCompoundSorted,
            IsCompoundSortedOptions::new([("order_id", SortDirection::Asc)]),
            || {
                StructArray::from_fields(&[("order_id", order_ids.clone().into_array())])
                    .unwrap()
                    .into_array()
            },
        );
    }

    #[divan::bench]
    fn is_sorted(bencher: Bencher) {
        let order_ids = order_ids();
        measure(bencher, IsSorted, IsSortedOptions { strict: false }, || {
            order_ids.clone().into_array()
        });
    }
}
