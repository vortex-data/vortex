// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

pub mod fixture;

use std::sync::LazyLock;

use divan::Bencher;
use fixture::DistributionVortex;
use fixture::VortexColumns;
use mimalloc::MiMalloc;
use populate_builtins::constant;
use populate_builtins::draw_if;
use populate_builtins::indexed;
use populate_builtins::nullable;
use populate_builtins::sample_range;
use populate_core::Columns;
use populate_core::Distribution;
use populate_rand::SeedableRng;
use populate_rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynGroupedAccumulator;
use vortex_array::aggregate_fn::GroupedAccumulator;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::count::Count;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::test_harness::mark_optimized;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<KernelSession>()
        .with::<AggregateFnSession>()
});

const GROUP_COUNT: usize = 128;
const GROUP_SIZE_SEED: u64 = 42;
const MIN_VALUES_PER_GROUP: usize = 1;
const MAX_VALUES_PER_GROUP: usize = 15;

fn random_group_sizes() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(GROUP_SIZE_SEED);
    sample_range(MIN_VALUES_PER_GROUP..=MAX_VALUES_PER_GROUP).collect(&mut rng, GROUP_COUNT)
}

fn total_element_count(group_sizes: &[usize]) -> usize {
    group_sizes.iter().sum()
}

fn contiguous_list_view(elements: ArrayRef, group_sizes: &[usize]) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let mut offset = 0usize;
    let offsets: Buffer<u32> = group_sizes
        .iter()
        .map(|&size| {
            let current_offset = offset;
            offset += size;
            current_offset as u32
        })
        .collect();
    let sizes: Buffer<u32> = group_sizes.iter().map(|&size| size as u32).collect();

    assert_eq!(elements.len(), total_element_count(group_sizes));

    let list_view = ListViewArray::try_new(
        elements,
        offsets.into_array(),
        sizes.into_array(),
        Validity::NonNullable,
        &mut ctx,
    )
    .unwrap()
    .into_array();
    mark_optimized(&list_view, &SESSION).unwrap();

    list_view
}

fn grouped_input<D: Distribution>(mut values: D) -> ArrayRef
where
    Columns<D::Item>: VortexColumns,
{
    let sizes = random_group_sizes();
    let mut rng = StdRng::seed_from_u64(GROUP_SIZE_SEED);
    contiguous_list_view(
        values.collect_vortex(&mut rng, total_element_count(&sizes)),
        &sizes,
    )
}

fn i32_value(i: usize) -> i32 {
    (i % 1024) as i32 - 512
}

fn present(i: usize) -> bool {
    !(i / 16).is_multiple_of(8)
}

fn i32_nullable_all_valid_input() -> ArrayRef {
    grouped_input(nullable(indexed(i32_value), constant(true)))
}

fn i32_clustered_nulls_input() -> ArrayRef {
    grouped_input(indexed(|i| present(i).then(|| i32_value(i))))
}

fn f64_all_valid_input() -> ArrayRef {
    grouped_input(sample_range(-1000.0f64..1000.0))
}

fn f64_clustered_nulls_input() -> ArrayRef {
    grouped_input(draw_if(indexed(present), sample_range(-1000.0f64..1000.0)))
}

fn varbinview_input() -> ArrayRef {
    let group_sizes = random_group_sizes();
    let element_count = total_element_count(&group_sizes);
    let values: Vec<String> = (0..element_count)
        .map(|i| format!("value-{i:06}"))
        .collect();
    contiguous_list_view(
        VarBinViewArray::from_iter_str(values.iter().map(String::as_str)).into_array(),
        &group_sizes,
    )
}

fn list_element_dtype(list_view: &ArrayRef) -> DType {
    match list_view.dtype() {
        DType::List(element_dtype, _) => element_dtype.as_ref().clone(),
        dtype => unreachable!("expected List dtype, got {dtype}"),
    }
}

fn grouped_accumulator<V>(list_view: &ArrayRef, vtable: V) -> ArrayRef
where
    V: AggregateFnVTable<Options = NumericalAggregateOpts> + Clone,
{
    let mut acc = GroupedAccumulator::try_new(
        vtable,
        NumericalAggregateOpts::default(),
        list_element_dtype(list_view),
    )
    .unwrap();
    acc.accumulate_list(list_view, &mut SESSION.create_execution_ctx())
        .unwrap();
    divan::black_box(acc.finish().unwrap())
}

#[divan::bench]
fn sum_i32_nullable_all_valid(bencher: Bencher) {
    let input = i32_nullable_all_valid_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, Sum));
}

#[divan::bench]
fn sum_i32_clustered_nulls(bencher: Bencher) {
    let input = i32_clustered_nulls_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, Sum));
}

#[divan::bench]
fn sum_f64_all_valid(bencher: Bencher) {
    let input = f64_all_valid_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, Sum));
}

#[divan::bench]
fn sum_f64_clustered_nulls(bencher: Bencher) {
    let input = f64_clustered_nulls_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, Sum));
}

#[divan::bench]
fn sum_v2_i32_nullable_all_valid(bencher: Bencher) {
    let input = i32_nullable_all_valid_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, SumV2));
}

#[divan::bench]
fn sum_v2_i32_clustered_nulls(bencher: Bencher) {
    let input = i32_clustered_nulls_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, SumV2));
}

#[divan::bench]
fn sum_v2_f64_all_valid(bencher: Bencher) {
    let input = f64_all_valid_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, SumV2));
}

#[divan::bench]
fn sum_v2_f64_clustered_nulls(bencher: Bencher) {
    let input = f64_clustered_nulls_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, SumV2));
}

/// Execute the lazy finalize result to canonical form so the benchmark includes the full cost of
/// producing usable sums.
fn grouped_accumulator_canonical<V>(list_view: &ArrayRef, vtable: V) -> ArrayRef
where
    V: AggregateFnVTable<Options = NumericalAggregateOpts> + Clone,
{
    let mut acc = GroupedAccumulator::try_new(
        vtable,
        NumericalAggregateOpts::default(),
        list_element_dtype(list_view),
    )
    .unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    acc.accumulate_list(list_view, &mut ctx).unwrap();
    let result = acc
        .finish()
        .unwrap()
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_array();
    divan::black_box(result)
}

#[divan::bench]
fn canonical_sum_i32_nullable_all_valid(bencher: Bencher) {
    let input = i32_nullable_all_valid_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator_canonical(input, Sum));
}

#[divan::bench]
fn canonical_sum_i32_clustered_nulls(bencher: Bencher) {
    let input = i32_clustered_nulls_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator_canonical(input, Sum));
}

#[divan::bench]
fn canonical_sum_f64_all_valid(bencher: Bencher) {
    let input = f64_all_valid_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator_canonical(input, Sum));
}

#[divan::bench]
fn canonical_sum_f64_clustered_nulls(bencher: Bencher) {
    let input = f64_clustered_nulls_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator_canonical(input, Sum));
}

#[divan::bench]
fn canonical_sum_v2_i32_nullable_all_valid(bencher: Bencher) {
    let input = i32_nullable_all_valid_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator_canonical(input, SumV2));
}

#[divan::bench]
fn canonical_sum_v2_i32_clustered_nulls(bencher: Bencher) {
    let input = i32_clustered_nulls_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator_canonical(input, SumV2));
}

#[divan::bench]
fn canonical_sum_v2_f64_all_valid(bencher: Bencher) {
    let input = f64_all_valid_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator_canonical(input, SumV2));
}

#[divan::bench]
fn canonical_sum_v2_f64_clustered_nulls(bencher: Bencher) {
    let input = f64_clustered_nulls_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator_canonical(input, SumV2));
}

#[divan::bench]
fn count_i32_clustered_nulls(bencher: Bencher) {
    let input = i32_clustered_nulls_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, Count));
}

#[divan::bench]
fn count_varbinview(bencher: Bencher) {
    let input = varbinview_input();
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, Count));
}
