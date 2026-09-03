// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::DynGroupedAccumulator;
use vortex_array::aggregate_fn::GroupIds;
use vortex_array::aggregate_fn::GroupedAccumulator;
use vortex_array::aggregate_fn::fns::count::Count;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::fns::sum_v2::SumV2;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

const GROUP_COUNT: usize = 128;
const GROUP_SIZE_SEED: u64 = 42;
const MIN_VALUES_PER_GROUP: usize = 1;
const MAX_VALUES_PER_GROUP: usize = 15;
const CARDINALITY_ELEMENT_COUNT: usize = 1 << 16;

#[derive(Clone, Copy, Debug)]
enum IdOrder {
    Clustered,
    Shuffled,
}

const CARDINALITY_ARGS: &[(usize, IdOrder)] = &[
    (128, IdOrder::Clustered),
    (128, IdOrder::Shuffled),
    (1 << 12, IdOrder::Clustered),
    (1 << 12, IdOrder::Shuffled),
    (1 << 16, IdOrder::Clustered),
    (1 << 16, IdOrder::Shuffled),
];

fn random_group_sizes() -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(GROUP_SIZE_SEED);
    (0..GROUP_COUNT)
        .map(|_| rng.random_range(MIN_VALUES_PER_GROUP..=MAX_VALUES_PER_GROUP))
        .collect()
}

fn total_element_count(group_sizes: &[usize]) -> usize {
    group_sizes.iter().sum()
}

struct DenseGroupedInput {
    values: ArrayRef,
    group_ids: GroupIds,
}

impl DenseGroupedInput {
    /// Assign each value the id of the group whose run of `group_sizes` covers it.
    fn clustered(values: ArrayRef, group_sizes: &[usize]) -> Self {
        assert_eq!(values.len(), total_element_count(group_sizes));

        let group_ids = GroupIds::from_iter(
            group_sizes
                .iter()
                .enumerate()
                .flat_map(|(group_id, &size)| std::iter::repeat_n(group_id as u32, size)),
            group_sizes.len(),
        )
        .unwrap();

        Self { values, group_ids }
    }
}

fn i32_nullable_all_valid_input() -> DenseGroupedInput {
    let group_sizes = random_group_sizes();
    let element_count = total_element_count(&group_sizes);
    let values: Buffer<i32> = (0..element_count)
        .map(|i| (i % 1024) as i32 - 512)
        .collect();
    let validity = Validity::from_iter(std::iter::repeat_n(true, element_count));
    DenseGroupedInput::clustered(
        PrimitiveArray::new(values, validity).into_array(),
        &group_sizes,
    )
}

fn i32_clustered_nulls_input() -> DenseGroupedInput {
    let group_sizes = random_group_sizes();
    let element_count = total_element_count(&group_sizes);
    let values = (0..element_count).map(|i| {
        if (i / 16) % 8 == 0 {
            None
        } else {
            Some((i % 1024) as i32 - 512)
        }
    });
    DenseGroupedInput::clustered(
        PrimitiveArray::from_option_iter(values).into_array(),
        &group_sizes,
    )
}

fn f64_all_valid_input() -> DenseGroupedInput {
    let group_sizes = random_group_sizes();
    let element_count = total_element_count(&group_sizes);
    let mut rng = StdRng::seed_from_u64(GROUP_SIZE_SEED);
    let values: Buffer<f64> = (0..element_count)
        .map(|_| rng.random_range(-1000.0..1000.0))
        .collect();
    DenseGroupedInput::clustered(
        PrimitiveArray::new(values, Validity::NonNullable).into_array(),
        &group_sizes,
    )
}

fn f64_clustered_nulls_input() -> DenseGroupedInput {
    let group_sizes = random_group_sizes();
    let element_count = total_element_count(&group_sizes);
    let mut rng = StdRng::seed_from_u64(GROUP_SIZE_SEED);
    let values = (0..element_count).map(|i| {
        if (i / 16) % 8 == 0 {
            None
        } else {
            Some(rng.random_range(-1000.0f64..1000.0))
        }
    });
    DenseGroupedInput::clustered(
        PrimitiveArray::from_option_iter(values).into_array(),
        &group_sizes,
    )
}

fn varbinview_input() -> DenseGroupedInput {
    let group_sizes = random_group_sizes();
    let element_count = total_element_count(&group_sizes);
    let values: Vec<String> = (0..element_count)
        .map(|i| format!("value-{i:06}"))
        .collect();
    DenseGroupedInput::clustered(
        VarBinViewArray::from_iter_str(values.iter().map(String::as_str)).into_array(),
        &group_sizes,
    )
}

fn i32_cardinality_input(group_count: usize, order: IdOrder) -> DenseGroupedInput {
    let mut rng = StdRng::seed_from_u64(GROUP_SIZE_SEED);
    let values: Buffer<i32> = (0..CARDINALITY_ELEMENT_COUNT)
        .map(|_| rng.random_range(-512..512))
        .collect();
    let mut group_ids: Vec<u32> = (0..CARDINALITY_ELEMENT_COUNT)
        .map(|idx| (idx * group_count / CARDINALITY_ELEMENT_COUNT) as u32)
        .collect();
    if matches!(order, IdOrder::Shuffled) {
        group_ids.shuffle(&mut rng);
    }

    DenseGroupedInput {
        values: PrimitiveArray::new(values, Validity::NonNullable).into_array(),
        group_ids: GroupIds::from_buffer(Buffer::from(group_ids), group_count).unwrap(),
    }
}

fn grouped_accumulator<V>(input: &DenseGroupedInput, vtable: V) -> ArrayRef
where
    V: AggregateFnVTable + Clone,
    V::Options: Default,
{
    let mut acc =
        GroupedAccumulator::try_new(vtable, V::Options::default(), input.values.dtype().clone())
            .unwrap();
    let mut ctx = SESSION.create_execution_ctx();
    acc.accumulate(&input.values, &input.group_ids, &mut ctx)
        .unwrap();
    let result = acc
        .finish()
        .unwrap()
        .execute::<Canonical>(&mut ctx)
        .unwrap()
        .into_array();
    divan::black_box(result)
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

#[divan::bench(args = CARDINALITY_ARGS)]
fn sum_i32_cardinality(bencher: Bencher, (group_count, order): (usize, IdOrder)) {
    let input = i32_cardinality_input(group_count, order);
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, Sum));
}

#[divan::bench(args = CARDINALITY_ARGS)]
fn sum_v2_i32_cardinality(bencher: Bencher, (group_count, order): (usize, IdOrder)) {
    let input = i32_cardinality_input(group_count, order);
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, SumV2));
}

#[divan::bench(args = CARDINALITY_ARGS)]
fn count_i32_cardinality(bencher: Bencher, (group_count, order): (usize, IdOrder)) {
    let input = i32_cardinality_input(group_count, order);
    bencher
        .with_inputs(|| &input)
        .bench_refs(|input| grouped_accumulator(input, Count));
}
