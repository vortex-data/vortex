// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use divan::Bencher;
use divan::counter::ItemsCount;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::StdRng;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::search_sorted::NullEquality;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::search_sorted::SortedArray;
use vortex_array::search_sorted::SortedDirection;
use vortex_array::search_sorted::SortedNulls;
use vortex_array::search_sorted::SortedOrder;
use vortex_array::search_sorted::sorted_membership_mask;
use vortex_buffer::BitBufferMut;
use vortex_utils::aliases::hash_set::HashSet;

fn main() {
    divan::main();
}

/// One search over 65536 elements is ~16 comparisons, small enough that the measurement is mostly
/// harness overhead and the two implementations below become indistinguishable. Search a batch per
/// iteration instead; the targets differ, so the calls cannot be folded together.
const SEARCH_TARGETS: usize = 64;

#[divan::bench]
fn binary_search_std(bencher: Bencher) {
    let (sorted_array, targets) = fixture();
    bencher
        .with_inputs(|| (&sorted_array, &targets))
        .bench_refs(|(array, targets)| {
            let mut found = 0;
            for target in targets.iter() {
                found += array.binary_search(target).unwrap_or_else(|idx| idx);
            }
            found
        });
}

#[divan::bench]
fn binary_search_vortex(bencher: Bencher) {
    let (sorted_array, targets) = fixture();
    bencher
        .with_inputs(|| (&sorted_array, &targets))
        .bench_refs(|(array, targets)| {
            let mut found = 0;
            for target in targets.iter() {
                found += array
                    .search_sorted(target, SearchSortedSide::Left)
                    .unwrap()
                    .to_index();
            }
            found
        });
}

#[divan::bench]
fn sorted_varbin_membership(bencher: Bencher) {
    let values = (0_u128..65_536)
        .map(|value| value.to_be_bytes())
        .collect::<Vec<_>>();
    let members = values.iter().step_by(16).cloned().collect::<Vec<_>>();
    let values = VarBinViewArray::from_iter_bin(values).into_array();
    let mut ctx = array_session().create_execution_ctx();
    let members = SortedArray::try_new(
        VarBinViewArray::from_iter_bin(members).into_array(),
        ascending(),
        &mut ctx,
    )
    .unwrap();

    bencher.bench_local(|| {
        sorted_membership_mask(
            divan::black_box(&values),
            divan::black_box(&members),
            NullEquality::Unequal,
            &mut ctx,
        )
        .unwrap()
        .true_count()
    });
}

#[divan::bench]
fn sorted_i64_membership(bencher: Bencher) {
    let values = PrimitiveArray::from_iter(0_i64..65_536).into_array();
    let mut ctx = array_session().create_execution_ctx();
    let members = SortedArray::try_new(
        PrimitiveArray::from_iter((0_i64..65_536).step_by(16)).into_array(),
        ascending(),
        &mut ctx,
    )
    .unwrap();

    bencher.bench_local(|| {
        sorted_membership_mask(
            divan::black_box(&values),
            divan::black_box(&members),
            NullEquality::Unequal,
            &mut ctx,
        )
        .unwrap()
        .true_count()
    });
}

fn ascending() -> SortedOrder {
    SortedOrder {
        direction: SortedDirection::Ascending,
        nulls: SortedNulls::First,
    }
}

mod membership_comparison {
    use super::*;

    /// One engine-sized probe chunk against increasingly large/sparse member
    /// domains. Construction and probe costs are separated because engines
    /// may already own either a sorted member array or a hash index.
    const PROBE_ROWS: usize = 8_192;
    const CASES: &[(usize, i64)] = &[(16_384, 1), (65_536, 4), (1_000_000, 16)];

    struct Fixture {
        values: Vec<i64>,
        values_array: vortex_array::ArrayRef,
        members: Vec<i64>,
        sorted_members: SortedArray,
        hashed_members: HashSet<i64>,
    }

    impl Fixture {
        fn new(member_count: usize, stride: i64) -> Self {
            let members = (0..member_count)
                .map(|index| index as i64 * stride)
                .collect::<Vec<_>>();
            let domain = member_count as i64 * stride;
            let start = domain / 2 - PROBE_ROWS as i64 / 2;
            let values = (start..start + PROBE_ROWS as i64).collect::<Vec<_>>();
            let values_array = PrimitiveArray::from_iter(values.iter().copied()).into_array();
            let mut ctx = array_session().create_execution_ctx();
            let sorted_members = SortedArray::try_new(
                PrimitiveArray::from_iter(members.iter().copied()).into_array(),
                ascending(),
                &mut ctx,
            )
            .unwrap();
            let hashed_members = members.iter().copied().collect();
            Self {
                values,
                values_array,
                members,
                sorted_members,
                hashed_members,
            }
        }
    }

    #[divan::bench(args = CASES)]
    fn narrowed_merge(bencher: Bencher, &(members, stride): &(usize, i64)) {
        let fixture = Fixture::new(members, stride);
        let mut ctx = array_session().create_execution_ctx();
        bencher
            .counter(ItemsCount::new(PROBE_ROWS))
            .bench_local(|| {
                sorted_membership_mask(
                    divan::black_box(&fixture.values_array),
                    divan::black_box(&fixture.sorted_members),
                    NullEquality::Unequal,
                    &mut ctx,
                )
                .unwrap()
                .true_count()
            });
    }

    #[divan::bench(args = CASES)]
    fn full_merge(bencher: Bencher, &(members, stride): &(usize, i64)) {
        let fixture = Fixture::new(members, stride);
        bencher
            .counter(ItemsCount::new(PROBE_ROWS))
            .bench_local(|| full_merge_mask(&fixture.values, &fixture.members));
    }

    #[divan::bench(args = CASES)]
    fn per_row_binary_search(bencher: Bencher, &(members, stride): &(usize, i64)) {
        let fixture = Fixture::new(members, stride);
        bencher
            .counter(ItemsCount::new(PROBE_ROWS))
            .bench_local(|| binary_search_mask(&fixture.values, &fixture.members));
    }

    #[divan::bench(args = CASES)]
    fn hash_probe(bencher: Bencher, &(members, stride): &(usize, i64)) {
        let fixture = Fixture::new(members, stride);
        bencher
            .counter(ItemsCount::new(PROBE_ROWS))
            .bench_local(|| hash_mask(&fixture.values, &fixture.hashed_members));
    }

    #[divan::bench(args = CASES)]
    fn sorted_wrapper_build(bencher: Bencher, &(members, stride): &(usize, i64)) {
        let fixture = Fixture::new(members, stride);
        let array = PrimitiveArray::from_iter(fixture.members.iter().copied()).into_array();
        let mut ctx = array_session().create_execution_ctx();
        bencher.counter(ItemsCount::new(members)).bench_local(|| {
            SortedArray::try_new(divan::black_box(array.clone()), ascending(), &mut ctx)
                .unwrap()
                .len()
        });
    }

    #[divan::bench(args = CASES)]
    fn hash_set_build(bencher: Bencher, &(members, stride): &(usize, i64)) {
        let fixture = Fixture::new(members, stride);
        bencher.counter(ItemsCount::new(members)).bench_local(|| {
            fixture
                .members
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
        });
    }

    fn full_merge_mask(values: &[i64], members: &[i64]) -> usize {
        let mut bits = BitBufferMut::with_capacity(values.len());
        let mut member = 0;
        for value in values {
            while member < members.len() && members[member] < *value {
                member += 1;
            }
            bits.append(member < members.len() && members[member] == *value);
        }
        bits.freeze().iter().filter(|selected| *selected).count()
    }

    fn binary_search_mask(values: &[i64], members: &[i64]) -> usize {
        let mut bits = BitBufferMut::with_capacity(values.len());
        for value in values {
            bits.append(members.binary_search(value).is_ok());
        }
        bits.freeze().iter().filter(|selected| *selected).count()
    }

    fn hash_mask(values: &[i64], members: &HashSet<i64>) -> usize {
        let mut bits = BitBufferMut::with_capacity(values.len());
        for value in values {
            bits.append(members.contains(value));
        }
        bits.freeze().iter().filter(|selected| *selected).count()
    }
}

fn fixture() -> (Vec<i32>, Vec<i32>) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0, 65_536).unwrap();
    let mut data: Vec<i32> = (0..65_536).map(|_| rng.sample(range)).collect();
    data.sort();

    let targets = (0..SEARCH_TARGETS).map(|_| rng.sample(range)).collect();

    (data, targets)
}
