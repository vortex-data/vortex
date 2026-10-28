// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::ops::Range;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

use super::PrimitiveRead;

#[rstest]
#[case::adjacent(vec![0, 1, 512, 1026], vec![80..4184, 8288..8296])]
#[case::isolated(vec![0, 1024, 2048], vec![80..88, 8272..8280, 16464..16472])]
fn coalesces_byte_ranges(#[case] indices: Vec<usize>, #[case] expected: Vec<Range<usize>>) {
    let mask = Mask::from_indices(65536, indices);
    let demand = PrimitiveRead::new(
        ReadContext::new(ArrayContext::empty().to_ids()),
        1 << 20,
        8,
        10,
        mask,
    );
    assert_eq!(demand.ranges(), Some(expected));
}

#[rstest]
#[case::empty(Mask::new_false(1024), None)]
#[case::small_contiguous(Mask::new_true(1024), Some(vec![80..8272]))]
#[case::too_many_bytes(Mask::new_true(32768), None)]
fn handles_uniform_masks(#[case] mask: Mask, #[case] expected: Option<Vec<Range<usize>>>) {
    let demand = PrimitiveRead::new(
        ReadContext::new(ArrayContext::empty().to_ids()),
        1 << 20,
        8,
        10,
        mask,
    );
    assert_eq!(demand.ranges(), expected);
}

#[rstest]
#[case::range_count(8192, 1 << 30, 33)]
#[case::byte_budget(8, 65536, 129)]
fn stops_at_the_read_budget(
    #[case] stride: usize,
    #[case] buffer_len: usize,
    #[case] count: usize,
) {
    let visited = Cell::new(0);
    let rows = (0..1000).map(|index| {
        visited.set(visited.get() + 1);
        (index * stride, index * stride + 1)
    });
    let demand = PrimitiveRead::new(
        ReadContext::new(ArrayContext::empty().to_ids()),
        buffer_len,
        8,
        0,
        Mask::new_true(0),
    );
    assert!(demand.collect_ranges(rows).is_none());
    assert_eq!(visited.get(), count);
}

#[rstest]
#[case::cached(true)]
#[case::uncached(false)]
fn preserves_ranges_without_materializing_a_new_mask_cache(#[case] populate: bool) {
    let mask = Mask::from_indices(131072, vec![0, 1, 1024, 2048]);
    let Mask::Values(values) = &mask else {
        unreachable!("the selection is neither empty nor full");
    };
    if populate {
        values.slices();
    }
    let demand = PrimitiveRead::new(
        ReadContext::new(ArrayContext::empty().to_ids()),
        1 << 20,
        8,
        0,
        mask.clone(),
    );
    assert_eq!(demand.ranges(), Some(vec![0..16, 8192..8200, 16384..16392]));
    assert_eq!(values.cached_slices().is_some(), populate);
}

#[rstest]
#[case::zero_width(0, 0)]
#[case::past_buffer(8, 131072)]
#[case::overflow(8, usize::MAX)]
fn rejects_invalid_bounds(#[case] width: usize, #[case] offset: usize) {
    let demand = PrimitiveRead::new(
        ReadContext::new(ArrayContext::empty().to_ids()),
        1 << 20,
        width,
        offset,
        Mask::new_true(1),
    );
    assert!(demand.ranges().is_none());
}
