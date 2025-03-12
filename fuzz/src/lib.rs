mod compare;
mod filter;
mod search_sorted;
mod slice;
mod sort;
mod take;

use std::fmt::Debug;
use std::iter;
use std::ops::{Range, RangeInclusive};

use libfuzzer_sys::arbitrary::Error::EmptyChoose;
use libfuzzer_sys::arbitrary::{Arbitrary, Result, Unstructured};
pub use sort::sort_canonical_array;
use vortex_array::aliases::hash_set::HashSet;
use vortex_array::arrays::ListEncoding;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::compute::{Operator, SearchResult, SearchSortedSide, scalar_at};
use vortex_array::vtable::EncodingVTable;
use vortex_array::{Array, ArrayRef, ArrayVisitorExt, EncodingId, IntoArray};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::Buffer;
use vortex_error::{VortexUnwrap, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::Scalar;
use vortex_scalar::arbitrary::random_scalar;

use crate::compare::compare_canonical_array;
use crate::filter::filter_canonical_array;
use crate::search_sorted::search_sorted_canonical_array;
use crate::slice::slice_canonical_array;
use crate::take::take_canonical_array;

#[derive(Debug)]
pub enum ExpectedValue {
    Array(ArrayRef),
    Search(SearchResult),
}

impl ExpectedValue {
    pub fn array(self) -> ArrayRef {
        match self {
            ExpectedValue::Array(array) => array,
            _ => vortex_panic!("expected array"),
        }
    }

    pub fn search(self) -> SearchResult {
        match self {
            ExpectedValue::Search(s) => s,
            _ => vortex_panic!("expected search"),
        }
    }
}

#[derive(Debug)]
pub struct FuzzArrayAction {
    pub array: ArrayRef,
    pub actions: Vec<(Action, ExpectedValue)>,
}

#[derive(Debug)]
pub enum Action {
    Compress,
    Slice(Range<usize>),
    Take(ArrayRef),
    SearchSorted(Scalar, SearchSortedSide),
    Filter(Mask),
    Compare(Scalar, Operator),
}

impl<'a> Arbitrary<'a> for FuzzArrayAction {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let array = ArbitraryArray::arbitrary(u)?.0;
        let mut current_array = array.to_array();

        let valid_actions = actions_for_array(&current_array);

        let mut actions = Vec::new();
        let action_count = u.int_in_range(1..=4)?;
        for _ in 0..action_count {
            actions.push(match random_value_from_list(u, valid_actions.as_slice())? {
                0 => {
                    if actions
                        .last()
                        .map(|(l, _)| matches!(l, Action::Compress))
                        .unwrap_or(false)
                    {
                        return Err(EmptyChoose);
                    }
                    (
                        Action::Compress,
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                1 => {
                    let start = u.choose_index(current_array.len())?;
                    let stop = u.int_in_range(start..=current_array.len())?;
                    current_array =
                        slice_canonical_array(&current_array, start, stop).vortex_unwrap();

                    (
                        Action::Slice(start..stop),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                2 => {
                    if current_array.is_empty() {
                        return Err(EmptyChoose);
                    }

                    let indices = random_vec_in_range(u, 0, current_array.len() - 1)?;
                    current_array = take_canonical_array(&current_array, &indices).vortex_unwrap();
                    let indices_array = indices
                        .iter()
                        .map(|i| *i as u64)
                        .collect::<Buffer<u64>>()
                        .into_array();
                    let compressed = BtrBlocksCompressor.compress(&indices_array).vortex_unwrap();
                    (
                        Action::Take(compressed.into_array()),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                3 => {
                    if current_array.dtype().is_struct() {
                        return Err(EmptyChoose);
                    }

                    let scalar = if u.arbitrary()? {
                        scalar_at(&current_array, u.choose_index(current_array.len())?)
                            .vortex_unwrap()
                    } else {
                        random_scalar(u, current_array.dtype())?
                    };

                    if scalar.is_null() {
                        return Err(EmptyChoose);
                    }

                    let sorted = sort_canonical_array(&current_array).vortex_unwrap();

                    let side = if u.arbitrary()? {
                        SearchSortedSide::Left
                    } else {
                        SearchSortedSide::Right
                    };
                    (
                        Action::SearchSorted(scalar.clone(), side),
                        ExpectedValue::Search(
                            search_sorted_canonical_array(&sorted, &scalar, side).vortex_unwrap(),
                        ),
                    )
                }
                4 => {
                    let mask = (0..current_array.len())
                        .map(|_| bool::arbitrary(u))
                        .collect::<Result<Vec<_>>>()?;
                    current_array = filter_canonical_array(&current_array, &mask).vortex_unwrap();
                    (
                        Action::Filter(Mask::from_iter(mask)),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                5 => {
                    let scalar = if u.arbitrary()? {
                        scalar_at(&current_array, u.choose_index(current_array.len())?)
                            .vortex_unwrap()
                    } else {
                        random_scalar(u, current_array.dtype())?
                    };

                    let op = u.arbitrary()?;
                    current_array =
                        compare_canonical_array(&current_array, &scalar, op).vortex_unwrap();
                    (
                        Action::Compare(scalar, op),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                _ => unreachable!(),
            })
        }

        Ok(Self { array, actions })
    }
}

fn random_vec_in_range(u: &mut Unstructured<'_>, min: usize, max: usize) -> Result<Vec<usize>> {
    iter::from_fn(|| {
        u.arbitrary()
            .unwrap_or(false)
            .then(|| u.int_in_range(min..=max))
    })
    .collect::<Result<Vec<_>>>()
}

fn random_value_from_list(u: &mut Unstructured<'_>, vec: &[usize]) -> Result<usize> {
    u.choose_iter(vec).cloned()
}

const ALL_ACTIONS: RangeInclusive<usize> = 0..=5;

fn actions_for_encoding(encoding_id: EncodingId) -> HashSet<usize> {
    if ListEncoding.id() == encoding_id {
        // compress, slice
        vec![0, 1].into_iter().collect()
    } else {
        ALL_ACTIONS.collect()
    }
}

fn actions_for_array(array: &dyn Array) -> Vec<usize> {
    array
        .depth_first_traversal()
        .map(|child| actions_for_encoding(child.encoding()))
        .fold(ALL_ACTIONS.collect::<Vec<_>>(), |mut acc, actions| {
            acc.retain(|a| actions.contains(a));
            acc
        })
}
