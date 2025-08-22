// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod filter;
mod search_sorted;
mod slice;
mod sort;
mod take;

use std::iter;
use std::ops::Range;

pub(crate) use cast::*;
pub(crate) use compare::*;
pub(crate) use filter::*;
use libfuzzer_sys::arbitrary::Error::EmptyChoose;
use libfuzzer_sys::arbitrary::{Arbitrary, Unstructured};
pub(crate) use search_sorted::*;
pub(crate) use slice::*;
pub use sort::sort_canonical_array;
use strum::EnumCount;
pub(crate) use take::*;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::compute::Operator;
use vortex_array::search_sorted::{SearchResult, SearchSortedSide};
use vortex_array::{ArrayRef, IntoArray};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexUnwrap, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::Scalar;
use vortex_scalar::arbitrary::random_scalar;
use vortex_utils::aliases::hash_set::HashSet;

use crate::array::Action::Cast;

#[derive(Debug)]
pub struct FuzzArrayAction {
    pub array: ArrayRef,
    pub actions: Vec<(Action, ExpectedValue)>,
}

#[derive(Debug, EnumCount)]
pub enum Action {
    Compress,
    Slice(Range<usize>),
    Take(ArrayRef),
    SearchSorted(Scalar, SearchSortedSide),
    Filter(Mask),
    Compare(Scalar, Operator),
    Cast(DType),
}

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

const ALL_ACTIONS: Range<usize> = 0..Action::COUNT;

impl<'a> Arbitrary<'a> for FuzzArrayAction {
    fn arbitrary(u: &mut Unstructured<'a>) -> libfuzzer_sys::arbitrary::Result<Self> {
        let array = ArbitraryArray::arbitrary(u)?.0;
        let mut current_array = array.to_array();

        let mut valid_actions = actions_for_dtype(current_array.dtype())
            .into_iter()
            .collect::<Vec<_>>();
        valid_actions.sort_unstable();

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
                    let indices_array = PrimitiveArray::from_option_iter(
                        indices.iter().map(|i| i.map(|i| i as u64)),
                    )
                    .into_array();

                    let compressed = BtrBlocksCompressor.compress(&indices_array).vortex_unwrap();
                    (
                        Action::Take(compressed),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                3 => {
                    if current_array.dtype().is_struct() {
                        return Err(EmptyChoose);
                    }

                    let scalar = if u.arbitrary()? {
                        current_array.scalar_at(u.choose_index(current_array.len())?)
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
                        .collect::<libfuzzer_sys::arbitrary::Result<Vec<_>>>()?;
                    current_array = filter_canonical_array(&current_array, &mask).vortex_unwrap();
                    (
                        Action::Filter(Mask::from_iter(mask)),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                5 => {
                    let scalar = if u.arbitrary()? {
                        current_array.scalar_at(u.choose_index(current_array.len())?)
                    } else {
                        // We can compare arrays with different nullability
                        let null: Nullability = u.arbitrary()?;
                        random_scalar(u, &current_array.dtype().union_nullability(null))?
                    };

                    let op = u.arbitrary()?;
                    current_array =
                        compare_canonical_array(&current_array, &scalar, op).vortex_unwrap();
                    (
                        Action::Compare(scalar, op),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                6 => {
                    let to: DType = u.arbitrary()?;
                    if Some(CastOutcome::Infallible) == allowed_casting(current_array.dtype(), &to)
                    {
                        return Err(EmptyChoose);
                    }
                    let Some(result) = cast_canonical_array(&current_array, &to)
                        .vortex_expect("should fail to create array")
                    else {
                        return Err(EmptyChoose);
                    };

                    (Cast(to), ExpectedValue::Array(result))
                }
                7.. => unreachable!(),
            })
        }

        Ok(Self { array, actions })
    }
}

fn actions_for_dtype(dtype: &DType) -> HashSet<usize> {
    match dtype {
        DType::Struct(sdt, _) => sdt
            .fields()
            .map(|child| actions_for_dtype(&child))
            // exclude compare
            .fold((0..=4).chain(iter::once(6)).collect(), |acc, actions| {
                acc.intersection(&actions).copied().collect()
            }),
        // Once we support more list operations also recurse here on child dtype
        // compress, slice
        DType::List(..) => [0, 1].into_iter().collect(),
        _ => ALL_ACTIONS.collect(),
    }
}

fn random_vec_in_range(
    u: &mut Unstructured<'_>,
    min: usize,
    max: usize,
) -> libfuzzer_sys::arbitrary::Result<Vec<Option<usize>>> {
    iter::from_fn(|| {
        u.arbitrary().unwrap_or(false).then(|| {
            if u.arbitrary()? {
                Ok(None)
            } else {
                Ok(Some(u.int_in_range(min..=max)?))
            }
        })
    })
    .collect::<libfuzzer_sys::arbitrary::Result<Vec<_>>>()
}

fn random_value_from_list(
    u: &mut Unstructured<'_>,
    vec: &[usize],
) -> libfuzzer_sys::arbitrary::Result<usize> {
    u.choose_iter(vec).cloned()
}
