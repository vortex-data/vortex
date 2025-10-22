// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod fill_null;
mod filter;
mod mask;
mod min_max;
mod search_sorted;
mod slice;
mod sort;
mod sum;
mod take;

use std::iter;
use std::ops::Range;

pub(crate) use cast::*;
pub(crate) use compare::*;
pub(crate) use fill_null::*;
pub(crate) use filter::*;
use libfuzzer_sys::arbitrary::Error::EmptyChoose;
use libfuzzer_sys::arbitrary::{Arbitrary, Unstructured};
pub(crate) use mask::*;
pub(crate) use min_max::*;
pub(crate) use search_sorted::*;
pub(crate) use slice::*;
pub use sort::sort_canonical_array;
use strum::{EnumCount, EnumDiscriminants, EnumIter, IntoEnumIterator};
pub(crate) use sum::*;
pub(crate) use take::*;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::compute::{MinMaxResult, Operator};
use vortex_array::search_sorted::{SearchResult, SearchSortedSide};
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexUnwrap, vortex_panic};
use vortex_mask::Mask;
use vortex_scalar::Scalar;
use vortex_scalar::arbitrary::random_scalar;
use vortex_utils::aliases::hash_set::HashSet;

#[derive(Debug)]
pub struct FuzzArrayAction {
    pub array: ArrayRef,
    pub actions: Vec<(Action, ExpectedValue)>,
}

#[derive(Debug, Clone, Copy)]
pub enum CompressorStrategy {
    Default,
    Compact,
}

impl<'a> Arbitrary<'a> for CompressorStrategy {
    fn arbitrary(u: &mut Unstructured<'a>) -> libfuzzer_sys::arbitrary::Result<Self> {
        if u.arbitrary()? {
            Ok(CompressorStrategy::Default)
        } else {
            Ok(CompressorStrategy::Compact)
        }
    }
}

#[derive(Debug, EnumCount, EnumDiscriminants)]
#[strum_discriminants(derive(Hash, EnumIter))]
#[strum_discriminants(name(ActionType))]
pub enum Action {
    Compress(CompressorStrategy),
    Slice(Range<usize>),
    Take(ArrayRef),
    SearchSorted(Scalar, SearchSortedSide),
    Filter(Mask),
    Compare(Scalar, Operator),
    Cast(DType),
    Sum,
    MinMax,
    FillNull(Scalar),
    Mask(Mask),
}

#[derive(Debug)]
pub enum ExpectedValue {
    Array(ArrayRef),
    Search(SearchResult),
    Scalar(Scalar),
    MinMax(Option<MinMaxResult>),
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

    pub fn scalar(self) -> Scalar {
        match self {
            ExpectedValue::Scalar(s) => s,
            _ => vortex_panic!("expected scalar"),
        }
    }

    pub fn min_max(self) -> Option<MinMaxResult> {
        match self {
            ExpectedValue::MinMax(m) => m,
            _ => vortex_panic!("expected min_max"),
        }
    }
}

impl<'a> Arbitrary<'a> for FuzzArrayAction {
    fn arbitrary(u: &mut Unstructured<'a>) -> libfuzzer_sys::arbitrary::Result<Self> {
        let array = ArbitraryArray::arbitrary(u)?.0;
        let mut current_array = array.to_array();

        let mut valid_actions = actions_for_dtype(current_array.dtype())
            .into_iter()
            .collect::<Vec<_>>();
        valid_actions.sort_unstable_by_key(|a| *a as usize);

        let mut actions = Vec::new();
        let action_count = u.int_in_range(1..=4)?;
        for _ in 0..action_count {
            let action_type = random_action_from_list(u, valid_actions.as_slice())?;

            actions.push(match action_type {
                ActionType::Compress => {
                    if actions
                        .last()
                        .map(|(l, _)| matches!(l, Action::Compress(_)))
                        .unwrap_or(false)
                    {
                        return Err(EmptyChoose);
                    }
                    let strategy = CompressorStrategy::arbitrary(u)?;
                    (
                        Action::Compress(strategy),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                ActionType::Slice => {
                    let start = u.choose_index(current_array.len())?;
                    let stop = u.int_in_range(start..=current_array.len())?;
                    current_array =
                        slice_canonical_array(&current_array, start, stop).vortex_unwrap();

                    (
                        Action::Slice(start..stop),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                ActionType::Take => {
                    if current_array.is_empty() {
                        return Err(EmptyChoose);
                    }

                    let indices = random_vec_in_range(u, 0, current_array.len() - 1)?;
                    current_array = take_canonical_array(&current_array, &indices).vortex_unwrap();
                    let indices_array = PrimitiveArray::from_option_iter(
                        indices.iter().map(|i| i.map(|i| i as u64)),
                    )
                    .into_array();

                    let compressed = BtrBlocksCompressor::default()
                        .compress(&indices_array)
                        .vortex_unwrap();
                    (
                        Action::Take(compressed),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                ActionType::SearchSorted => {
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
                ActionType::Filter => {
                    let mask = (0..current_array.len())
                        .map(|_| bool::arbitrary(u))
                        .collect::<libfuzzer_sys::arbitrary::Result<Vec<_>>>()?;
                    current_array = filter_canonical_array(&current_array, &mask).vortex_unwrap();
                    (
                        Action::Filter(Mask::from_iter(mask)),
                        ExpectedValue::Array(current_array.to_array()),
                    )
                }
                ActionType::Compare => {
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
                ActionType::Cast => {
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

                    (Action::Cast(to), ExpectedValue::Array(result))
                }
                ActionType::Sum => {
                    // Sum - returns a scalar, does NOT update current_array (terminal operation)
                    let sum_result =
                        sum_canonical_array(current_array.to_canonical()).vortex_unwrap();
                    (Action::Sum, ExpectedValue::Scalar(sum_result))
                }
                ActionType::MinMax => {
                    // MinMax - returns a scalar, does NOT update current_array (terminal operation)
                    let min_max_result =
                        min_max_canonical_array(current_array.to_canonical()).vortex_unwrap();
                    (Action::MinMax, ExpectedValue::MinMax(min_max_result))
                }
                ActionType::FillNull => {
                    // FillNull - returns an array, updates current_array
                    if !current_array.dtype().nullability().is_nullable() {
                        return Err(EmptyChoose);
                    }
                    let fill_value = if u.arbitrary()? && !current_array.is_empty() {
                        current_array.scalar_at(u.choose_index(current_array.len())?)
                    } else {
                        random_scalar(
                            u,
                            &current_array
                                .dtype()
                                .with_nullability(Nullability::NonNullable),
                        )?
                    };

                    if fill_value.is_null() {
                        return Err(EmptyChoose);
                    }

                    // Compute expected result on canonical form
                    let expected_result =
                        fill_null_canonical_array(current_array.to_canonical(), &fill_value)
                            .vortex_unwrap();
                    // Update current_array to the result for chaining
                    current_array = expected_result.clone();
                    (
                        Action::FillNull(fill_value),
                        ExpectedValue::Array(expected_result),
                    )
                }
                ActionType::Mask => {
                    // Mask - returns an array, updates current_array
                    let mask = (0..current_array.len())
                        .map(|_| bool::arbitrary(u))
                        .collect::<libfuzzer_sys::arbitrary::Result<Vec<_>>>()?;

                    // Compute expected result on canonical form
                    let expected_result = mask_canonical_array(
                        current_array.to_canonical(),
                        &Mask::from_iter(mask.iter().copied()),
                    )
                    .vortex_unwrap();
                    // Update current_array to the result for chaining
                    current_array = expected_result.clone();
                    (
                        Action::Mask(Mask::from_iter(mask)),
                        ExpectedValue::Array(expected_result),
                    )
                }
            })
        }

        Ok(Self { array, actions })
    }
}

fn actions_for_dtype(dtype: &DType) -> HashSet<ActionType> {
    use ActionType::*;

    match dtype {
        DType::Struct(sdt, _) => {
            // Struct supports: Compress, Slice, Take, Filter, MinMax, Mask
            // Does NOT support: SearchSorted (requires scalar comparison), Compare, Cast, Sum, FillNull
            let struct_actions = [Compress, Slice, Take, Filter, MinMax, Mask];
            sdt.fields()
                .map(|child| actions_for_dtype(&child))
                .fold(struct_actions.into(), |acc, actions| {
                    acc.intersection(&actions).copied().collect()
                })
        }
        DType::List(..) | DType::FixedSizeList(..) => {
            // List supports: Compress, Slice, Take, Filter, MinMax, Mask
            // Does NOT support: SearchSorted, Compare, Cast, Sum, FillNull
            [Compress, Slice, Take, Filter, MinMax, Mask].into()
        }
        DType::Utf8(_) | DType::Binary(_) => {
            // Utf8/Binary supports everything except Sum
            // Actions: Compress, Slice, Take, SearchSorted, Filter, Compare, Cast, MinMax, FillNull, Mask
            [
                Compress,
                Slice,
                Take,
                SearchSorted,
                Filter,
                Compare,
                Cast,
                MinMax,
                FillNull,
                Mask,
            ]
            .into()
        }
        DType::Bool(_) | DType::Primitive(..) | DType::Decimal(..) => {
            // These support all actions
            ActionType::iter().collect()
        }
        DType::Null => {
            // Null arrays support most operations but not Sum or MinMax (return None for dtype)
            [
                Compress,
                Slice,
                Take,
                SearchSorted,
                Filter,
                Compare,
                Cast,
                FillNull,
                Mask,
            ]
            .into()
        }
        DType::Extension(_) => {
            // Extension types delegate to storage dtype, support most operations
            ActionType::iter().collect()
        }
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

fn random_action_from_list(
    u: &mut Unstructured<'_>,
    actions: &[ActionType],
) -> libfuzzer_sys::arbitrary::Result<ActionType> {
    u.choose_iter(actions).copied()
}
