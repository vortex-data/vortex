// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) use cast::*;
pub(crate) use compare::*;
pub(crate) use fill_null::*;
pub(crate) use filter::*;
pub(crate) use mask::*;
pub(crate) use min_max::*;
pub(crate) use scalar_at::*;
pub(crate) use search_sorted::*;
pub(crate) use slice::*;
pub use sort::sort_canonical_array;
pub(crate) use sum::*;
pub(crate) use take::*;

mod cast;
mod compare;
mod fill_null;
mod filter;
mod mask;
mod min_max;
mod scalar_at;
mod search_sorted;
mod slice;
mod sort;
mod sum;
mod take;

use std::iter;
use std::ops::Range;

use arbitrary::Arbitrary;
use arbitrary::Error::EmptyChoose;
use arbitrary::Unstructured;
use itertools::Itertools;
use strum::EnumCount;
use strum::EnumDiscriminants;
use strum::EnumIter;
use strum::IntoEnumIterator;
use tracing::debug;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::arbitrary::random_scalar;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_btrblocks::BtrBlocksCompressor;
#[cfg(feature = "zstd")]
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_set::HashSet;

use crate::SESSION;
use crate::error::Backtrace;
use crate::error::VortexFuzzError;
use crate::error::VortexFuzzResult;

#[derive(Debug)]
pub struct FuzzArrayAction {
    pub array: ArrayRef,
    pub actions: Vec<(Action, ExpectedValue)>,
}

#[derive(Debug, Clone, Copy)]
pub enum CompressorStrategy {
    Default,
    #[cfg(feature = "zstd")]
    Compact,
}

impl<'a> Arbitrary<'a> for CompressorStrategy {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        #[cfg(feature = "zstd")]
        {
            if u.arbitrary()? {
                Ok(CompressorStrategy::Default)
            } else {
                Ok(CompressorStrategy::Compact)
            }
        }
        #[cfg(not(feature = "zstd"))]
        {
            let _ = u;
            Ok(CompressorStrategy::Default)
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
    Compare(Scalar, CompareOperator),
    Cast(DType),
    Sum,
    MinMax,
    FillNull(Scalar),
    Mask(Mask),
    // Here we want to try multiple values.
    ScalarAt(Vec<usize>),
}

#[derive(Debug, Clone)]
pub enum ExpectedValue {
    Array(ArrayRef),
    Search(SearchResult),
    Scalar(Scalar),
    MinMax(Option<MinMaxResult>),
    ScalarVec(Vec<Scalar>),
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

    pub fn scalar_vec(self) -> Vec<Scalar> {
        match self {
            ExpectedValue::ScalarVec(v) => v,
            _ => vortex_panic!("expected scalar_vec"),
        }
    }
}

impl<'a> Arbitrary<'a> for FuzzArrayAction {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let array = ArbitraryArray::arbitrary(u)?.0;
        let mut current_array = array.clone();

        let mut ctx = SESSION.create_execution_ctx();

        let mut valid_actions = actions_for_dtype(current_array.dtype())
            .into_iter()
            .collect::<Vec<_>>();
        valid_actions.sort_unstable_by_key(|a| *a as usize);

        let mut actions = Vec::new();
        let action_count = u.int_in_range(1..=4.min(valid_actions.len()))?;
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
                        ExpectedValue::Array(current_array.clone()),
                    )
                }
                ActionType::Slice => {
                    let start = u.choose_index(current_array.len())?;
                    let stop = u.int_in_range(start..=current_array.len())?;
                    current_array = slice_canonical_array(&current_array, start, stop, &mut ctx)
                        .vortex_expect("slice_canonical_array should succeed in fuzz test");

                    (
                        Action::Slice(start..stop),
                        ExpectedValue::Array(current_array.clone()),
                    )
                }
                ActionType::Take => {
                    if current_array.is_empty() {
                        return Err(EmptyChoose);
                    }

                    let indices = random_vec_in_range(u, 0, current_array.len() - 1)?;
                    let nullable = indices.contains(&None);

                    current_array = take_canonical_array(&current_array, &indices, &mut ctx)
                        .vortex_expect("take_canonical_array should succeed in fuzz test");
                    let indices_array = if nullable {
                        PrimitiveArray::from_option_iter(
                            indices.iter().map(|i| i.map(|i| i as u64)),
                        )
                        .into_array()
                    } else {
                        PrimitiveArray::from_iter(
                            indices
                                .iter()
                                .map(|i| i.vortex_expect("must be present"))
                                .map(|i| i as u64),
                        )
                        .into_array()
                    };

                    let compressed = BtrBlocksCompressor::default()
                        .compress(&indices_array, &mut ctx)
                        .vortex_expect("BtrBlocksCompressor compress should succeed in fuzz test");
                    (
                        Action::Take(compressed),
                        ExpectedValue::Array(current_array.clone()),
                    )
                }
                ActionType::SearchSorted => {
                    if current_array.dtype().is_struct() {
                        return Err(EmptyChoose);
                    }

                    let scalar = if u.arbitrary()? {
                        current_array
                            .execute_scalar(u.choose_index(current_array.len())?, &mut ctx)
                            .vortex_expect("scalar_at")
                    } else {
                        random_scalar(u, current_array.dtype())?
                    };

                    if scalar.is_null() {
                        return Err(EmptyChoose);
                    }

                    let sorted = sort_canonical_array(&current_array, &mut ctx)
                        .vortex_expect("sort_canonical_array should succeed in fuzz test");

                    let side = if u.arbitrary()? {
                        SearchSortedSide::Left
                    } else {
                        SearchSortedSide::Right
                    };
                    (
                        Action::SearchSorted(scalar.clone(), side),
                        ExpectedValue::Search(
                            search_sorted_canonical_array(&sorted, &scalar, side, &mut ctx)
                                .vortex_expect(
                                    "search_sorted_canonical_array should succeed in fuzz test",
                                ),
                        ),
                    )
                }
                ActionType::Filter => {
                    let mask = (0..current_array.len())
                        .map(|_| bool::arbitrary(u))
                        .collect::<arbitrary::Result<Vec<_>>>()?;
                    current_array = filter_canonical_array(&current_array, &mask, &mut ctx)
                        .vortex_expect("filter_canonical_array should succeed in fuzz test");
                    (
                        Action::Filter(Mask::from_iter(mask)),
                        ExpectedValue::Array(current_array.clone()),
                    )
                }
                ActionType::Compare => {
                    let scalar = if u.arbitrary()? {
                        current_array
                            .execute_scalar(u.choose_index(current_array.len())?, &mut ctx)
                            .vortex_expect("scalar_at")
                    } else {
                        // We can compare arrays with different nullability
                        let null: Nullability = u.arbitrary()?;
                        random_scalar(u, &current_array.dtype().union_nullability(null))?
                    };

                    let op = u.arbitrary()?;
                    current_array = compare_canonical_array(&current_array, &scalar, op, &mut ctx);
                    (
                        Action::Compare(scalar, op),
                        ExpectedValue::Array(current_array.clone()),
                    )
                }
                ActionType::Cast => {
                    let to: DType = u.arbitrary()?;
                    if Some(CastOutcome::Infallible) == allowed_casting(current_array.dtype(), &to)
                    {
                        return Err(EmptyChoose);
                    }
                    let Some(result) = cast_canonical_array(&current_array, &to, &mut ctx)
                        .vortex_expect("should fail to create array")
                    else {
                        return Err(EmptyChoose);
                    };

                    (Action::Cast(to), ExpectedValue::Array(result))
                }
                ActionType::Sum => {
                    // Do not try to fuzz float operations, they have unpredictable error behavior
                    if current_array.dtype().is_float() {
                        return Err(EmptyChoose);
                    }

                    // Sum - returns a scalar, does NOT update current_array (terminal operation)
                    let current_array_canonical = current_array
                        .clone()
                        .execute::<Canonical>(&mut ctx)
                        .vortex_expect("execute canonical should succeed in fuzz test");
                    let sum_result = sum_canonical_array(current_array_canonical, &mut ctx)
                        .vortex_expect("sum_canonical_array should succeed in fuzz test");
                    (Action::Sum, ExpectedValue::Scalar(sum_result))
                }
                ActionType::MinMax => {
                    // MinMax - returns a scalar, does NOT update current_array (terminal operation)
                    let current_array_canonical = current_array
                        .clone()
                        .execute::<Canonical>(&mut ctx)
                        .vortex_expect("execute canonical should succeed in fuzz test");
                    let min_max_result = min_max_canonical_array(current_array_canonical, &mut ctx)
                        .vortex_expect("min_max_canonical_array should succeed in fuzz test");
                    (Action::MinMax, ExpectedValue::MinMax(min_max_result))
                }
                ActionType::FillNull => {
                    // FillNull - returns an array, updates current_array
                    if !current_array.dtype().nullability().is_nullable() {
                        return Err(EmptyChoose);
                    }
                    let fill_value = if u.arbitrary()? && !current_array.is_empty() {
                        current_array
                            .execute_scalar(u.choose_index(current_array.len())?, &mut ctx)
                            .vortex_expect("scalar_at")
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
                    let current_array_canonical = current_array
                        .clone()
                        .execute::<Canonical>(&mut ctx)
                        .vortex_expect("execute canonical should succeed in fuzz test");
                    let expected_result =
                        fill_null_canonical_array(current_array_canonical, &fill_value, &mut ctx)
                            .vortex_expect("fill_null_canonical_array should succeed in fuzz test");
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
                        .collect::<arbitrary::Result<Vec<_>>>()?;

                    // Compute expected result on canonical form
                    let current_array_canonical = current_array
                        .clone()
                        .execute::<Canonical>(&mut ctx)
                        .vortex_expect("execute canonical should succeed in fuzz test");
                    let expected_result = mask_canonical_array(
                        current_array_canonical,
                        &Mask::from_iter(mask.clone()),
                        &mut ctx,
                    )
                    .vortex_expect("mask_canonical_array should succeed in fuzz test");
                    // Update current_array to the result for chaining
                    current_array = expected_result.clone();
                    (
                        Action::Mask(Mask::from_iter(mask)),
                        ExpectedValue::Array(expected_result),
                    )
                }
                ActionType::ScalarAt => {
                    if current_array.is_empty() {
                        return Err(EmptyChoose);
                    }

                    let num_indices = u.int_in_range(1..=5.min(current_array.len()))?;
                    let indices_vec = (0..num_indices)
                        .map(|_| {
                            u.choose_index(current_array.len())
                                .ok()
                                .vortex_expect("cannot pick")
                        })
                        .unique()
                        .collect::<Vec<_>>();

                    // Compute expected scalars using the baseline implementation
                    let expected_scalars: Vec<Scalar> = indices_vec
                        .iter()
                        .map(|&idx| {
                            let canonical = current_array
                                .clone()
                                .execute::<Canonical>(&mut ctx)
                                .vortex_expect("execute canonical should succeed in fuzz test");
                            scalar_at_canonical_array(canonical, idx, &mut ctx).vortex_expect(
                                "scalar_at_canonical_array should succeed in fuzz test",
                            )
                        })
                        .collect();

                    (
                        Action::ScalarAt(indices_vec),
                        ExpectedValue::ScalarVec(expected_scalars),
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
            // Struct supports: Compress, Slice, Take, Filter, MinMax, Mask, ScalarAt
            // Does NOT support: SearchSorted (requires scalar comparison), Compare, Cast, Sum, FillNull
            let struct_actions = [Compress, Slice, Take, Filter, MinMax, Mask, ScalarAt];
            sdt.fields()
                .map(|child| actions_for_dtype(&child))
                .fold(struct_actions.into(), |acc, actions| {
                    acc.intersection(&actions).copied().collect()
                })
        }
        DType::List(..) | DType::FixedSizeList(..) => {
            // List supports: Compress, Slice, Take, Filter, MinMax, Mask, ScalarAt
            // Does NOT support: SearchSorted, Compare, Cast, Sum, FillNull
            [Compress, Slice, Take, Filter, MinMax, Mask, ScalarAt].into()
        }
        DType::Utf8(_) | DType::Binary(_) => {
            // Utf8/Binary supports everything except Sum and FillNull
            // Actions: Compress, Slice, Take, SearchSorted, Filter, Compare, Cast, MinMax, Mask, ScalarAt
            [
                Compress,
                Slice,
                Take,
                SearchSorted,
                Filter,
                Compare,
                Cast,
                MinMax,
                Mask,
                ScalarAt,
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
                ScalarAt,
            ]
            .into()
        }
        DType::Extension(_) => {
            // Extension types delegate to storage dtype, support most operations
            ActionType::iter().collect()
        }
        // Currently, no support at all
        DType::Variant(_) => unreachable!("Variant dtype shouldn't be fuzzed"),
    }
}

fn random_vec_in_range(
    u: &mut Unstructured<'_>,
    min: usize,
    max: usize,
) -> arbitrary::Result<Vec<Option<usize>>> {
    iter::from_fn(|| {
        u.arbitrary().unwrap_or(false).then(|| {
            if u.arbitrary()? {
                Ok(None)
            } else {
                Ok(Some(u.int_in_range(min..=max)?))
            }
        })
    })
    .collect::<arbitrary::Result<Vec<_>>>()
}

fn random_action_from_list(
    u: &mut Unstructured<'_>,
    actions: &[ActionType],
) -> arbitrary::Result<ActionType> {
    u.choose_iter(actions).copied()
}

/// Compress an array using the given strategy.
#[cfg(feature = "zstd")]
pub fn compress_array(array: &ArrayRef, strategy: CompressorStrategy) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    match strategy {
        CompressorStrategy::Default => BtrBlocksCompressor::default()
            .compress(array, &mut ctx)
            .vortex_expect("BtrBlocksCompressor compress should succeed in fuzz test"),
        CompressorStrategy::Compact => BtrBlocksCompressorBuilder::default()
            .with_compact()
            .build()
            .compress(array, &mut ctx)
            .vortex_expect("Compact compress should succeed in fuzz test"),
    }
}

/// Compress an array using the given strategy (only Default).
#[cfg(not(feature = "zstd"))]
pub fn compress_array(array: &ArrayRef, _strategy: CompressorStrategy) -> ArrayRef {
    BtrBlocksCompressor::default()
        .compress(array, &mut SESSION.create_execution_ctx())
        .vortex_expect("BtrBlocksCompressor compress should succeed in fuzz test")
}

/// Run a fuzz action and return whether to keep it in the corpus.
///
/// Returns:
/// - `Ok(true)` - keep in corpus
/// - `Ok(false)` - reject from corpus
/// - `Err(_)` - a bug was found
#[expect(clippy::result_large_err)]
pub fn run_fuzz_action(fuzz_action: FuzzArrayAction) -> VortexFuzzResult<bool> {
    let FuzzArrayAction { array, actions } = fuzz_action;
    let mut current_array = array;

    let mut ctx = SESSION.create_execution_ctx();

    debug!(
        "Initial array:\nTree:\n{}Values:\n{:#}",
        current_array.display_tree(),
        current_array.display_values()
    );

    for (i, (action, expected)) in actions.into_iter().enumerate() {
        debug!(id = i, action = ?action);
        match action {
            Action::Compress(strategy) => {
                let canonical = current_array
                    .clone()
                    .execute::<Canonical>(&mut ctx)
                    .vortex_expect("execute canonical should succeed in fuzz test");
                current_array = compress_array(&canonical.into_array(), strategy);
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::Slice(range) => {
                current_array = current_array
                    .slice(range)
                    .vortex_expect("slice operation should succeed in fuzz test");
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::Take(indices) => {
                if indices.is_empty() {
                    return Ok(false); // Reject
                }
                current_array = current_array
                    .take(indices)
                    .vortex_expect("take operation should succeed in fuzz test");
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::SearchSorted(s, side) => {
                let mut sorted = sort_canonical_array(&current_array, &mut ctx)
                    .vortex_expect("sort_canonical_array should succeed in fuzz test");

                if !current_array.is_canonical() {
                    sorted = compress_array(&sorted, CompressorStrategy::Default);
                }
                assert_search_sorted(sorted, s, side, expected.search(), i)?;
            }
            Action::Filter(mask_val) => {
                current_array = current_array
                    .filter(mask_val)
                    .vortex_expect("filter operation should succeed in fuzz test");
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::Compare(v, op) => {
                let compare_result = current_array
                    .binary(
                        ConstantArray::new(v.clone(), current_array.len()).into_array(),
                        Operator::from(op),
                    )
                    .vortex_expect("compare operation should succeed in fuzz test");
                if let Err(e) = assert_array_eq(&expected.array(), &compare_result, i) {
                    vortex_panic!(
                        "Failed to compare {}with {op} {v}\nError: {e}",
                        current_array.display_tree()
                    )
                }
                current_array = compare_result;
            }
            Action::Cast(to) => {
                let cast_result = current_array
                    .cast(to.clone())
                    .vortex_expect("cast operation should succeed in fuzz test");
                if let Err(e) = assert_array_eq(&expected.array(), &cast_result, i) {
                    vortex_panic!(
                        "Failed to cast {} to dtype {to}\nError: {e}",
                        current_array.display_tree()
                    )
                }
                current_array = cast_result;
            }
            Action::Sum => {
                let sum_result = sum(&current_array, &mut ctx)
                    .vortex_expect("sum operation should succeed in fuzz test");
                assert_scalar_eq(&expected.scalar(), &sum_result, i)?;
            }
            Action::MinMax => {
                let min_max_result = min_max(&current_array, &mut ctx)
                    .vortex_expect("min_max operation should succeed in fuzz test");
                assert_min_max_eq(expected.min_max().as_ref(), min_max_result.as_ref(), i)?;
            }
            Action::FillNull(fill_value) => {
                current_array = current_array
                    .fill_null(fill_value.clone())
                    .vortex_expect("fill_null operation should succeed in fuzz test");
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::Mask(mask_val) => {
                current_array = current_array
                    .mask(mask_val.into_array())
                    .vortex_expect("mask operation should succeed in fuzz test");
                assert_array_eq(&expected.array(), &current_array, i)?;
            }
            Action::ScalarAt(indices) => {
                let expected_scalars = expected.scalar_vec();
                for (j, &idx) in indices.iter().enumerate() {
                    let scalar = current_array
                        .execute_scalar(idx, &mut ctx)
                        .vortex_expect("scalar_at");
                    assert_scalar_eq(&expected_scalars[j], &scalar, i)?;
                }
            }
        }
    }
    Ok(true) // Keep in corpus
}

#[expect(clippy::result_large_err)]
fn assert_search_sorted(
    array: ArrayRef,
    s: Scalar,
    side: SearchSortedSide,
    expected: SearchResult,
    step: usize,
) -> VortexFuzzResult<()> {
    let search_result = array
        .search_sorted(&s, side)
        .map_err(|e| VortexFuzzError::VortexError(e, Backtrace::capture()))?;
    if search_result != expected {
        Err(VortexFuzzError::SearchSortedError(
            s,
            expected,
            array,
            side,
            search_result,
            step,
            Backtrace::capture(),
        ))
    } else {
        Ok(())
    }
}

/// Assert two arrays are equal.
#[expect(clippy::result_large_err)]
pub fn assert_array_eq(lhs: &ArrayRef, rhs: &ArrayRef, step: usize) -> VortexFuzzResult<()> {
    if lhs.dtype() != rhs.dtype() {
        return Err(VortexFuzzError::DTypeMismatch(
            lhs.clone(),
            rhs.clone(),
            step,
            Backtrace::capture(),
        ));
    }

    if lhs.len() != rhs.len() {
        return Err(VortexFuzzError::LengthMismatch(
            lhs.len(),
            rhs.len(),
            lhs.clone(),
            rhs.clone(),
            step,
            Backtrace::capture(),
        ));
    }
    let mut ctx = SESSION.create_execution_ctx();
    for idx in 0..lhs.len() {
        let l = lhs.execute_scalar(idx, &mut ctx).vortex_expect("scalar_at");
        let r = rhs.execute_scalar(idx, &mut ctx).vortex_expect("scalar_at");

        if l != r {
            return Err(VortexFuzzError::ArrayNotEqual(
                l,
                r,
                idx,
                lhs.clone(),
                rhs.clone(),
                step,
                Backtrace::capture(),
            ));
        }
    }
    Ok(())
}

/// Assert two scalars are equal.
#[expect(clippy::result_large_err)]
pub fn assert_scalar_eq(lhs: &Scalar, rhs: &Scalar, step: usize) -> VortexFuzzResult<()> {
    if lhs != rhs {
        return Err(VortexFuzzError::ScalarMismatch(
            lhs.clone(),
            rhs.clone(),
            step,
            Backtrace::capture(),
        ));
    }
    Ok(())
}

/// Assert two min/max results are equal.
#[expect(clippy::result_large_err)]
pub fn assert_min_max_eq(
    lhs: Option<&MinMaxResult>,
    rhs: Option<&MinMaxResult>,
    step: usize,
) -> VortexFuzzResult<()> {
    if lhs != rhs {
        return Err(VortexFuzzError::MinMaxMismatch(
            lhs.cloned(),
            rhs.cloned(),
            step,
            Backtrace::capture(),
        ));
    }
    Ok(())
}
