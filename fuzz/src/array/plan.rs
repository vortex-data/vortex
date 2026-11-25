// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A structured, shrinkable description of an `array_ops` case for ephyra.
//!
//! [`ArrayOpsPlan`] holds plain data only, so ephyra can shrink and mutate it field by field:
//! fewer actions, shorter index lists, fewer array bytes. [`ArrayOpsPlan::materialize`] turns it
//! into the same [`FuzzArrayAction`] the libFuzzer target runs, computing expected values with the
//! same canonical oracles. Parameters are reduced modulo the current array length, so every plan
//! materializes; an action that does not apply to the current dtype is skipped rather than
//! rejected, which keeps shrink candidates valid.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use ephyra::gens;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::arbitrary::random_scalar;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Action;
use super::ActionType;
use super::CompressorStrategy;
use super::ExpectedValue;
use super::FuzzArrayAction;
use super::actions_for_dtype;
use super::cast_canonical_array;
use super::compare_canonical_array;
use super::fill_null_canonical_array;
use super::filter_canonical_array;
use super::mask_canonical_array;
use super::min_max_canonical_array;
use super::scalar_at_canonical_array;
use super::search_sorted_canonical_array;
use super::slice_canonical_array;
use super::sort_canonical_array;
use super::sum_canonical_array;
use super::take_canonical_array;
use crate::FUZZ_ARRAY_MAX_LEN;
use crate::SESSION;

/// The initial array, as bytes for [`ArbitraryArray`], followed by the actions to apply to it.
#[derive(Clone, Debug, ephyra::Arbitrary)]
pub struct ArrayOpsPlan {
    /// The longest initial array, usually short so edge cases near empty arrays come up often.
    #[ephyra(gen = gens::frequency(vec![
        (3, gens::int(0u16..=16).boxed()),
        (1, gens::int(0u16..=u16::try_from(FUZZ_ARRAY_MAX_LEN).unwrap_or(u16::MAX)).boxed()),
    ]))]
    pub max_len: u16,
    /// A primitive dtype for the initial array, as a [`PTYPES`] index and nullability; `None`
    /// lets the bytes pick any dtype. Only primitive arrays reach the cast oracle, and few random
    /// dtypes are primitive.
    #[ephyra(gen = gens::option((ephyra::any::<u8>(), gens::bool())))]
    pub primitive: Option<(u8, bool)>,
    /// Bytes decoded into the initial array. Fewer bytes decode to a shorter, simpler array.
    #[ephyra(gen = gens::vec(gens::int(0u8..=255)).len(0..=4096))]
    pub array_bytes: Vec<u8>,
    /// Actions applied in order, like the libFuzzer target's at most four.
    #[ephyra(gen = gens::vec(ephyra::any::<ActionPlan>()).len(1..=4))]
    pub actions: Vec<ActionPlan>,
}

/// Where a scalar parameter comes from.
#[derive(Clone, Debug, ephyra::Arbitrary)]
pub enum ScalarPlan {
    /// The value at this position (modulo the length) of the current array.
    FromArray(u16),
    /// A value decoded from these bytes by `random_scalar`.
    Random(#[ephyra(gen = gens::vec(gens::int(0u8..=255)).len(0..=32))] Vec<u8>),
}

/// One action with its parameters, in the order of the simplest first.
#[derive(Clone, Debug, ephyra::Arbitrary)]
pub enum ActionPlan {
    Sum,
    MinMax,
    Compress {
        compact: bool,
    },
    Slice {
        start: u16,
        len: u16,
    },
    ScalarAt {
        #[ephyra(gen = gens::vec(ephyra::any::<u16>()).len(1..=5))]
        indices: Vec<u16>,
    },
    Filter {
        /// Repeated to the array length; empty keeps nothing.
        #[ephyra(gen = gens::vec(gens::bool()).len(0..=64))]
        pattern: Vec<bool>,
    },
    Mask {
        /// Repeated to the array length; empty masks nothing.
        #[ephyra(gen = gens::vec(gens::bool()).len(0..=64))]
        pattern: Vec<bool>,
    },
    Take {
        /// Reduced modulo the array length; `None` is a null index.
        #[ephyra(gen = gens::vec(gens::option(ephyra::any::<u16>())).len(1..=64))]
        indices: Vec<Option<u16>>,
    },
    SearchSorted {
        value: ScalarPlan,
        left: bool,
    },
    Compare {
        value: ScalarPlan,
        /// Compare against a nullable scalar when the value is random.
        nullable: bool,
        op: u8,
    },
    FillNull {
        value: ScalarPlan,
    },
    /// Cast to a primitive type; only primitive casts have an oracle.
    Cast {
        /// Index into [`PTYPES`], reduced modulo its length.
        ptype: u8,
        nullable: bool,
    },
}

impl ActionPlan {
    fn action_type(&self) -> ActionType {
        match self {
            ActionPlan::Sum => ActionType::Sum,
            ActionPlan::MinMax => ActionType::MinMax,
            ActionPlan::Compress { .. } => ActionType::Compress,
            ActionPlan::Slice { .. } => ActionType::Slice,
            ActionPlan::ScalarAt { .. } => ActionType::ScalarAt,
            ActionPlan::Filter { .. } => ActionType::Filter,
            ActionPlan::Mask { .. } => ActionType::Mask,
            ActionPlan::Take { .. } => ActionType::Take,
            ActionPlan::SearchSorted { .. } => ActionType::SearchSorted,
            ActionPlan::Compare { .. } => ActionType::Compare,
            ActionPlan::FillNull { .. } => ActionType::FillNull,
            ActionPlan::Cast { .. } => ActionType::Cast,
        }
    }
}

impl ArrayOpsPlan {
    /// Decode the initial array and compute every action's expected value with the canonical
    /// oracles.
    ///
    /// Returns `None` when the bytes do not decode to an array or no action applies, so the
    /// property can discard the case. The oracles panic on internal errors, as in the libFuzzer
    /// target.
    pub fn materialize(&self) -> Option<FuzzArrayAction> {
        let array = ArbitraryArray::arbitrary_with_config(
            &mut Unstructured::new(&self.array_bytes),
            &ArbitraryArrayConfig {
                dtype: self
                    .primitive
                    .map(|(ptype, nullable)| primitive_dtype(ptype, nullable)),
                len: 0..=usize::from(self.max_len).min(FUZZ_ARRAY_MAX_LEN),
            },
        )
        .ok()?
        .0;

        let mut ctx = SESSION.create_execution_ctx();
        let valid_actions = actions_for_dtype(array.dtype());
        let mut current = array.clone();
        let mut actions: Vec<(Action, ExpectedValue)> = Vec::new();

        for plan in &self.actions {
            if !valid_actions.contains(&plan.action_type()) {
                continue;
            }
            let previous_was_compress = actions
                .last()
                .is_some_and(|(action, _)| matches!(action, Action::Compress(_)));
            if matches!(plan, ActionPlan::Compress { .. }) && previous_was_compress {
                continue;
            }
            if let Some(step) = plan_step(plan, &mut current, &mut ctx)
                .vortex_expect("oracle should succeed in fuzz test")
            {
                actions.push(step);
            }
        }

        (!actions.is_empty()).then_some(FuzzArrayAction { array, actions })
    }
}

/// Cast targets, from which the oracle supports infallible integer and float widening.
const PTYPES: [PType; 11] = [
    PType::U8,
    PType::U16,
    PType::U32,
    PType::U64,
    PType::I8,
    PType::I16,
    PType::I32,
    PType::I64,
    PType::F16,
    PType::F32,
    PType::F64,
];

fn primitive_dtype(ptype: u8, nullable: bool) -> DType {
    DType::Primitive(
        PTYPES[usize::from(ptype) % PTYPES.len()],
        Nullability::from(nullable),
    )
}

/// Repeat `pattern` to `len` booleans; an empty pattern gives all `false`.
fn repeat_pattern(pattern: &[bool], len: usize) -> Vec<bool> {
    if pattern.is_empty() {
        return vec![false; len];
    }
    pattern.iter().copied().cycle().take(len).collect()
}

/// The scalar a [`ScalarPlan`] names in the context of `current`, if it has one.
fn resolve_scalar(
    plan: &ScalarPlan,
    current: &ArrayRef,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Scalar>> {
    match plan {
        ScalarPlan::FromArray(_) if current.is_empty() => Ok(None),
        ScalarPlan::FromArray(index) => current
            .execute_scalar(usize::from(*index) % current.len(), ctx)
            .map(Some),
        ScalarPlan::Random(bytes) => Ok(random_scalar(&mut Unstructured::new(bytes), dtype).ok()),
    }
}

/// The action and expected value for one plan step, advancing `current` for chaining actions.
///
/// `Ok(None)` skips a step whose parameters do not apply, the cases where the libFuzzer generator
/// rejects its input.
fn plan_step(
    plan: &ActionPlan,
    current: &mut ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<(Action, ExpectedValue)>> {
    let len = current.len();
    Ok(Some(match plan {
        ActionPlan::Compress { compact } => {
            let strategy = if *compact {
                compact_strategy()
            } else {
                CompressorStrategy::Default
            };
            (
                Action::Compress(strategy),
                ExpectedValue::Array(current.clone()),
            )
        }
        ActionPlan::Slice {
            start,
            len: slice_len,
        } => {
            let start = usize::from(*start) % (len + 1);
            let stop = start + usize::from(*slice_len) % (len - start + 1);
            *current = slice_canonical_array(current, start, stop, ctx)?;
            (
                Action::Slice(start..stop),
                ExpectedValue::Array(current.clone()),
            )
        }
        ActionPlan::Take { indices } => {
            if len == 0 {
                return Ok(None);
            }
            let indices = indices
                .iter()
                .map(|i| i.map(|i| usize::from(i) % len))
                .collect::<Vec<_>>();
            *current = take_canonical_array(current, &indices, ctx)?;
            let indices_array = if indices.contains(&None) {
                PrimitiveArray::from_option_iter(indices.iter().map(|i| i.map(|i| i as u64)))
                    .into_array()
            } else {
                PrimitiveArray::from_iter(indices.iter().flatten().map(|&i| i as u64)).into_array()
            };
            let compressed =
                BtrBlocksCompressor::from_session(&SESSION).compress(&indices_array, ctx)?;
            (
                Action::Take(compressed),
                ExpectedValue::Array(current.clone()),
            )
        }
        ActionPlan::SearchSorted { value, left } => {
            if current.dtype().is_struct() {
                return Ok(None);
            }
            let Some(scalar) = resolve_scalar(value, current, &current.dtype().clone(), ctx)?
            else {
                return Ok(None);
            };
            if scalar.is_null() {
                return Ok(None);
            }
            let side = if *left {
                SearchSortedSide::Left
            } else {
                SearchSortedSide::Right
            };
            let sorted = sort_canonical_array(current, ctx)?;
            let expected = search_sorted_canonical_array(&sorted, &scalar, side, ctx)?;
            (
                Action::SearchSorted(scalar, side),
                ExpectedValue::Search(expected),
            )
        }
        ActionPlan::Filter { pattern } => {
            let mask = repeat_pattern(pattern, len);
            *current = filter_canonical_array(current, &mask, ctx)?;
            (
                Action::Filter(Mask::from_iter(mask)),
                ExpectedValue::Array(current.clone()),
            )
        }
        ActionPlan::Compare {
            value,
            nullable,
            op,
        } => {
            let nullability = if *nullable {
                Nullability::Nullable
            } else {
                Nullability::NonNullable
            };
            let dtype = current.dtype().union_nullability(nullability);
            let Some(scalar) = resolve_scalar(value, current, &dtype, ctx)? else {
                return Ok(None);
            };
            let Ok(op) = CompareOperator::arbitrary(&mut Unstructured::new(&[*op])) else {
                return Ok(None);
            };
            *current = compare_canonical_array(current, &scalar, op, ctx);
            (
                Action::Compare(scalar, op),
                ExpectedValue::Array(current.clone()),
            )
        }
        ActionPlan::Cast { ptype, nullable } => {
            let to = primitive_dtype(*ptype, *nullable);
            let Some(result) = cast_canonical_array(current, &to, ctx)? else {
                return Ok(None);
            };
            *current = result.clone();
            (Action::Cast(to), ExpectedValue::Array(result))
        }
        ActionPlan::Sum => {
            // Float sums have unpredictable rounding, as in the libFuzzer target.
            if current.dtype().is_float() {
                return Ok(None);
            }
            let Some(sum) = sum_canonical_array(current, ctx)? else {
                return Ok(None);
            };
            (Action::Sum, ExpectedValue::Scalar(sum))
        }
        ActionPlan::MinMax => {
            let canonical = current.clone().execute::<Canonical>(ctx)?;
            (
                Action::MinMax,
                ExpectedValue::MinMax(min_max_canonical_array(canonical, ctx)?),
            )
        }
        ActionPlan::FillNull { value } => {
            if !current.dtype().nullability().is_nullable() {
                return Ok(None);
            }
            let dtype = current.dtype().with_nullability(Nullability::NonNullable);
            let Some(fill) = resolve_scalar(value, current, &dtype, ctx)? else {
                return Ok(None);
            };
            if fill.is_null() {
                return Ok(None);
            }
            let canonical = current.clone().execute::<Canonical>(ctx)?;
            *current = fill_null_canonical_array(canonical, &fill, ctx)?;
            (
                Action::FillNull(fill),
                ExpectedValue::Array(current.clone()),
            )
        }
        ActionPlan::Mask { pattern } => {
            let mask = Mask::from_iter(repeat_pattern(pattern, len));
            let canonical = current.clone().execute::<Canonical>(ctx)?;
            *current = mask_canonical_array(canonical, &mask, ctx)?;
            (Action::Mask(mask), ExpectedValue::Array(current.clone()))
        }
        ActionPlan::ScalarAt { indices } => {
            if len == 0 {
                return Ok(None);
            }
            let indices = indices
                .iter()
                .map(|&i| usize::from(i) % len)
                .unique()
                .collect::<Vec<_>>();
            let expected = indices
                .iter()
                .map(|&idx| {
                    let canonical = current.clone().execute::<Canonical>(ctx)?;
                    scalar_at_canonical_array(canonical, idx, ctx)
                })
                .collect::<VortexResult<Vec<_>>>()?;
            (
                Action::ScalarAt(indices),
                ExpectedValue::ScalarVec(expected),
            )
        }
    }))
}

#[cfg(feature = "zstd")]
fn compact_strategy() -> CompressorStrategy {
    CompressorStrategy::Compact
}

#[cfg(not(feature = "zstd"))]
fn compact_strategy() -> CompressorStrategy {
    CompressorStrategy::Default
}
