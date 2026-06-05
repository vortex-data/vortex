// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`MergeN`] encoding: a lazy, order-preserving interleave of `N` compact branches into one
//! array, routed by a selector.
//!
//! # Specification
//!
//! A [`MergeN`] array has `N + 1` children: `N` *branches* followed by a *selector*. The output
//! has `selector.len()` rows, and output row `i` comes from `branches[selector[i]]`. Each branch
//! is consumed in order: the rows routed to a branch appear in the output in the same order they
//! appear in that branch. Equivalently, [`MergeN`] interleaves the branches back together in the
//! order dictated by the selector.
//!
//! It is the inverse (the "mux") of partitioning rows into branches by a predicate: split rows
//! into per-branch compact runs, process each branch, then [`MergeN`] re-assembles them in the
//! original order using the selector that recorded each row's branch.
//!
//! The branches are **compact**: each holds only the rows it owns, so they are generally shorter
//! than the output. This distinguishes [`MergeN`] from an element-wise select such as `zip`,
//! whose arguments are all full-length.
//!
//! ## Invariants
//!
//! - The selector is **non-nullable** and `selector[i] < branches.len()` for every `i`. It only
//!   records *where* each output row goes, which is always a definite decision. Predicate
//!   nullability (e.g. a SQL `WHEN` that evaluates to null) must be resolved into a definite
//!   branch by the caller *before* the merge is built — exactly as `case_when` does with
//!   `to_mask_fill_null_false`.
//! - `count(selector == j) == branches[j].len()` for every branch `j`.
//! - All branches share a logical type up to nullability. The output type is that shared type
//!   with the union of the branches' nullabilities. This is orthogonal to the selector: a row's
//!   *value* may be null even though its branch assignment is definite.
//! - The output length equals `selector.len()`.
//!
//! ## Boolean selector
//!
//! When there are exactly two branches the selector is a non-nullable boolean array, interpreted
//! as an index: `false` selects `branches[0]` and `true` selects `branches[1]`. This is the only
//! case implemented today; see [`MergeNArray::try_new`].

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArraySlots;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::TypedArrayRef;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::bool::BoolArrayExt;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::executor::ExecutionResult;
use crate::require_child;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`MergeN`]-encoded Vortex array. See the [module docs](self) for the specification.
pub type MergeNArray = Array<MergeN>;

/// The [`MergeN`] encoding. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct MergeN;

/// Per-array metadata for a [`MergeNArray`].
///
/// The branches and selector live in the array's slots; only the branch count is stored here so
/// the selector slot can be located (`slots[num_branches]`).
#[derive(Clone, Debug)]
pub struct MergeData {
    pub(crate) num_branches: usize,
}

impl Display for MergeData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "num_branches: {}", self.num_branches)
    }
}

impl ArrayHash for MergeData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        state.write_usize(self.num_branches);
    }
}

impl ArrayEq for MergeData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.num_branches == other.num_branches
    }
}

/// Accessors for the branches and selector of a [`MergeNArray`].
pub trait MergeNArrayExt: TypedArrayRef<MergeN> {
    /// The number of branches (one fewer than the number of children).
    fn num_branches(&self) -> usize {
        self.num_branches
    }

    /// The `idx`-th branch (a compact array holding the rows routed to branch `idx`).
    fn branch(&self, idx: usize) -> &ArrayRef {
        self.as_ref().slots()[idx]
            .as_ref()
            .vortex_expect("validated merge branch slot")
    }

    /// The selector array routing each output row to a branch.
    fn selector(&self) -> &ArrayRef {
        self.as_ref().slots()[self.num_branches]
            .as_ref()
            .vortex_expect("validated merge selector slot")
    }
}
impl<T: TypedArrayRef<MergeN>> MergeNArrayExt for T {}

impl Array<MergeN> {
    /// Constructs a new [`MergeNArray`] from `branches` and a `selector`.
    ///
    /// See the [module docs](self) for the full specification and invariants.
    ///
    /// Only the two-branch, boolean-selector case is implemented today: `false` routes to
    /// `branches[0]` and `true` routes to `branches[1]`. Other shapes are rejected.
    ///
    /// The selector must be non-nullable: it records a definite branch per row. Null-predicate
    /// handling (a null condition falling through to "not matched") is the caller's
    /// responsibility, resolved before the merge is constructed.
    pub fn try_new(branches: Vec<ArrayRef>, selector: ArrayRef) -> VortexResult<Self> {
        // TODO(joe): extend MergeN to more than two branches with an integer selector.
        if branches.len() != 2 {
            vortex_bail!(
                "merge_n currently supports exactly 2 branches, got {} (todo: extend this)",
                branches.len()
            );
        }
        if !selector.dtype().is_boolean() {
            vortex_bail!(
                "merge_n currently requires a boolean selector, got {} (todo: extend this)",
                selector.dtype()
            );
        }
        if selector.dtype().is_nullable() {
            vortex_bail!(
                "merge_n requires a non-nullable selector; resolve predicate nullability (e.g. via \
                 to_mask_fill_null_false) before constructing the merge, got {}",
                selector.dtype()
            );
        }

        let base_dtype = branches[0].dtype();
        let mut nullability = Nullability::NonNullable;
        let mut total = 0;
        for branch in &branches {
            vortex_ensure!(
                branch.dtype().eq_ignore_nullability(base_dtype),
                "merge_n branches must share a dtype up to nullability: {} vs {}",
                base_dtype,
                branch.dtype()
            );
            nullability |= branch.dtype().nullability();
            total += branch.len();
        }
        vortex_ensure!(
            total == selector.len(),
            "merge_n branch lengths sum to {} but selector has length {}",
            total,
            selector.len()
        );

        let dtype = base_dtype.with_nullability(nullability);
        let len = selector.len();
        let num_branches = branches.len();

        let mut slots: ArraySlots = branches.into_iter().map(Some).collect();
        slots.push(Some(selector));

        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(MergeN, dtype, len, MergeData { num_branches }).with_slots(slots),
            )
        })
    }
}

impl VTable for MergeN {
    type TypedArrayData = MergeData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.merge_n");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots.len() == data.num_branches + 1,
            "MergeNArray expected {} slots (branches + selector), got {}",
            data.num_branches + 1,
            slots.len()
        );
        vortex_ensure!(
            slots.iter().all(|s| s.is_some()),
            "MergeNArray slots must all be present"
        );

        let selector = slots[data.num_branches]
            .as_ref()
            .vortex_expect("validated selector slot");
        vortex_ensure!(
            selector.dtype() == &DType::Bool(Nullability::NonNullable),
            "MergeNArray selector must be a non-nullable boolean, got {}",
            selector.dtype()
        );
        vortex_ensure!(
            selector.len() == len,
            "MergeNArray selector length {} does not match outer length {}",
            selector.len(),
            len
        );

        let mut nullability = Nullability::NonNullable;
        let mut total = 0;
        for slot in &slots[..data.num_branches] {
            let branch = slot.as_ref().vortex_expect("validated branch slot");
            vortex_ensure!(
                branch.dtype().eq_ignore_nullability(dtype),
                "MergeNArray branch dtype {} does not match outer dtype {} up to nullability",
                branch.dtype(),
                dtype
            );
            nullability |= branch.dtype().nullability();
            total += branch.len();
        }
        vortex_ensure!(
            dtype.nullability() == nullability,
            "MergeNArray dtype nullability {} does not match the union of branch nullabilities {}",
            dtype.nullability(),
            nullability
        );
        vortex_ensure!(
            total == len,
            "MergeNArray branch lengths sum to {} but outer length is {}",
            total,
            len
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("MergeNArray has no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        if idx == array.num_branches() {
            "selector".to_string()
        } else {
            format!("branch_{idx}")
        }
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("MergeN array is not serializable")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("MergeN array is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        debug_assert_eq!(
            array.num_branches(),
            2,
            "MergeN::execute only supports the two-branch boolean-selector case"
        );

        // Drive the branches and selector to canonical `Bool` so we can operate on raw bits.
        let array = require_child!(array, array.branch(0), 0 => Bool);
        let array = require_child!(array, array.branch(1), 1 => Bool);
        let array = require_child!(array, array.selector(), 2 => Bool);

        let dtype = array.as_ref().dtype().clone();
        let len = array.as_ref().len();
        let nullable = dtype.is_nullable();

        let b0 = array.branch(0).as_::<Bool>();
        let b1 = array.branch(1).as_::<Bool>();
        // The selector is non-nullable (enforced at construction), so its bits are the routing
        // mask directly: `false` selects branch 0, `true` selects branch 1.
        let selector = Mask::from_buffer(array.selector().as_::<Bool>().to_bit_buffer());

        let b0_bits = b0.to_bit_buffer();
        let b1_bits = b1.to_bit_buffer();
        vortex_ensure!(
            selector.true_count() == b1_bits.len() && len - selector.true_count() == b0_bits.len(),
            "merge_n selector does not partition into the branch lengths"
        );

        // Branch validity is only materialized when the output can be null.
        let (v0, v1) = if nullable {
            (
                Some(b0.validity()?.execute_mask(b0_bits.len(), ctx)?),
                Some(b1.validity()?.execute_mask(b1_bits.len(), ctx)?),
            )
        } else {
            (None, None)
        };

        let mut values = BitBufferMut::new_unset(len);
        let mut validity = nullable.then(|| BitBufferMut::new_set(len));

        let (mut c0, mut c1) = (0usize, 0usize);
        for i in 0..len {
            // Scatter one bit (and its validity) from the selected branch's current cursor.
            let (bit, valid) = if selector.value(i) {
                let out = (b1_bits.value(c1), v1.as_ref().is_none_or(|m| m.value(c1)));
                c1 += 1;
                out
            } else {
                let out = (b0_bits.value(c0), v0.as_ref().is_none_or(|m| m.value(c0)));
                c0 += 1;
                out
            };
            if bit {
                values.set(i);
            }
            // `validity` is `Some` exactly when `nullable`, and `valid` is always true otherwise.
            if !valid {
                validity
                    .as_mut()
                    .vortex_expect("validity buffer present when nullable")
                    .unset(i);
            }
        }

        let validity = match validity {
            Some(bits) => Validity::from(bits.freeze()),
            None => Validity::NonNullable,
        };
        Ok(ExecutionResult::done(BoolArray::try_new(
            values.freeze(),
            validity,
        )?))
    }
}

impl OperationsVTable<MergeN> for MergeN {
    fn scalar_at(
        array: ArrayView<'_, MergeN>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Resolve the selector to its (non-nullable) routing mask and find the position of `index`
        // within its branch: the count of same-branch rows strictly before `index`.
        let selector = array.selector().clone().execute::<Mask>(ctx)?;
        let trues_before = selector.slice(0..index).true_count();
        let (branch_idx, cursor) = if selector.value(index) {
            (1, trues_before)
        } else {
            (0, index - trues_before)
        };

        let scalar = array.branch(branch_idx).execute_scalar(cursor, ctx)?;
        // The branch may be non-nullable while the merged output is nullable; align the dtype.
        Ok(if array.as_ref().dtype().is_nullable() {
            scalar.into_nullable()
        } else {
            scalar
        })
    }
}

impl ValidityVTable<MergeN> for MergeN {
    fn validity(array: ArrayView<'_, MergeN>) -> VortexResult<Validity> {
        if !array.as_ref().dtype().is_nullable() {
            return Ok(Validity::NonNullable);
        }
        // The output validity is itself a merge — by the same selector — of the branches'
        // validities, expressed as non-nullable boolean arrays. This bottoms out immediately
        // because the inner merge is non-nullable.
        let mut branch_validities: Vec<ArrayRef> = Vec::with_capacity(array.num_branches());
        for i in 0..array.num_branches() {
            branch_validities.push(branch_validity_array(array.branch(i))?);
        }
        let merged = MergeNArray::try_new(branch_validities, array.selector().clone())?;
        Ok(Validity::Array(merged.into_array()))
    }
}

/// Materializes a branch's validity as a non-nullable boolean array of the branch's length, where
/// `true` marks a valid (non-null) row.
fn branch_validity_array(branch: &ArrayRef) -> VortexResult<ArrayRef> {
    Ok(match branch.validity()? {
        Validity::NonNullable | Validity::AllValid => {
            ConstantArray::new(true, branch.len()).into_array()
        }
        Validity::AllInvalid => ConstantArray::new(false, branch.len()).into_array(),
        Validity::Array(array) => array,
    })
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::assert_arrays_eq;

    /// Reference (oracle) implementation of the merge spec, used only to validate the optimized
    /// [`MergeN::execute`] path. It is intentionally simple and slow: it pulls each output element
    /// one [`Scalar`] at a time via [`ArrayRef::execute_scalar`] and never touches raw bits.
    ///
    /// This is deliberately *not* wired into the array execution path — it exists purely as a
    /// trustworthy comparison point in tests.
    fn merge_reference(
        branches: &[ArrayRef],
        selector: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let len = selector.len();
        let nullable = branches.iter().any(|b| b.dtype().is_nullable());
        let mut cursors = vec![0usize; branches.len()];
        let mut values: Vec<Option<bool>> = Vec::with_capacity(len);

        for i in 0..len {
            // Non-nullable boolean selector: false => branch 0, true => branch 1.
            let j = selector
                .execute_scalar(i, ctx)?
                .as_bool()
                .value()
                .vortex_expect("selector is non-nullable") as usize;
            let cursor = cursors[j];
            cursors[j] += 1;
            values.push(branches[j].execute_scalar(cursor, ctx)?.as_bool().value());
        }

        Ok(if nullable {
            BoolArray::from_iter(values).into_array()
        } else {
            BoolArray::from_iter(
                values
                    .into_iter()
                    .map(|v| v.vortex_expect("non-nullable branch produced a null")),
            )
            .into_array()
        })
    }

    /// Splits a row-aligned `values`/`selector` pair into the two compact branches a
    /// [`MergeNArray`] expects.
    fn split(values: &[Option<bool>], selector: &[bool]) -> (Vec<Option<bool>>, Vec<Option<bool>>) {
        let mut b0 = Vec::new();
        let mut b1 = Vec::new();
        for (&value, &sel) in values.iter().zip(selector) {
            let target = if sel { &mut b1 } else { &mut b0 };
            target.push(value);
        }
        (b0, b1)
    }

    /// Asserts that the optimized execute path and the reference implementation agree, exercising
    /// `MergeNArray` construction, `execute`, `scalar_at`, and `validity` (via `assert_arrays_eq`).
    fn check(values: &[Option<bool>], selector: &[bool]) -> VortexResult<()> {
        let (b0, b1) = split(values, selector);
        let nullable = values.iter().any(Option::is_none);

        let to_branch = |vals: Vec<Option<bool>>| -> ArrayRef {
            if nullable {
                BoolArray::from_iter(vals).into_array()
            } else {
                BoolArray::from_iter(
                    vals.into_iter()
                        .map(|v| v.vortex_expect("non-nullable branch produced a null")),
                )
                .into_array()
            }
        };

        let branches = vec![to_branch(b0), to_branch(b1)];
        let selector_array = BoolArray::from_iter(selector.iter().copied()).into_array();

        let merged = MergeNArray::try_new(branches.clone(), selector_array.clone())?.into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let reference = merge_reference(&branches, &selector_array, &mut ctx)?;

        assert_arrays_eq!(merged, reference);
        Ok(())
    }

    #[test]
    fn merge_non_nullable_mixed() -> VortexResult<()> {
        // selector:  F     T     F      T     T
        // branch0:  true,        false
        // branch1:        false,       true, true
        // output:   true, false, false, true, true
        check(
            &[Some(true), Some(false), Some(false), Some(true), Some(true)],
            &[false, true, false, true, true],
        )
    }

    #[test]
    fn merge_all_false_selector() -> VortexResult<()> {
        check(
            &[Some(true), Some(false), Some(true)],
            &[false, false, false],
        )
    }

    #[test]
    fn merge_all_true_selector() -> VortexResult<()> {
        check(&[Some(false), Some(true), Some(false)], &[true, true, true])
    }

    #[test]
    fn merge_nullable_with_nulls_in_both_branches() -> VortexResult<()> {
        check(
            &[None, Some(true), None, Some(false), None, Some(true)],
            &[false, true, true, false, true, false],
        )
    }

    #[test]
    fn merge_empty() -> VortexResult<()> {
        check(&[], &[])
    }

    #[test]
    fn merge_single_element_each_branch() -> VortexResult<()> {
        check(&[Some(false), Some(true)], &[true, false])
    }

    #[test]
    fn rejects_more_than_two_branches() {
        let branch = BoolArray::from_iter([true]).into_array();
        let selector = BoolArray::from_iter([true, false, true]).into_array();
        let err = MergeNArray::try_new(vec![branch.clone(), branch.clone(), branch], selector)
            .err()
            .vortex_expect("expected merge_n to reject more than two branches");
        assert!(err.to_string().contains("todo"), "{err}");
    }

    #[test]
    fn rejects_non_boolean_selector() {
        let branch = BoolArray::from_iter([true, false]).into_array();
        let selector = crate::arrays::PrimitiveArray::from_iter([0u8, 1]).into_array();
        let err = MergeNArray::try_new(vec![branch.clone(), branch], selector)
            .err()
            .vortex_expect("expected merge_n to reject a non-boolean selector");
        assert!(err.to_string().contains("todo"), "{err}");
    }

    #[test]
    fn rejects_nullable_selector() {
        let branch = BoolArray::from_iter([true, false]).into_array();
        // A nullable boolean selector is ambiguous: predicate nullability must be resolved upstream.
        let selector = BoolArray::from_iter([Some(true), Some(false)]).into_array();
        let err = MergeNArray::try_new(vec![branch.clone(), branch], selector)
            .err()
            .vortex_expect("expected merge_n to reject a nullable selector");
        assert!(err.to_string().contains("non-nullable"), "{err}");
    }
}
