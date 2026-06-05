// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Merge`] encoding: a lazy, order-preserving interleave of `N` compact branches into one
//! array, routed by a selector.
//!
//! # Specification
//!
//! A [`Merge`] array has `N + 1` children: `N` *branches* followed by a *selector*. The output
//! has `selector.len()` rows, and output row `i` comes from `branches[selector[i]]`. Each branch
//! is consumed in order: the rows routed to a branch appear in the output in the same order they
//! appear in that branch. Equivalently, [`Merge`] interleaves the branches back together in the
//! order dictated by the selector.
//!
//! It is the inverse (the "mux") of partitioning rows into branches by a predicate: split rows
//! into per-branch compact runs, process each branch, then [`Merge`] re-assembles them in the
//! original order using the selector that recorded each row's branch.
//!
//! The branches are **compact**: each holds only the rows it owns, so they are generally shorter
//! than the output. This distinguishes [`Merge`] from an element-wise select such as `zip`,
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
//! ## Selector type
//!
//! The selector encodes the branch index per row: a non-nullable **boolean** for exactly two
//! branches (`false` → `branches[0]`, `true` → `branches[1]`), or a non-nullable **unsigned
//! integer** for two or more branches. Today only the boolean selector is wired into the
//! [execution path](execute); integer selectors construct but panic on execution.

mod execute;

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

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
use crate::arrays::ConstantArray;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::executor::ExecutionResult;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`Merge`]-encoded Vortex array. See the [module docs](self) for the specification.
pub type MergeArray = Array<Merge>;

/// The [`Merge`] encoding. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct Merge;

/// Per-array metadata for a [`MergeArray`].
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

/// Accessors for the branches and selector of a [`MergeArray`].
pub trait MergeArrayExt: TypedArrayRef<Merge> {
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
impl<T: TypedArrayRef<Merge>> MergeArrayExt for T {}

impl Merge {
    /// The single source of truth for [`MergeArray`] invariants.
    ///
    /// Validates `branches` and `selector` against the [specification](self) and returns the
    /// output [`DType`] (the shared branch type with the union of branch nullabilities). Both the
    /// public constructor and the [`VTable::validate`] hook funnel through here.
    fn check(branches: &[ArrayRef], selector: &ArrayRef) -> VortexResult<DType> {
        vortex_ensure!(
            branches.len() >= 2,
            "merge requires at least 2 branches, got {}",
            branches.len()
        );

        // The selector is non-nullable and indexes the branches: a boolean for exactly two
        // branches, or an unsigned integer for two or more.
        match selector.dtype() {
            DType::Bool(nullability) => {
                vortex_ensure!(
                    !nullability.is_nullable(),
                    "merge selector must be non-nullable, got {}",
                    selector.dtype()
                );
                vortex_ensure!(
                    branches.len() == 2,
                    "merge with a boolean selector requires exactly 2 branches, got {}",
                    branches.len()
                );
            }
            DType::Primitive(ptype, nullability) if ptype.is_unsigned_int() => {
                vortex_ensure!(
                    !nullability.is_nullable(),
                    "merge selector must be non-nullable, got {}",
                    selector.dtype()
                );
            }
            other => vortex_bail!(
                "merge selector must be a non-nullable boolean (exactly 2 branches) or unsigned \
                 integer (2 or more branches), got {}",
                other
            ),
        }

        let base_dtype = branches[0].dtype();
        let mut nullability = Nullability::NonNullable;
        let mut total = 0;
        for branch in branches {
            vortex_ensure!(
                branch.dtype().eq_ignore_nullability(base_dtype),
                "merge branches must share a dtype up to nullability: {} vs {}",
                base_dtype,
                branch.dtype()
            );
            nullability |= branch.dtype().nullability();
            total += branch.len();
        }
        vortex_ensure!(
            total == selector.len(),
            "merge branch lengths sum to {} but selector has length {}",
            total,
            selector.len()
        );

        Ok(base_dtype.with_nullability(nullability))
    }
}

impl Array<Merge> {
    /// Constructs a new [`MergeArray`] from `branches` and a `selector`.
    ///
    /// See the [module docs](self) for the full specification and invariants. The selector must be
    /// non-nullable: it records a definite branch per row, so null-predicate handling (a null
    /// condition falling through to "not matched") is the caller's responsibility, resolved before
    /// the merge is constructed.
    pub fn try_new(branches: Vec<ArrayRef>, selector: ArrayRef) -> VortexResult<Self> {
        let dtype = Merge::check(&branches, &selector)?;
        let len = selector.len();
        let num_branches = branches.len();

        let mut slots: ArraySlots = branches.into_iter().map(Some).collect();
        slots.push(Some(selector));

        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Merge, dtype, len, MergeData { num_branches }).with_slots(slots),
            )
        })
    }
}

impl VTable for Merge {
    type TypedArrayData = MergeData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.merge");
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
            "MergeArray expected {} slots (branches + selector), got {}",
            data.num_branches + 1,
            slots.len()
        );
        vortex_ensure!(
            slots.iter().all(|s| s.is_some()),
            "MergeArray slots must all be present"
        );

        let branches: Vec<ArrayRef> = slots[..data.num_branches]
            .iter()
            .map(|s| s.clone().vortex_expect("validated branch slot"))
            .collect();
        let selector = slots[data.num_branches]
            .clone()
            .vortex_expect("validated selector slot");

        // All semantic invariants live in `check`; here we only confirm the array's cached `dtype`
        // and `len` agree with what the children imply.
        let expected_dtype = Merge::check(&branches, &selector)?;
        vortex_ensure!(
            dtype == &expected_dtype,
            "MergeArray dtype {} does not match the dtype implied by its children {}",
            dtype,
            expected_dtype
        );
        vortex_ensure!(
            len == selector.len(),
            "MergeArray length {} does not match selector length {}",
            len,
            selector.len()
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("MergeArray has no buffers")
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
        vortex_bail!("Merge array is not serializable")
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
        vortex_bail!("Merge array is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        execute::execute(array, ctx)
    }
}

impl OperationsVTable<Merge> for Merge {
    fn scalar_at(
        array: ArrayView<'_, Merge>,
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

impl ValidityVTable<Merge> for Merge {
    fn validity(array: ArrayView<'_, Merge>) -> VortexResult<Validity> {
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
        let merged = MergeArray::try_new(branch_validities, array.selector().clone())?;
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
    use crate::Canonical;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;

    /// Reference (oracle) implementation of the merge spec, used only to validate the optimized
    /// [execute](super::execute) path. It is intentionally simple and slow: it pulls each output
    /// element one [`Scalar`] at a time via [`ArrayRef::execute_scalar`] and never touches raw
    /// bits.
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
    /// [`MergeArray`] expects.
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
    /// `MergeArray` construction, `execute`, `scalar_at`, and `validity` (via `assert_arrays_eq`).
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

        let merged = MergeArray::try_new(branches.clone(), selector_array.clone())?.into_array();

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
    fn rejects_boolean_selector_with_three_branches() {
        let branch = BoolArray::from_iter([true]).into_array();
        let selector = BoolArray::from_iter([true, false, true]).into_array();
        let err = MergeArray::try_new(vec![branch.clone(), branch.clone(), branch], selector)
            .err()
            .vortex_expect("expected merge to reject a boolean selector with three branches");
        assert!(err.to_string().contains("exactly 2 branches"), "{err}");
    }

    #[test]
    fn rejects_signed_integer_selector() {
        let branch = BoolArray::from_iter([true]).into_array();
        let selector = PrimitiveArray::from_iter([0i32, 1]).into_array();
        let err = MergeArray::try_new(vec![branch.clone(), branch], selector)
            .err()
            .vortex_expect("expected merge to reject a signed integer selector");
        assert!(err.to_string().contains("unsigned integer"), "{err}");
    }

    #[test]
    fn rejects_nullable_selector() {
        let branch = BoolArray::from_iter([true, false]).into_array();
        // A nullable boolean selector is ambiguous: predicate nullability must be resolved upstream.
        let selector = BoolArray::from_iter([Some(true), Some(false)]).into_array();
        let err = MergeArray::try_new(vec![branch.clone(), branch], selector)
            .err()
            .vortex_expect("expected merge to reject a nullable selector");
        assert!(err.to_string().contains("non-nullable"), "{err}");
    }

    #[test]
    fn accepts_unsigned_integer_selector() -> VortexResult<()> {
        // Construction is permitted for an unsigned-integer selector with 2+ branches, even though
        // the execute path does not yet handle it (see `primitive_selector_execution_panics`).
        let branch = BoolArray::from_iter([true]).into_array();
        let selector = PrimitiveArray::from_iter([0u32, 1, 2]).into_array();
        MergeArray::try_new(vec![branch.clone(), branch.clone(), branch], selector)?;
        Ok(())
    }

    #[test]
    #[should_panic(expected = "only implemented for boolean branches")]
    fn non_boolean_branch_execution_panics() {
        // Execution dispatches on the branch value type: primitive branches have no kernel yet.
        let b0 = PrimitiveArray::from_iter([1u32]).into_array();
        let b1 = PrimitiveArray::from_iter([2u32]).into_array();
        let selector = BoolArray::from_iter([true, false]).into_array();
        let merged = MergeArray::try_new(vec![b0, b1], selector)
            .vortex_expect("primitive branches with a boolean selector should construct")
            .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Dispatch panics before returning; the trailing `.ok()` just avoids an unused-result let.
        merged.execute::<Canonical>(&mut ctx).ok();
    }

    #[test]
    #[should_panic(expected = "boolean selector")]
    fn boolean_branches_with_integer_selector_unimplemented() {
        // Boolean branches dispatch to the bool kernel, which only handles a boolean selector.
        let branch = BoolArray::from_iter([true]).into_array();
        let selector = PrimitiveArray::from_iter([0u32, 1, 2]).into_array();
        let merged = MergeArray::try_new(vec![branch.clone(), branch.clone(), branch], selector)
            .vortex_expect("unsigned-integer selector should construct")
            .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        merged.execute::<Canonical>(&mut ctx).ok();
    }
}
