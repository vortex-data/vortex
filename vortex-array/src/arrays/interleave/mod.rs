// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Interleave`] encoding: a lazy, random-access gather of `N` value arrays into one array,
//! routed by a per-row `(array_index, row_index)` pair.
//!
//! # Specification
//!
//! An [`Interleave`] array has `N + 2` children: `N` *values* followed by an `array_indices`
//! selector and a `row_indices` selector. The output has `array_indices.len()` rows, and output
//! row `i` comes from `values[array_indices[i]][row_indices[i]]`.
//!
//! Unlike a `Merge`, which consumes each branch in order under a cursor, an [`Interleave`] is
//! **random-access**: `row_indices` names an explicit position within the selected value array, so
//! rows may be reordered, skipped, or repeated. A `Merge` is the special case where each value
//! array is consumed front-to-back exactly once.
//!
//! Like a `Merge`, the value arrays are independent: each holds only its own rows, and the
//! selectors stitch them back together. This distinguishes [`Interleave`] from an element-wise
//! select such as `zip`, whose arguments are all full-length.
//!
//! ## Invariants
//!
//! - Both selectors are **non-nullable** and equal in length, which is the output length. They
//!   record *where* each output row comes from, which is always a definite decision. Predicate
//!   nullability must be resolved into definite indices by the caller *before* the interleave is
//!   built.
//! - `array_indices[i] < values.len()` and `row_indices[i] < values[array_indices[i]].len()` for
//!   every `i`. These per-row bounds depend on the selector *values* and so are a runtime
//!   precondition of the caller, checked in the execution kernels rather than at construction.
//! - All values share a logical type up to nullability. The output type is that shared type with
//!   the union of the values' nullabilities. This is orthogonal to the selectors: a row's *value*
//!   may be null even though its `(array_index, row_index)` is definite.
//! - The output length equals `array_indices.len()` (`== row_indices.len()`).
//!
//! ## Selector types
//!
//! `array_indices` encodes the value array per row as a non-nullable **unsigned integer**
//! (`array_indices[i]` is the index into `values`). `row_indices` is likewise a non-nullable
//! **unsigned integer** naming the position within the selected value array.

mod execute;

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
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

/// An [`Interleave`]-encoded Vortex array. See the [module docs](self) for the specification.
pub type InterleaveArray = Array<Interleave>;

/// The [`Interleave`] encoding. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct Interleave;

/// Per-array metadata for an [`InterleaveArray`].
///
/// The values and selectors live in the array's slots; only the value count is stored here so the
/// selector slots can be located (`slots[num_values]` and `slots[num_values + 1]`).
#[derive(Clone, Debug)]
pub struct InterleaveData {
    pub(crate) num_values: usize,
}

impl Display for InterleaveData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "num_values: {}", self.num_values)
    }
}

impl ArrayHash for InterleaveData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        state.write_usize(self.num_values);
    }
}

impl ArrayEq for InterleaveData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.num_values == other.num_values
    }
}

/// Accessors for the values and selectors of an [`InterleaveArray`].
pub trait InterleaveArrayExt: TypedArrayRef<Interleave> {
    /// The number of value arrays (two fewer than the number of children).
    fn num_values(&self) -> usize {
        self.num_values
    }

    /// The `idx`-th value array (holding the rows that `array_indices` routes to it).
    fn value(&self, idx: usize) -> &ArrayRef {
        self.as_ref().slots()[idx]
            .as_ref()
            .vortex_expect("validated interleave value slot")
    }

    /// The selector routing each output row to a value array.
    fn array_indices(&self) -> &ArrayRef {
        self.as_ref().slots()[self.num_values]
            .as_ref()
            .vortex_expect("validated interleave array_indices slot")
    }

    /// The selector naming each output row's position within its value array.
    fn row_indices(&self) -> &ArrayRef {
        self.as_ref().slots()[self.num_values + 1]
            .as_ref()
            .vortex_expect("validated interleave row_indices slot")
    }
}
impl<T: TypedArrayRef<Interleave>> InterleaveArrayExt for T {}

impl Interleave {
    /// The single source of truth for [`InterleaveArray`] invariants.
    ///
    /// Validates `values`, `array_indices`, and `row_indices` against the [specification](self) and
    /// returns the output [`DType`] (the shared value type with the union of value nullabilities).
    /// Both the public constructor and the [`VTable::validate`] hook funnel through here.
    fn check(
        values: &[ArrayRef],
        array_indices: &ArrayRef,
        row_indices: &ArrayRef,
    ) -> VortexResult<DType> {
        vortex_ensure!(
            values.len() >= 2,
            "interleave requires at least 2 values, got {}",
            values.len()
        );

        // Both selectors are non-nullable unsigned integers: `array_indices` indexes the values and
        // `row_indices` names a position within the selected value.
        for (name, selector) in [
            ("array_indices", array_indices),
            ("row_indices", row_indices),
        ] {
            match selector.dtype() {
                DType::Primitive(ptype, nullability) if ptype.is_unsigned_int() => {
                    vortex_ensure!(
                        !nullability.is_nullable(),
                        "interleave {name} must be non-nullable, got {}",
                        selector.dtype()
                    );
                }
                other => vortex_bail!(
                    "interleave {name} must be a non-nullable unsigned integer, got {other}"
                ),
            }
        }

        vortex_ensure!(
            array_indices.len() == row_indices.len(),
            "interleave selectors must have equal length, got array_indices {} and row_indices {}",
            array_indices.len(),
            row_indices.len()
        );

        let base_dtype = values[0].dtype();
        let mut nullability = Nullability::NonNullable;
        for value in values {
            vortex_ensure!(
                value.dtype().eq_ignore_nullability(base_dtype),
                "interleave values must share a dtype up to nullability: {} vs {}",
                base_dtype,
                value.dtype()
            );
            nullability |= value.dtype().nullability();
        }

        Ok(base_dtype.with_nullability(nullability))
    }
}

impl Array<Interleave> {
    /// Constructs a new [`InterleaveArray`] from `values` and the `array_indices` / `row_indices`
    /// selectors.
    ///
    /// See the [module docs](self) for the full specification and invariants. The selectors must be
    /// non-nullable: they record a definite `(array_index, row_index)` per row, so null-predicate
    /// handling is the caller's responsibility, resolved before the interleave is constructed. The
    /// per-row bounds on the selector values are a runtime precondition checked during execution.
    pub fn try_new(
        values: Vec<ArrayRef>,
        array_indices: ArrayRef,
        row_indices: ArrayRef,
    ) -> VortexResult<Self> {
        let dtype = Interleave::check(&values, &array_indices, &row_indices)?;
        let len = array_indices.len();
        let num_values = values.len();

        let mut slots: ArraySlots = values.into_iter().map(Some).collect();
        slots.push(Some(array_indices));
        slots.push(Some(row_indices));

        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Interleave, dtype, len, InterleaveData { num_values })
                    .with_slots(slots),
            )
        })
    }
}

impl VTable for Interleave {
    type TypedArrayData = InterleaveData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.interleave");
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
            slots.len() == data.num_values + 2,
            "InterleaveArray expected {} slots (values + array_indices + row_indices), got {}",
            data.num_values + 2,
            slots.len()
        );
        vortex_ensure!(
            slots.iter().all(|s| s.is_some()),
            "InterleaveArray slots must all be present"
        );

        let values: Vec<ArrayRef> = slots[..data.num_values]
            .iter()
            .map(|s| s.clone().vortex_expect("validated value slot"))
            .collect();
        let array_indices = slots[data.num_values]
            .clone()
            .vortex_expect("validated array_indices slot");
        let row_indices = slots[data.num_values + 1]
            .clone()
            .vortex_expect("validated row_indices slot");

        // All semantic invariants live in `check`; here we only confirm the array's cached `dtype`
        // and `len` agree with what the children imply.
        let expected_dtype = Interleave::check(&values, &array_indices, &row_indices)?;
        vortex_ensure!(
            dtype == &expected_dtype,
            "InterleaveArray dtype {} does not match the dtype implied by its children {}",
            dtype,
            expected_dtype
        );
        vortex_ensure!(
            len == array_indices.len(),
            "InterleaveArray length {} does not match array_indices length {}",
            len,
            array_indices.len()
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("InterleaveArray has no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn slot_name(array: ArrayView<'_, Self>, idx: usize) -> String {
        if idx == array.num_values() {
            "array_indices".to_string()
        } else if idx == array.num_values() + 1 {
            "row_indices".to_string()
        } else {
            format!("value_{idx}")
        }
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("Interleave array is not serializable")
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
        vortex_bail!("Interleave array is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        execute::execute(array, ctx)
    }
}

impl OperationsVTable<Interleave> for Interleave {
    fn scalar_at(
        array: ArrayView<'_, Interleave>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Random-access gather: read the routing pair for `index` directly, then pull that row from
        // the selected value array. No cursor walk is required.
        let branch_idx = array
            .array_indices()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("interleave array_indices is non-nullable");
        let row = array
            .row_indices()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("interleave row_indices is non-nullable");

        let scalar = array.value(branch_idx).execute_scalar(row, ctx)?;
        // The value may be non-nullable while the interleaved output is nullable; align the dtype.
        Ok(if array.as_ref().dtype().is_nullable() {
            scalar.into_nullable()
        } else {
            scalar
        })
    }
}

impl ValidityVTable<Interleave> for Interleave {
    fn validity(array: ArrayView<'_, Interleave>) -> VortexResult<Validity> {
        if !array.as_ref().dtype().is_nullable() {
            return Ok(Validity::NonNullable);
        }
        // The output validity is itself an interleave — by the same selectors — of the values'
        // validities, expressed as non-nullable boolean arrays. This bottoms out immediately
        // because the inner interleave is non-nullable.
        let mut value_validities: Vec<ArrayRef> = Vec::with_capacity(array.num_values());
        for i in 0..array.num_values() {
            value_validities.push(value_validity_array(array.value(i))?);
        }
        let interleaved = InterleaveArray::try_new(
            value_validities,
            array.array_indices().clone(),
            array.row_indices().clone(),
        )?;
        Ok(Validity::Array(interleaved.into_array()))
    }
}

/// Materializes a value's validity as a non-nullable boolean array of the value's length, where
/// `true` marks a valid (non-null) row.
fn value_validity_array(value: &ArrayRef) -> VortexResult<ArrayRef> {
    Ok(match value.validity()? {
        Validity::NonNullable | Validity::AllValid => {
            ConstantArray::new(true, value.len()).into_array()
        }
        Validity::AllInvalid => ConstantArray::new(false, value.len()).into_array(),
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

    /// Reference (oracle) implementation of the interleave spec, used only to validate the optimized
    /// [execute](super::execute) path. It is intentionally simple and slow: it pulls each output
    /// element one [`Scalar`] at a time via [`ArrayRef::execute_scalar`] and never touches raw bits.
    ///
    /// This is deliberately *not* wired into the array execution path — it exists purely as a
    /// trustworthy comparison point in tests.
    fn interleave_reference(
        values: &[ArrayRef],
        array_indices: &ArrayRef,
        row_indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let len = array_indices.len();
        let nullable = values.iter().any(|v| v.dtype().is_nullable());
        let mut out: Vec<Option<bool>> = Vec::with_capacity(len);

        for i in 0..len {
            let j = array_indices
                .execute_scalar(i, ctx)?
                .as_primitive()
                .as_::<usize>()
                .vortex_expect("array_indices is non-nullable");
            let row = row_indices
                .execute_scalar(i, ctx)?
                .as_primitive()
                .as_::<usize>()
                .vortex_expect("row_indices is non-nullable");
            out.push(values[j].execute_scalar(row, ctx)?.as_bool().value());
        }

        Ok(if nullable {
            BoolArray::from_iter(out).into_array()
        } else {
            BoolArray::from_iter(
                out.into_iter()
                    .map(|v| v.vortex_expect("non-nullable value produced a null")),
            )
            .into_array()
        })
    }

    /// Builds the compact value arrays and the unsigned `(array_indices, row_indices)` selectors for
    /// a gather described by per-output `(array_index, row_index)` pairs over `branches`.
    fn build(
        branches: &[&[Option<bool>]],
        indices: &[(usize, usize)],
    ) -> (Vec<ArrayRef>, ArrayRef, ArrayRef) {
        let nullable = branches.iter().flat_map(|b| b.iter()).any(Option::is_none);
        let to_value = |vals: &[Option<bool>]| -> ArrayRef {
            if nullable {
                BoolArray::from_iter(vals.iter().copied()).into_array()
            } else {
                BoolArray::from_iter(
                    vals.iter()
                        .map(|v| v.vortex_expect("non-nullable value produced a null")),
                )
                .into_array()
            }
        };

        let values = branches.iter().map(|b| to_value(b)).collect();
        let array_indices = PrimitiveArray::from_iter(
            indices
                .iter()
                .map(|&(a, _)| u32::try_from(a).vortex_expect("array index fits in u32")),
        )
        .into_array();
        let row_indices = PrimitiveArray::from_iter(
            indices
                .iter()
                .map(|&(_, r)| u32::try_from(r).vortex_expect("row index fits in u32")),
        )
        .into_array();
        (values, array_indices, row_indices)
    }

    /// Asserts that the optimized execute path and the reference implementation agree, exercising
    /// `InterleaveArray` construction, `execute`, `scalar_at`, and `validity` (via
    /// `assert_arrays_eq`).
    fn check(branches: &[&[Option<bool>]], indices: &[(usize, usize)]) -> VortexResult<()> {
        let (values, array_indices, row_indices) = build(branches, indices);

        let interleaved =
            InterleaveArray::try_new(values.clone(), array_indices.clone(), row_indices.clone())?
                .into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let reference = interleave_reference(&values, &array_indices, &row_indices, &mut ctx)?;

        assert_arrays_eq!(interleaved, reference);
        Ok(())
    }

    #[test]
    fn interleave_reorders_and_repeats() -> VortexResult<()> {
        // Random access: rows are pulled out of order and branch 0 row 0 is repeated.
        check(
            &[&[Some(true), Some(false)], &[Some(false), Some(true)]],
            &[(0, 1), (1, 0), (0, 0), (1, 1), (0, 0)],
        )
    }

    #[test]
    fn interleave_skips_rows() -> VortexResult<()> {
        // Branch 0 row 1 and branch 1 row 0 are never gathered.
        check(
            &[
                &[Some(true), Some(false), Some(true)],
                &[Some(false), Some(true)],
            ],
            &[(0, 0), (1, 1), (0, 2)],
        )
    }

    #[test]
    fn interleave_three_values() -> VortexResult<()> {
        // An unsigned `array_indices` routes among three values with full random access.
        check(
            &[
                &[Some(true), Some(false)],
                &[Some(false)],
                &[Some(true), Some(true), Some(false)],
            ],
            &[(2, 1), (0, 0), (1, 0), (2, 2), (0, 1), (2, 0)],
        )
    }

    #[test]
    fn interleave_only_one_branch() -> VortexResult<()> {
        check(
            &[&[Some(true), Some(false), Some(true)], &[Some(false)]],
            &[(0, 2), (0, 0), (0, 1)],
        )
    }

    #[test]
    fn interleave_nullable_with_nulls_in_values() -> VortexResult<()> {
        check(
            &[&[None, Some(true), None], &[Some(false), None]],
            &[(1, 1), (0, 0), (1, 0), (0, 2), (0, 1)],
        )
    }

    #[test]
    fn interleave_empty() -> VortexResult<()> {
        check(&[&[Some(true)], &[Some(false)]], &[])
    }

    #[test]
    fn rejects_boolean_array_indices() {
        let value = BoolArray::from_iter([true, false]).into_array();
        let array_indices = BoolArray::from_iter([true, false]).into_array();
        let row_indices = PrimitiveArray::from_iter([0u32, 1]).into_array();
        let err = InterleaveArray::try_new(vec![value.clone(), value], array_indices, row_indices)
            .err()
            .vortex_expect("expected interleave to reject a boolean array_indices");
        assert!(err.to_string().contains("unsigned integer"), "{err}");
    }

    #[test]
    fn rejects_signed_integer_array_indices() {
        let value = BoolArray::from_iter([true]).into_array();
        let array_indices = PrimitiveArray::from_iter([0i32, 1]).into_array();
        let row_indices = PrimitiveArray::from_iter([0u32, 0]).into_array();
        let err = InterleaveArray::try_new(vec![value.clone(), value], array_indices, row_indices)
            .err()
            .vortex_expect("expected interleave to reject a signed integer array_indices");
        assert!(err.to_string().contains("unsigned integer"), "{err}");
    }

    #[test]
    fn rejects_nullable_row_indices() {
        let value = BoolArray::from_iter([true, false]).into_array();
        let array_indices = PrimitiveArray::from_iter([0u32, 1]).into_array();
        let row_indices = PrimitiveArray::from_option_iter([Some(0u32), Some(1)]).into_array();
        let err = InterleaveArray::try_new(vec![value.clone(), value], array_indices, row_indices)
            .err()
            .vortex_expect("expected interleave to reject nullable row_indices");
        assert!(err.to_string().contains("non-nullable"), "{err}");
    }

    #[test]
    fn rejects_mismatched_selector_lengths() {
        let value = BoolArray::from_iter([true, false]).into_array();
        let array_indices = PrimitiveArray::from_iter([0u32, 1]).into_array();
        let row_indices = PrimitiveArray::from_iter([0u32]).into_array();
        let err = InterleaveArray::try_new(vec![value.clone(), value], array_indices, row_indices)
            .err()
            .vortex_expect("expected interleave to reject mismatched selector lengths");
        assert!(err.to_string().contains("equal length"), "{err}");
    }

    #[test]
    #[should_panic(expected = "only implemented for boolean values")]
    fn non_boolean_value_execution_panics() {
        // Execution dispatches on the value type: primitive values have no kernel yet.
        let v0 = PrimitiveArray::from_iter([1u32]).into_array();
        let v1 = PrimitiveArray::from_iter([2u32]).into_array();
        let array_indices = PrimitiveArray::from_iter([0u32, 1]).into_array();
        let row_indices = PrimitiveArray::from_iter([0u32, 0]).into_array();
        let interleaved = InterleaveArray::try_new(vec![v0, v1], array_indices, row_indices)
            .vortex_expect("primitive values should construct")
            .into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        interleaved.execute::<Canonical>(&mut ctx).ok();
    }
}
