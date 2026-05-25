// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Experiment: model nullable primitives as "pure data + validity-in-wrapper".
//!
//! Rather than embedding validity inside [`PrimitiveArray`], we keep the primitive child free of
//! nulls (`Validity::NonNullable`) and carry definedness in the [`MaskedArray`] wrapper. These
//! tests pressure-test that model end-to-end: construction, validity readout, canonicalization, and
//! Arrow export with validity stitched in. They also assert that the values buffer survives Arrow
//! export without a copy, which is the crux of the "zero-copy export, stitch validity on the way
//! out" proposal.

use arrow_array::Array as ArrowArray;
use arrow_array::cast::AsArray as _;
use arrow_array::types::Int32Type;
use vortex_error::VortexResult;

use crate::arrays::Masked;
use crate::arrays::MaskedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::masked::array::MaskedArraySlotsExt as _;
use crate::arrow::ArrowSessionExt as _;
#[expect(deprecated)]
use crate::canonical::ToCanonical as _;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::optimizer::ArrayOptimizer as _;
use crate::validity::Validity;
use crate::{ArrayRef, IntoArray as _, LEGACY_SESSION, VortexSessionExecute as _};

/// A primitive child holding only data, with no embedded validity.
fn pure_data_child() -> ArrayRef {
    PrimitiveArray::new(
        vortex_buffer::buffer![1i32, 2, 3, 4, 5],
        Validity::NonNullable,
    )
    .into_array()
}

#[test]
fn pure_data_child_carries_no_validity() -> VortexResult<()> {
    let child = pure_data_child();
    // The child is genuinely null-free; nullability lives only in the wrapper.
    assert_eq!(
        child.dtype(),
        &DType::Primitive(PType::I32, Nullability::NonNullable)
    );

    let masked =
        MaskedArray::try_new(child, Validity::from_iter([true, false, true, false, true]))?;
    assert_eq!(masked.len(), 5);
    assert_eq!(
        masked.dtype(),
        &DType::Primitive(PType::I32, Nullability::Nullable)
    );
    Ok(())
}

#[test]
fn pure_data_primitive_exports_to_arrow_with_stitched_validity() -> VortexResult<()> {
    let masked = MaskedArray::try_new(
        pure_data_child(),
        Validity::from_iter([true, false, true, false, true]),
    )?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let arrow = LEGACY_SESSION
        .arrow()
        .execute_arrow(masked.into_array(), None, &mut ctx)?;
    let arrow = arrow.as_primitive::<Int32Type>();

    // Validity was stitched into the exported Arrow null buffer.
    assert_eq!(arrow.null_count(), 2);
    assert!(arrow.is_valid(0) && arrow.is_null(1) && arrow.is_valid(2));
    assert!(arrow.is_null(3) && arrow.is_valid(4));

    // Values are preserved at the valid positions.
    assert_eq!(arrow.value(0), 1);
    assert_eq!(arrow.value(2), 3);
    assert_eq!(arrow.value(4), 5);
    Ok(())
}

#[test]
fn pure_data_primitive_arrow_export_is_zero_copy() -> VortexResult<()> {
    let child = pure_data_child();
    #[expect(deprecated)]
    let src_ptr = child.to_primitive().as_slice::<i32>().as_ptr();

    let masked =
        MaskedArray::try_new(child, Validity::from_iter([true, false, true, false, true]))?;

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let arrow = LEGACY_SESSION
        .arrow()
        .execute_arrow(masked.into_array(), None, &mut ctx)?;
    let arrow = arrow.as_primitive::<Int32Type>();

    // The exported Arrow values buffer points at the same allocation as the Vortex child: only the
    // null buffer is newly attached, the data is shared.
    assert_eq!(arrow.values().as_ptr(), src_ptr);
    Ok(())
}

#[test]
fn nullable_primitive_optimizes_into_masked_wrapper() -> VortexResult<()> {
    // A primitive carrying a real validity buffer should have its definedness lifted out into a
    // MaskedArray, leaving the primitive child as pure NonNullable data.
    let nullable =
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]).into_array();
    assert!(matches!(nullable.validity()?, Validity::Array(_)));

    let optimized = nullable.optimize()?;

    let masked = optimized
        .as_opt::<Masked>()
        .expect("nullable primitive should optimize into a MaskedArray");
    assert!(matches!(masked.child().validity()?, Validity::NonNullable));
    Ok(())
}
