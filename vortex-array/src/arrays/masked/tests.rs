// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexResult;

use super::*;
use crate::Array;
use crate::IntoArray;
use crate::ToCanonical as _;
use crate::arrays::PrimitiveArray;
use crate::validity::Validity;

#[rstest]
#[case(Validity::AllValid, Nullability::Nullable)]
#[case(Validity::from_iter([true, false, true]), Nullability::Nullable)]
fn test_dtype_nullability(#[case] validity: Validity, #[case] expected: Nullability) {
    let child = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let array = MaskedArray::try_new(child, validity).unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Primitive(vortex_dtype::PType::I32, expected)
    );
}

#[test]
fn test_dtype_nullability_with_nullable_child() {
    // Child can have nullable dtype but no actual nulls.
    // MaskedArray dtype should be determined by validity, not child's dtype.
    let child =
        PrimitiveArray::new(vortex_buffer::buffer![1i32, 2, 3], Validity::AllValid).into_array();

    // Child has nullable dtype.
    assert!(child.dtype().is_nullable());
}

#[test]
fn test_canonical_dtype_matches_array_dtype() -> VortexResult<()> {
    // The canonical form should have the same nullability as the array's dtype.
    let child = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let array = MaskedArray::try_new(child, Validity::AllValid).unwrap();

    let canonical = array.to_canonical()?;
    assert_eq!(canonical.as_ref().dtype(), array.dtype());
    Ok(())
}

#[test]
fn test_masked_child_with_validity() -> VortexResult<()> {
    // When validity has nulls, masked_child should apply inverted mask.
    let child = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array();
    let array =
        MaskedArray::try_new(child, Validity::from_iter([true, false, true, false, true])).unwrap();

    let masked = array.masked_child().unwrap();
    let prim = masked.to_primitive()?;

    // Positions where validity is false should be null in masked_child.
    assert_eq!(prim.valid_count()?, 3);
    assert!(prim.is_valid(0)?);
    assert!(!prim.is_valid(1)?);
    assert!(prim.is_valid(2)?);
    assert!(!prim.is_valid(3)?);
    assert!(prim.is_valid(4)?);

    assert_eq!(
        array.as_ref().display_values().to_string(),
        masked.display_values().to_string()
    );
    Ok(())
}

#[test]
fn test_masked_child_all_valid() -> VortexResult<()> {
    // When validity is AllValid, masked_child should invert to AllInvalid.
    let child = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
    let array = MaskedArray::try_new(child, Validity::AllValid).unwrap();

    let masked = array.masked_child().unwrap();
    assert_eq!(masked.len(), 3);
    assert_eq!(masked.valid_count()?, 3);
    assert_eq!(
        array.as_ref().display_values().to_string(),
        masked.display_values().to_string()
    );
    Ok(())
}

#[rstest]
#[case(Validity::AllValid)]
#[case(Validity::from_iter([true, true, true]))]
#[case(Validity::from_iter([false, false, false]))]
#[case(Validity::from_iter([true, false, true, false]))]
fn test_masked_child_preserves_length(#[case] validity: Validity) -> VortexResult<()> {
    let len = match &validity {
        Validity::Array(arr) => arr.len(),
        _ => 3,
    };

    #[allow(clippy::cast_possible_truncation)]
    let child = PrimitiveArray::from_iter(0..len as i32).into_array();
    let array = MaskedArray::try_new(child, validity.clone()).unwrap();

    let masked = array.masked_child().unwrap();
    assert_eq!(masked.len(), len);
    assert_eq!(masked.validity_mask()?, validity.to_mask(len)?);
    assert_eq!(
        array.as_ref().display_values().to_string(),
        masked.display_values().to_string()
    );
    Ok(())
}
