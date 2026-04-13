// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::*;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::ToCanonical as _;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::validity::Validity;

#[rstest]
#[case(Validity::AllValid, Nullability::Nullable)]
#[case(Validity::from_iter([true, false, true]), Nullability::Nullable)]
fn test_dtype_nullability(#[case] validity: Validity, #[case] expected: Nullability) {
    let child = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
    let array = MaskedArray::try_new(child, validity).unwrap();

    assert_eq!(
        array.dtype(),
        &DType::Primitive(crate::dtype::PType::I32, expected)
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
    assert_eq!(canonical.dtype(), array.dtype());
    Ok(())
}

#[test]
fn test_masked_child_with_validity() {
    // When validity has nulls, masked_child should apply inverted mask.
    let child = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array();
    let array =
        MaskedArray::try_new(child, Validity::from_iter([true, false, true, false, true])).unwrap();

    let prim = array.as_array().to_primitive();

    // Positions where validity is false should be null in masked_child.
    assert_eq!(prim.valid_count().unwrap(), 3);
    assert!(prim.is_valid(0).unwrap());
    assert!(!prim.is_valid(1).unwrap());
    assert!(prim.is_valid(2).unwrap());
    assert!(!prim.is_valid(3).unwrap());
    assert!(prim.is_valid(4).unwrap());
}

#[test]
fn test_masked_child_all_valid() {
    // When validity is AllValid, masked_child should invert to AllInvalid.
    let child = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
    let array = MaskedArray::try_new(child, Validity::AllValid).unwrap();

    assert_eq!(array.len(), 3);
    assert_eq!(array.valid_count().unwrap(), 3);
    assert_arrays_eq!(
        PrimitiveArray::from_option_iter([10i32, 20, 30].map(Some)),
        array
    );
}

#[rstest]
#[case(Validity::AllValid)]
#[case(Validity::from_iter([true, true, true]))]
#[case(Validity::from_iter([false, false, false]))]
#[case(Validity::from_iter([true, false, true, false]))]
fn test_masked_child_preserves_length(#[case] validity: Validity) {
    let len = match &validity {
        Validity::Array(arr) => arr.len(),
        _ => 3,
    };

    #[expect(clippy::cast_possible_truncation)]
    let child = PrimitiveArray::from_iter(0..len as i32).into_array();
    let array = MaskedArray::try_new(child, validity.clone()).unwrap();

    assert_eq!(array.len(), len);

    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    assert!(
        array
            .validity()
            .vortex_expect("masked validity should be derivable")
            .mask_eq(&validity, &mut ctx)
            .unwrap(),
    );
}
