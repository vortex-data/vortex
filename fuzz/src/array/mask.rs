// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::arrays::{
    BoolArray, DecimalArray, ExtensionArray, FixedSizeListArray, ListViewArray, PrimitiveArray,
    StructArray, VarBinViewArray,
};
use vortex_array::vtable::ValidityHelper;
use vortex_array::{ArrayRef, Canonical, IntoArray};
use vortex_dtype::{ExtDType, match_each_decimal_value_type};
use vortex_error::{VortexResult, VortexUnwrap};
use vortex_mask::Mask;

/// Apply mask on the canonical form of the array to get a consistent baseline.
/// This implementation manually applies the mask to each canonical type
/// without using the mask_fn method, to serve as an independent baseline for testing.
pub fn mask_canonical_array(canonical: Canonical, mask: &Mask) -> VortexResult<ArrayRef> {
    Ok(match canonical {
        Canonical::Null(array) => {
            // Null arrays are already all invalid, masking has no effect
            array.into_array()
        }
        Canonical::Bool(array) => {
            let new_validity = array.validity().mask(mask);
            BoolArray::from_bit_buffer(array.bit_buffer().clone(), new_validity).into_array()
        }
        Canonical::Primitive(array) => {
            let new_validity = array.validity().mask(mask);
            PrimitiveArray::from_byte_buffer(
                array.byte_buffer().clone(),
                array.ptype(),
                new_validity,
            )
            .into_array()
        }
        Canonical::Decimal(array) => {
            let new_validity = array.validity().mask(mask);
            match_each_decimal_value_type!(array.values_type(), |D| {
                DecimalArray::new(array.buffer::<D>(), array.decimal_dtype(), new_validity)
                    .into_array()
            })
        }
        Canonical::VarBinView(array) => {
            let new_validity = array.validity().mask(mask);
            VarBinViewArray::new(
                array.views().clone(),
                array.buffers().clone(),
                array.dtype().with_nullability(new_validity.nullability()),
                new_validity,
            )
            .into_array()
        }
        Canonical::List(array) => {
            let new_validity = array.validity().mask(mask);

            // SAFETY: Since we are only masking the validity and everything else comes from an
            // already valid `ListViewArray`, all of the invariants are still upheld.
            unsafe {
                ListViewArray::new_unchecked(
                    array.elements().clone(),
                    array.offsets().clone(),
                    array.sizes().clone(),
                    new_validity,
                )
                .with_zero_copy_to_list(array.is_zero_copy_to_list())
            }
            .into_array()
        }
        Canonical::FixedSizeList(array) => {
            let new_validity = array.validity().mask(mask);
            FixedSizeListArray::new(
                array.elements().clone(),
                array.list_size(),
                new_validity,
                array.len(),
            )
            .into_array()
        }
        Canonical::Struct(array) => {
            let new_validity = array.validity().mask(mask);
            StructArray::try_new_with_dtype(
                array.fields().clone(),
                array.struct_fields().clone(),
                array.len(),
                new_validity,
            )
            .vortex_unwrap()
            .into_array()
        }
        Canonical::Extension(array) => {
            // Recursively mask the storage array
            let masked_storage =
                mask_canonical_array(array.storage().to_canonical(), mask).vortex_unwrap();

            if masked_storage.dtype().nullability()
                == array.ext_dtype().storage_dtype().nullability()
            {
                ExtensionArray::new(array.ext_dtype().clone(), masked_storage).into_array()
            } else {
                // The storage dtype changed (i.e., became nullable due to masking)
                let ext_dtype = Arc::new(ExtDType::new(
                    array.ext_dtype().id().clone(),
                    Arc::new(masked_storage.dtype().clone()),
                    array.ext_dtype().metadata().cloned(),
                ));
                ExtensionArray::new(ext_dtype, masked_storage).into_array()
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::{
        BoolArray, DecimalArray, FixedSizeListArray, ListViewArray, NullArray, PrimitiveArray,
        StructArray, VarBinViewArray,
    };
    use vortex_array::{Array, IntoArray};
    use vortex_dtype::{DecimalDType, FieldNames, Nullability};
    use vortex_mask::Mask;
    use vortex_scalar::Scalar;

    use super::mask_canonical_array;

    #[test]
    fn test_mask_null_array() {
        let array = NullArray::new(5);
        let mask = Mask::from_iter([true, false, true, false, true]);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 5);
        // All values should still be null
        for i in 0..5 {
            assert!(!result.is_valid(i));
        }
    }

    #[test]
    fn test_mask_bool_array() {
        let array = BoolArray::from_iter([true, false, true, false, true]);
        let mask = Mask::from_iter([true, false, false, true, false]);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 5);
        assert!(!result.is_valid(0));
        assert_eq!(result.scalar_at(1), Scalar::from(Some(false)));
        assert_eq!(result.scalar_at(2), Scalar::from(Some(true)));
        assert!(!result.is_valid(3));
        assert_eq!(result.scalar_at(4), Scalar::from(Some(true)));
    }

    #[test]
    fn test_mask_primitive_array() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let mask = Mask::from_iter([false, true, false, true, false]);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.scalar_at(0), Scalar::from(Some(1)));
        assert!(!result.is_valid(1));
        assert_eq!(result.scalar_at(2), Scalar::from(Some(3)));
        assert!(!result.is_valid(3));
        assert_eq!(result.scalar_at(4), Scalar::from(Some(5)));
    }

    #[test]
    fn test_mask_primitive_array_with_nulls() {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]);
        let mask = Mask::from_iter([true, false, false, true, false]);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 5);
        assert!(!result.is_valid(0));
        assert!(!result.is_valid(1)); // was already null
        assert_eq!(result.scalar_at(2), Scalar::from(Some(3)));
        assert!(!result.is_valid(3));
        assert!(!result.is_valid(4)); // was already null
    }

    #[test]
    fn test_mask_decimal_array() {
        let array = DecimalArray::from_option_iter(
            [Some(1i128), Some(2), Some(3), Some(4), Some(5)],
            DecimalDType::new(10, 2),
        );
        let mask = Mask::from_iter([false, false, true, false, false]);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 5);
        assert!(result.is_valid(0));
        assert!(result.is_valid(1));
        assert!(!result.is_valid(2));
        assert!(result.is_valid(3));
        assert!(result.is_valid(4));
    }

    #[test]
    fn test_mask_varbinview_array() {
        let array = VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]);
        let mask = Mask::from_iter([true, false, true, false, true]);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 5);
        assert!(!result.is_valid(0));
        assert_eq!(
            result.scalar_at(1),
            Scalar::utf8("two", Nullability::Nullable)
        );
        assert!(!result.is_valid(2));
        assert_eq!(
            result.scalar_at(3),
            Scalar::utf8("four", Nullability::Nullable)
        );
        assert!(!result.is_valid(4));
    }

    #[test]
    fn test_mask_list_array() {
        let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]).into_array();
        let offsets = PrimitiveArray::from_iter([0i32, 2, 4]).into_array();
        let sizes = PrimitiveArray::from_iter([2i32, 2, 2]).into_array();
        let array = unsafe {
            ListViewArray::new_unchecked(elements, offsets, sizes, Nullability::NonNullable.into())
                .with_zero_copy_to_list(true)
        };

        let mask = Mask::from_iter([false, true, false]);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.is_valid(0));
        assert!(!result.is_valid(1));
        assert!(result.is_valid(2));
    }

    #[test]
    fn test_mask_fixed_size_list_array() {
        let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]).into_array();
        let array =
            FixedSizeListArray::try_new(elements, 2, Nullability::NonNullable.into(), 3).unwrap();

        let mask = Mask::from_iter([true, false, true]);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 3);
        assert!(!result.is_valid(0));
        assert!(result.is_valid(1));
        assert!(!result.is_valid(2));
    }

    #[test]
    fn test_mask_struct_array() {
        let field1 = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let field2 = PrimitiveArray::from_iter([4i32, 5, 6]).into_array();
        let fields = vec![field1, field2];

        let array = StructArray::try_new(
            FieldNames::from(["a", "b"]),
            fields,
            3,
            Nullability::NonNullable.into(),
        )
        .unwrap();

        let mask = Mask::from_iter([false, true, false]);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.is_valid(0));
        assert!(!result.is_valid(1));
        assert!(result.is_valid(2));
    }

    #[test]
    fn test_mask_all_true() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let mask = Mask::AllTrue(5);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 5);
        // All values should be masked out (null)
        for i in 0..5 {
            assert!(!result.is_valid(i));
        }
    }

    #[test]
    fn test_mask_all_false() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let mask = Mask::AllFalse(5);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 5);
        // No values should be masked out
        for i in 0..5 {
            assert!(result.is_valid(i));
            #[allow(clippy::cast_possible_truncation)]
            let expected = (i + 1) as i32;
            assert_eq!(result.scalar_at(i), Scalar::from(Some(expected)));
        }
    }

    #[test]
    fn test_mask_empty_array() {
        let array = PrimitiveArray::from_iter(Vec::<i32>::new());
        let mask = Mask::AllFalse(0);

        let result = mask_canonical_array(array.to_canonical(), &mask).unwrap();

        assert_eq!(result.len(), 0);
    }
}
