// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_decimal_value_type;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

/// Apply a logical AND of a validity and a mask.
/// This needs to be coherent with applications of Mask.
/// The result is always nullable. The result has the same length as self.
#[inline]
pub fn mask_validity(validity: &Validity, mask: &Mask) -> Validity {
    let out = match mask.bit_buffer() {
        AllOr::All => validity.clone().into_nullable(),
        AllOr::None => Validity::AllInvalid,
        AllOr::Some(make_valid) => match validity {
            Validity::AllInvalid => Validity::AllInvalid,
            Validity::NonNullable | Validity::AllValid => {
                Validity::from_bit_buffer(make_valid.clone(), Nullability::Nullable)
            }
            Validity::Array(is_valid) => {
                let is_valid = is_valid.to_bool();
                Validity::from_bit_buffer(
                    is_valid.to_bit_buffer() & make_valid,
                    Nullability::Nullable,
                )
            }
        },
    };

    tracing::debug!(validity = ?validity, mask = ?mask, out = ?out, "generated fuzzer mask");
    out
}

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
            let new_validity = mask_validity(&array.validity()?, mask);
            BoolArray::new(array.to_bit_buffer(), new_validity).into_array()
        }
        Canonical::Primitive(array) => {
            let new_validity = mask_validity(&array.validity()?, mask);
            PrimitiveArray::from_buffer_handle(
                array.buffer_handle().clone(),
                array.ptype(),
                new_validity,
            )
            .into_array()
        }
        Canonical::Decimal(array) => {
            let new_validity = mask_validity(&array.validity()?, mask);
            match_each_decimal_value_type!(array.values_type(), |D| {
                DecimalArray::new(array.buffer::<D>(), array.decimal_dtype(), new_validity)
                    .into_array()
            })
        }
        Canonical::VarBinView(array) => {
            let new_validity = mask_validity(&array.validity()?, mask);
            VarBinViewArray::new_handle(
                array.views_handle().clone(),
                Arc::clone(array.data_buffers()),
                array.dtype().with_nullability(new_validity.nullability()),
                new_validity,
            )
            .into_array()
        }
        Canonical::List(array) => {
            let new_validity = mask_validity(&array.validity()?, mask);

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
            let new_validity = mask_validity(&array.validity()?, mask);
            FixedSizeListArray::new(
                array.elements().clone(),
                array.list_size(),
                new_validity,
                array.len(),
            )
            .into_array()
        }
        Canonical::Struct(array) => {
            let new_validity = mask_validity(&array.validity()?, mask);
            StructArray::try_new_with_dtype(
                array.unmasked_fields(),
                array.struct_fields().clone(),
                array.len(),
                new_validity,
            )
            .vortex_expect("StructArray creation should succeed in fuzz test")
            .into_array()
        }
        Canonical::Extension(array) => {
            // Recursively mask the storage array
            let masked_storage = mask_canonical_array(array.storage_array().to_canonical()?, mask)
                .vortex_expect("mask_canonical_array should succeed in fuzz test");

            let ext_dtype = array
                .ext_dtype()
                .with_nullability(masked_storage.dtype().nullability());
            ExtensionArray::new(ext_dtype, masked_storage).into_array()
        }
        Canonical::Variant(_) => unreachable!("Variant arrays are not fuzzed"),
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::ListViewArray;
    use vortex_array::arrays::NullArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_mask::Mask;

    use super::mask_canonical_array;

    #[test]
    fn test_mask_null_array() {
        let array = NullArray::new(5);
        let mask = Mask::from_iter([true, false, true, false, true]);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        assert_eq!(result.len(), 5);
        // All values should still be null
        for i in 0..5 {
            assert!(!result.is_valid(i).unwrap());
        }
    }

    #[test]
    fn test_mask_bool_array() {
        let array = BoolArray::from_iter([true, false, true, false, true]);
        let mask = Mask::from_iter([false, true, true, false, true]);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        let expected = BoolArray::from_iter([None, Some(false), Some(true), None, Some(true)]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_mask_primitive_array() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let mask = Mask::from_iter([true, false, true, false, true]);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        let expected = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_mask_primitive_array_with_nulls() {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]);
        let mask = Mask::from_iter([false, true, true, false, true]);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        let expected = PrimitiveArray::from_option_iter([None, None, Some(3i32), None, None]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_mask_decimal_array() {
        let dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::from_option_iter(
            [Some(1i128), Some(2), Some(3), Some(4), Some(5)],
            dtype,
        );
        let mask = Mask::from_iter([true, true, false, true, true]);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        let expected =
            DecimalArray::from_option_iter([Some(1i128), Some(2), None, Some(4), Some(5)], dtype);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_mask_varbinview_array() {
        let array = VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]);
        let mask = Mask::from_iter([false, true, false, true, false]);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        let expected =
            VarBinViewArray::from_iter_nullable_str([None, Some("two"), None, Some("four"), None]);
        assert_arrays_eq!(result, expected);
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

        let mask = Mask::from_iter([true, false, true]);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.is_valid(0).unwrap());
        assert!(!result.is_valid(1).unwrap());
        assert!(result.is_valid(2).unwrap());
    }

    #[test]
    fn test_mask_fixed_size_list_array() {
        let elements = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6]).into_array();
        let array =
            FixedSizeListArray::try_new(elements, 2, Nullability::NonNullable.into(), 3).unwrap();

        let mask = Mask::from_iter([false, true, false]);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        assert_eq!(result.len(), 3);
        assert!(!result.is_valid(0).unwrap());
        assert!(result.is_valid(1).unwrap());
        assert!(!result.is_valid(2).unwrap());
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

        let mask = Mask::from_iter([true, false, true]);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.is_valid(0).unwrap());
        assert!(!result.is_valid(1).unwrap());
        assert!(result.is_valid(2).unwrap());
    }

    #[test]
    fn test_mask_all_false() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let mask = Mask::AllFalse(5);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        let expected = PrimitiveArray::from_option_iter([None, None, None, None, None::<i32>]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_mask_all_true() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]);
        let mask = Mask::AllTrue(5);

        let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();

        let expected =
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), Some(3), Some(4), Some(5)]);
        assert_arrays_eq!(result, expected);
    }

    #[test]
    fn test_mask_empty_array() {
        let array = PrimitiveArray::from_iter(Vec::<i32>::new());
        for mask in [Mask::AllFalse(0), Mask::AllTrue(0)] {
            let result = mask_canonical_array(array.to_canonical().unwrap(), &mask).unwrap();
            assert_eq!(result.len(), 0);
        }
    }
}
