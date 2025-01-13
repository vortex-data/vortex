//! This module defines array traits for each Vortex DType.
//!
//! When callers only want to make assumptions about the DType, and not about any specific
//! encoding, they can use these traits to write encoding-agnostic code.

use std::sync::Arc;

use vortex_dtype::{DType, ExtDType, Field, FieldInfo, FieldNames, PType};
use vortex_error::{vortex_panic, VortexError, VortexExpect as _, VortexResult};
use vortex_scalar::Scalar;

use crate::array::ConstantArray;
use crate::compute::{invert, mask, try_cast, FilterMask};
use crate::encoding::Encoding;
use crate::validity::LogicalValidity;
use crate::{ArrayDType, ArrayData, ArrayTrait, IntoArrayData as _};

/// An Array encoding must declare which DTypes it can be downcast into.
pub trait VariantsVTable<Array> {
    fn as_null_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn NullArrayTrait> {
        None
    }

    fn as_bool_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn BoolArrayTrait> {
        None
    }

    fn as_primitive_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn PrimitiveArrayTrait> {
        None
    }

    fn as_utf8_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn Utf8ArrayTrait> {
        None
    }

    fn as_binary_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn BinaryArrayTrait> {
        None
    }

    fn as_struct_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn StructArrayTrait> {
        None
    }

    fn as_list_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn ListArrayTrait> {
        None
    }

    fn as_extension_array<'a>(&self, _array: &'a Array) -> Option<&'a dyn ExtensionArrayTrait> {
        None
    }
}

impl<E: Encoding> VariantsVTable<ArrayData> for E
where
    E: VariantsVTable<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn as_null_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn NullArrayTrait> {
        let (array_ref, encoding) = array
            .downcast_array_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_null_array(encoding, array_ref)
    }

    fn as_bool_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn BoolArrayTrait> {
        let (array_ref, encoding) = array
            .downcast_array_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_bool_array(encoding, array_ref)
    }

    fn as_primitive_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn PrimitiveArrayTrait> {
        let (array_ref, encoding) = array
            .downcast_array_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_primitive_array(encoding, array_ref)
    }

    fn as_utf8_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn Utf8ArrayTrait> {
        let (array_ref, encoding) = array
            .downcast_array_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_utf8_array(encoding, array_ref)
    }

    fn as_binary_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn BinaryArrayTrait> {
        let (array_ref, encoding) = array
            .downcast_array_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_binary_array(encoding, array_ref)
    }

    fn as_struct_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn StructArrayTrait> {
        let (array_ref, encoding) = array
            .downcast_array_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_struct_array(encoding, array_ref)
    }

    fn as_list_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn ListArrayTrait> {
        let (array_ref, encoding) = array
            .downcast_array_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_list_array(encoding, array_ref)
    }

    fn as_extension_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn ExtensionArrayTrait> {
        let (array_ref, encoding) = array
            .downcast_array_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_extension_array(encoding, array_ref)
    }
}

/// Provide functions on type-erased ArrayData to downcast into dtype-specific array variants.
impl ArrayData {
    pub fn as_null_array(&self) -> Option<&dyn NullArrayTrait> {
        matches!(self.dtype(), DType::Null)
            .then(|| self.encoding().as_null_array(self))
            .flatten()
    }

    pub fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        matches!(self.dtype(), DType::Bool(..))
            .then(|| self.encoding().as_bool_array(self))
            .flatten()
    }

    pub fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        matches!(self.dtype(), DType::Primitive(..))
            .then(|| self.encoding().as_primitive_array(self))
            .flatten()
    }

    pub fn as_utf8_array(&self) -> Option<&dyn Utf8ArrayTrait> {
        matches!(self.dtype(), DType::Utf8(..))
            .then(|| self.encoding().as_utf8_array(self))
            .flatten()
    }

    pub fn as_binary_array(&self) -> Option<&dyn BinaryArrayTrait> {
        matches!(self.dtype(), DType::Binary(..))
            .then(|| self.encoding().as_binary_array(self))
            .flatten()
    }

    pub fn as_struct_array(&self) -> Option<&dyn StructArrayTrait> {
        matches!(self.dtype(), DType::Struct(..))
            .then(|| self.encoding().as_struct_array(self))
            .flatten()
    }

    pub fn as_list_array(&self) -> Option<&dyn ListArrayTrait> {
        matches!(self.dtype(), DType::List(..))
            .then(|| self.encoding().as_list_array(self))
            .flatten()
    }

    pub fn as_extension_array(&self) -> Option<&dyn ExtensionArrayTrait> {
        matches!(self.dtype(), DType::Extension(..))
            .then(|| self.encoding().as_extension_array(self))
            .flatten()
    }
}

pub trait NullArrayTrait: ArrayTrait {}

pub trait BoolArrayTrait: ArrayTrait {}

pub trait PrimitiveArrayTrait: ArrayTrait {
    /// The logical primitive type of the array.
    ///
    /// This is a type that can safely be converted into a `NativePType` for use in
    /// `maybe_null_slice` or `into_maybe_null_slice`.
    fn ptype(&self) -> PType {
        if let DType::Primitive(ptype, ..) = self.dtype() {
            *ptype
        } else {
            vortex_panic!("array must have primitive data type");
        }
    }
}

pub trait Utf8ArrayTrait: ArrayTrait {}

pub trait BinaryArrayTrait: ArrayTrait {}

pub trait StructArrayTrait: ArrayTrait {
    fn names(&self) -> &FieldNames {
        let DType::Struct(st, _) = self.dtype() else {
            unreachable!()
        };
        st.names()
    }

    fn field_info(&self, field: &Field) -> VortexResult<FieldInfo> {
        let DType::Struct(st, _) = self.dtype() else {
            unreachable!()
        };

        st.field_info(field)
    }

    fn dtypes(&self) -> Vec<DType> {
        let DType::Struct(st, _) = self.dtype() else {
            unreachable!()
        };
        st.dtypes().collect()
    }

    fn nfields(&self) -> usize {
        self.names().len()
    }

    /// Return a field's array by index, masking by the struct's validity.
    ///
    /// If either this array or the field array is invalid at a position, the result is invalid at
    /// that position. Consequently, if either array is nullable, the result is nullable.
    ///
    /// # Examples
    ///
    /// The field of a non-nullable struct array is the same whether accessed by
    /// [StructArrayTrait::field_by_idx] or [StructArrayTrait::maybe_null_field_by_idx]:
    ///
    /// ```
    /// use vortex_array::array::{BoolArray, PrimitiveArray, StructArray};
    /// use vortex_array::validity::{ArrayValidity, Validity};
    /// use vortex_array::variants::StructArrayTrait;
    /// use vortex_array::{ArrayDType, IntoArrayData};
    /// use vortex_dtype::FieldNames;
    ///
    /// let original_field = PrimitiveArray::from_option_iter([
    ///     Some(1), None, Some(3), None, Some(4),
    /// ]).into_array();
    /// let array = StructArray::try_new(
    ///     FieldNames::from(["a".into()]),
    ///     vec![original_field],
    ///     5,
    ///     Validity::NonNullable,
    /// ).unwrap();
    /// let field = array.field_by_idx(0).unwrap().unwrap();
    /// let maybe_null_field = array.maybe_null_field_by_idx(0).unwrap();
    ///
    /// assert_eq!(field.dtype(), maybe_null_field.dtype());
    /// assert!((0..field.len()).all(|i| {
    ///     field.is_valid(i) == maybe_null_field.is_valid(i)
    /// }));
    /// ```
    ///
    /// When both a struct and its field are nullable, [StructArrayTrait::field_by_idx] returns the
    /// intersection of the validity, which is to say: a position is valid if and only if both the
    /// struct and the field are valid at that position.
    ///
    /// ```
    /// use vortex_array::array::{BoolArray, PrimitiveArray, StructArray};
    /// use vortex_array::compute::scalar_at;
    /// use vortex_array::validity::{ArrayValidity, Validity};
    /// use vortex_array::variants::StructArrayTrait;
    /// use vortex_array::{ArrayDType, IntoArrayData};
    /// use vortex_dtype::FieldNames;
    /// use vortex_scalar::Scalar;
    ///
    /// let original_field = PrimitiveArray::from_option_iter([
    ///     Some(1), None, Some(3), None, Some(5),
    /// ]).into_array();
    /// let struct_validity = Validity::Array(BoolArray::from_iter([
    ///     true, true, false, false, true,
    /// ]).into_array());
    /// let array = StructArray::try_new(
    ///     FieldNames::from(["a".into()]),
    ///     vec![original_field],
    ///     5,
    ///     struct_validity,
    /// ).unwrap();
    /// let field = array.field_by_idx(0).unwrap().unwrap();
    ///
    /// assert!(field.dtype().is_nullable());
    /// assert_eq!(scalar_at(&field, 0).unwrap(), Scalar::from(Some(1)));
    /// assert!(!field.is_valid(1));
    /// assert!(!field.is_valid(2));
    /// assert!(!field.is_valid(3));
    /// assert_eq!(scalar_at(&field, 4).unwrap(), Scalar::from(Some(5)));
    /// ```
    ///
    /// When a field is non-nullable, but the struct is nullable, the field receives the struct's
    /// validity.
    ///
    /// ```
    /// use vortex_array::array::{BoolArray, StructArray};
    /// use vortex_array::compute::scalar_at;
    /// use vortex_array::validity::{ArrayValidity, Validity};
    /// use vortex_array::variants::StructArrayTrait;
    /// use vortex_array::{ArrayDType, IntoArrayData};
    /// use vortex_buffer::buffer;
    /// use vortex_dtype::FieldNames;
    /// use vortex_scalar::Scalar;
    ///
    /// let original_field = buffer![1, 2, 3, 4, 5].into_array();
    /// let struct_validity = Validity::Array(BoolArray::from_iter([
    ///     true, true, false, false, true,
    /// ]).into_array());
    /// let array = StructArray::try_new(
    ///     FieldNames::from(["a".into()]),
    ///     vec![original_field],
    ///     5,
    ///     struct_validity,
    /// ).unwrap();
    /// let field = array.field_by_idx(0).unwrap().unwrap();
    ///
    /// assert!(field.dtype().is_nullable());
    /// assert_eq!(scalar_at(&field, 0).unwrap(), Scalar::from(Some(1)));
    /// assert_eq!(scalar_at(&field, 1).unwrap(), Scalar::from(Some(2)));
    /// assert!(!field.is_valid(2));
    /// assert!(!field.is_valid(3));
    /// assert_eq!(scalar_at(&field, 4).unwrap(), Scalar::from(Some(5)));
    /// ```
    fn field_by_idx(&self, idx: usize) -> VortexResult<Option<ArrayData>> {
        let Some(maybe_null_field) = self.maybe_null_field_by_idx(idx) else {
            return Ok(None);
        };

        if !self.dtype().is_nullable() {
            return Ok(Some(maybe_null_field));
        }

        match self.logical_validity() {
            LogicalValidity::AllValid(_) => {
                let nullable_dtype = maybe_null_field.dtype().as_nullable();
                try_cast(maybe_null_field, &nullable_dtype).map(Some)
            }
            LogicalValidity::AllInvalid(_) => {
                let nullable_dtype = maybe_null_field.dtype().as_nullable();

                Ok(Some(
                    ConstantArray::new(Scalar::null(nullable_dtype), maybe_null_field.len())
                        .into_array(),
                ))
            }
            LogicalValidity::Array(is_valid) => {
                mask(&maybe_null_field, FilterMask::try_from(invert(&is_valid)?)?).map(Some)
            }
        }
    }

    /// Return a field's array by name, masking by the struct's validity.
    ///
    /// See also [StructArrayTrait::field_by_idx].
    fn field_by_name(&self, name: &str) -> VortexResult<Option<ArrayData>> {
        let field_idx = self
            .names()
            .iter()
            .position(|field_name| field_name.as_ref() == name);

        match field_idx {
            None => Ok(None),
            Some(field_idx) => self.field_by_idx(field_idx),
        }
    }

    /// Return a field's array by name or index, masking by the struct's validity.
    ///
    /// See also [StructArrayTrait::field_by_idx].
    fn field(&self, field: &Field) -> VortexResult<Option<ArrayData>> {
        match field {
            Field::Index(idx) => self.field_by_idx(*idx),
            Field::Name(name) => self.field_by_name(name.as_ref()),
        }
    }

    /// Return a field's array by index, ignoring struct nullability
    fn maybe_null_field_by_idx(&self, idx: usize) -> Option<ArrayData>;

    /// Return a field's array by name, ignoring struct nullability
    fn maybe_null_field_by_name(&self, name: &str) -> Option<ArrayData> {
        let field_idx = self
            .names()
            .iter()
            .position(|field_name| field_name.as_ref() == name);

        field_idx.and_then(|field_idx| self.maybe_null_field_by_idx(field_idx))
    }

    /// Return a field's array by name or index, ignoring struct nullability
    fn maybe_null_field(&self, field: &Field) -> Option<ArrayData> {
        match field {
            Field::Index(idx) => self.maybe_null_field_by_idx(*idx),
            Field::Name(name) => self.maybe_null_field_by_name(name.as_ref()),
        }
    }

    fn project(&self, projection: &[Field]) -> VortexResult<ArrayData>;
}

pub trait ListArrayTrait: ArrayTrait {}

pub trait ExtensionArrayTrait: ArrayTrait {
    /// Returns the extension logical [`DType`].
    fn ext_dtype(&self) -> &Arc<ExtDType> {
        let DType::Extension(ext_dtype) = self.dtype() else {
            vortex_panic!("Expected ExtDType")
        };
        ext_dtype
    }

    /// Returns the underlying [`ArrayData`], without the [`ExtDType`].
    fn storage_data(&self) -> ArrayData;
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::FieldNames;
    use vortex_scalar::Scalar;

    use crate::array::{BoolArray, PrimitiveArray, StructArray};
    use crate::compute::scalar_at;
    use crate::validity::{ArrayValidity, Validity};
    use crate::variants::StructArrayTrait;
    use crate::{ArrayDType, IntoArrayData};

    #[test]
    fn test_field() {
        let original_field =
            PrimitiveArray::from_option_iter([Some(1), None, Some(3), None, Some(5)]).into_array();
        let array = StructArray::try_new(
            FieldNames::from(["a".into()]),
            vec![original_field.clone()],
            5,
            Validity::NonNullable,
        )
        .unwrap();
        let field = array.field_by_idx(0).unwrap().unwrap();
        let maybe_null_field = array.maybe_null_field_by_idx(0).unwrap();

        assert_eq!(field.dtype(), maybe_null_field.dtype());
        assert!((0..field.len()).all(|i| { field.is_valid(i) == maybe_null_field.is_valid(i) }));

        let struct_validity =
            Validity::Array(BoolArray::from_iter([true, true, false, false, true]).into_array());
        let array = StructArray::try_new(
            FieldNames::from(["a".into()]),
            vec![original_field],
            5,
            struct_validity.clone(),
        )
        .unwrap();
        let field = array.field_by_idx(0).unwrap().unwrap();

        assert!(field.dtype().is_nullable());
        assert_eq!(scalar_at(&field, 0).unwrap(), Scalar::from(Some(1)));
        assert!(!field.is_valid(1));
        assert!(!field.is_valid(2));
        assert!(!field.is_valid(3));
        assert_eq!(scalar_at(&field, 4).unwrap(), Scalar::from(Some(5)));

        let original_field = buffer![1, 2, 3, 4, 5].into_array();
        let array = StructArray::try_new(
            FieldNames::from(["a".into()]),
            vec![original_field],
            5,
            struct_validity,
        )
        .unwrap();
        let field = array.field_by_idx(0).unwrap().unwrap();

        assert!(field.dtype().is_nullable());
        assert_eq!(scalar_at(&field, 0).unwrap(), Scalar::from(Some(1)));
        assert_eq!(scalar_at(&field, 1).unwrap(), Scalar::from(Some(2)));
        assert!(!field.is_valid(2));
        assert!(!field.is_valid(3));
        assert_eq!(scalar_at(&field, 4).unwrap(), Scalar::from(Some(5)));
    }
}
