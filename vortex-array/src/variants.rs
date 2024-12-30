//! This module defines array traits for each Vortex DType.
//!
//! When callers only want to make assumptions about the DType, and not about any specific
//! encoding, they can use these traits to write encoding-agnostic code.

use std::sync::Arc;

use vortex_dtype::field::Field;
use vortex_dtype::{DType, ExtDType, FieldNames, PType};
use vortex_error::{vortex_panic, VortexError, VortexExpect as _, VortexResult};

use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData, ArrayTrait};

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
        let array_ref =
            <&E::Array>::try_from(array).vortex_expect("Failed to get array as reference");
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_null_array(encoding, array_ref)
    }

    fn as_bool_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn BoolArrayTrait> {
        let array_ref =
            <&E::Array>::try_from(array).vortex_expect("Failed to get array as reference");
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_bool_array(encoding, array_ref)
    }

    fn as_primitive_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn PrimitiveArrayTrait> {
        let array_ref =
            <&E::Array>::try_from(array).vortex_expect("Failed to get array as reference");
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_primitive_array(encoding, array_ref)
    }

    fn as_utf8_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn Utf8ArrayTrait> {
        let array_ref =
            <&E::Array>::try_from(array).vortex_expect("Failed to get array as reference");
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_utf8_array(encoding, array_ref)
    }

    fn as_binary_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn BinaryArrayTrait> {
        let array_ref =
            <&E::Array>::try_from(array).vortex_expect("Failed to get array as reference");
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_binary_array(encoding, array_ref)
    }

    fn as_struct_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn StructArrayTrait> {
        let array_ref =
            <&E::Array>::try_from(array).vortex_expect("Failed to get array as reference");
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_struct_array(encoding, array_ref)
    }

    fn as_list_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn ListArrayTrait> {
        let array_ref =
            <&E::Array>::try_from(array).vortex_expect("Failed to get array as reference");
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .vortex_expect("Failed to downcast encoding");
        VariantsVTable::as_list_array(encoding, array_ref)
    }

    fn as_extension_array<'a>(&self, array: &'a ArrayData) -> Option<&'a dyn ExtensionArrayTrait> {
        let array_ref =
            <&E::Array>::try_from(array).vortex_expect("Failed to get array as reference");
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
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

    fn dtypes(&self) -> &[DType] {
        let DType::Struct(st, _) = self.dtype() else {
            unreachable!()
        };
        st.dtypes()
    }

    fn nfields(&self) -> usize {
        self.names().len()
    }

    /// Return a field's array by index
    fn field(&self, idx: usize) -> Option<ArrayData>;

    /// Return a field's array by name
    fn field_by_name(&self, name: &str) -> Option<ArrayData> {
        let field_idx = self
            .names()
            .iter()
            .position(|field_name| field_name.as_ref() == name);

        field_idx.and_then(|field_idx| self.field(field_idx))
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
