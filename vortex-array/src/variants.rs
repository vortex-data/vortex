//! This module defines array traits for each Vortex DType.
//!
//! When callers only want to make assumptions about the DType, and not about any specific
//! encoding, they can use these traits to write encoding-agnostic code.

use std::sync::Arc;

use vortex_dtype::{DType, ExtDType, Field, FieldInfo, FieldName, FieldNames, PType};
use vortex_error::{vortex_panic, VortexResult};

use crate::{ArrayDType, ArrayData, ArrayTrait};

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

    fn maybe_null_field(&self, field: &Field) -> Option<ArrayData> {
        match field {
            Field::Index(idx) => self.maybe_null_field_by_idx(*idx),
            Field::Name(name) => self.maybe_null_field_by_name(name.as_ref()),
        }
    }

    fn project(&self, projection: &[FieldName]) -> VortexResult<ArrayData>;
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
