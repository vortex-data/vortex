//! This module defines array traits for each Vortex DType.
//!
//! When callers only want to make assumptions about the DType, and not about any specific
//! encoding, they can use these traits to write encoding-agnostic code.

use std::ops::Deref;
use std::sync::Arc;

use vortex_dtype::{DType, ExtDType, FieldName, FieldNames, PType};
use vortex_error::{vortex_err, vortex_panic, VortexResult};

use crate::Array;

/// Provide functions on type-erased Array to downcast into dtype-specific array variants.
impl Array {
    pub fn as_null_array(&self) -> Option<&dyn NullArrayTrait> {
        matches!(self.dtype(), DType::Null)
            .then(|| self.vtable().as_null_array(self))
            .flatten()
    }

    pub fn as_bool_array(&self) -> Option<&dyn BoolArrayTrait> {
        matches!(self.dtype(), DType::Bool(..))
            .then(|| self.vtable().as_bool_array(self))
            .flatten()
    }

    pub fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        matches!(self.dtype(), DType::Primitive(..))
            .then(|| self.vtable().as_primitive_array(self))
            .flatten()
    }

    pub fn as_utf8_array(&self) -> Option<&dyn Utf8ArrayTrait> {
        matches!(self.dtype(), DType::Utf8(..))
            .then(|| self.vtable().as_utf8_array(self))
            .flatten()
    }

    pub fn as_binary_array(&self) -> Option<&dyn BinaryArrayTrait> {
        matches!(self.dtype(), DType::Binary(..))
            .then(|| self.vtable().as_binary_array(self))
            .flatten()
    }

    pub fn as_struct_array(&self) -> Option<&dyn StructArrayTrait> {
        matches!(self.dtype(), DType::Struct(..))
            .then(|| self.vtable().as_struct_array(self))
            .flatten()
    }

    pub fn as_list_array(&self) -> Option<&dyn ListArrayTrait> {
        matches!(self.dtype(), DType::List(..))
            .then(|| self.vtable().as_list_array(self))
            .flatten()
    }

    pub fn as_extension_array(&self) -> Option<&dyn ExtensionArrayTrait> {
        matches!(self.dtype(), DType::Extension(..))
            .then(|| self.vtable().as_extension_array(self))
            .flatten()
    }
}

pub trait NullArrayTrait {}

pub trait BoolArrayTrait {}

pub trait PrimitiveArrayTrait: Deref<Target = Array> {
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

pub trait Utf8ArrayTrait {}

pub trait BinaryArrayTrait {}

pub trait StructArrayTrait: Deref<Target = Array> {
    fn names(&self) -> &FieldNames {
        let DType::Struct(st, _) = self.dtype() else {
            unreachable!()
        };
        st.names()
    }

    fn dtypes(&self) -> Vec<DType> {
        let DType::Struct(st, _) = self.dtype() else {
            unreachable!()
        };
        st.fields().collect()
    }

    fn nfields(&self) -> usize {
        self.names().len()
    }

    /// Return a field's array by index, ignoring struct nullability
    fn maybe_null_field_by_idx(&self, idx: usize) -> VortexResult<Array>;

    /// Return a field's array by name, ignoring struct nullability
    fn maybe_null_field_by_name(&self, name: &str) -> VortexResult<Array> {
        let field_idx = self
            .names()
            .iter()
            .position(|field_name| field_name.as_ref() == name)
            .ok_or_else(|| vortex_err!("Field not found: {}", name))?;
        self.maybe_null_field_by_idx(field_idx)
    }

    fn project(&self, projection: &[FieldName]) -> VortexResult<Array>;
}

pub trait ListArrayTrait {}

pub trait ExtensionArrayTrait: Deref<Target = Array> {
    /// Returns the extension logical [`DType`].
    fn ext_dtype(&self) -> &Arc<ExtDType> {
        let DType::Extension(ext_dtype) = self.dtype() else {
            vortex_panic!("Expected ExtDType")
        };
        ext_dtype
    }

    /// Returns the underlying [`Array`], without the [`ExtDType`].
    fn storage_data(&self) -> Array;
}
