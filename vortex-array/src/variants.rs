//! This module defines array traits for each Vortex DType.
//!
//! When callers only want to make assumptions about the DType, and not about any specific
//! encoding, they can use these traits to write encoding-agnostic code.

use std::sync::Arc;

use vortex_dtype::{DType, ExtDType, FieldName, FieldNames, PType};
use vortex_error::{vortex_err, vortex_panic, VortexExpect, VortexResult};

use crate::{Array, ArrayRef};

pub trait NullArrayTrait {}

pub trait BoolArrayTrait {}

pub trait PrimitiveArrayTrait: Array {
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

pub trait StructArrayTrait: Array {
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
    fn maybe_null_field_by_idx(&self, idx: usize) -> VortexResult<ArrayRef>;

    /// Return a field's array by name, ignoring struct nullability
    fn maybe_null_field_by_name(&self, name: &str) -> VortexResult<ArrayRef> {
        let field_idx = self
            .names()
            .iter()
            .position(|field_name| field_name.as_ref() == name)
            .ok_or_else(|| vortex_err!("Field not found: {}", name))?;
        self.maybe_null_field_by_idx(field_idx)
    }

    fn project(&self, projection: &[FieldName]) -> VortexResult<ArrayRef>;
}

impl dyn StructArrayTrait + '_ {
    pub fn fields(&self) -> impl Iterator<Item = ArrayRef> + '_ {
        (0..self.nfields()).map(|i| {
            self.maybe_null_field_by_idx(i)
                .vortex_expect("never out of bounds")
        })
    }
}

pub trait ListArrayTrait {}

pub trait ExtensionArrayTrait: Array {
    /// Returns the extension logical [`DType`].
    fn ext_dtype(&self) -> &Arc<ExtDType> {
        let DType::Extension(ext_dtype) = self.dtype() else {
            vortex_panic!("Expected ExtDType")
        };
        ext_dtype
    }

    /// Returns the underlying [`ArrayRef`], without the [`ExtDType`].
    fn storage_data(&self) -> ArrayRef;
}
