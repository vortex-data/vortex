//! This module defines array traits for each Vortex DType.
//!
//! When callers only want to make assumptions about the DType, and not about any specific
//! encoding, they can use these traits to write encoding-agnostic code.

use std::cmp::Ordering;
use std::sync::Arc;

use vortex_dtype::{DType, ExtDType, FieldName, FieldNames, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_scalar::PValue;

use crate::compute::{IndexOrd, sum};
use crate::{Array, ArrayRef};

pub trait NullArrayTrait: Array {}

pub trait BoolArrayTrait: Array {}

impl dyn BoolArrayTrait + '_ {
    pub fn true_count(&self) -> VortexResult<usize> {
        let true_count = sum(self)?;
        Ok(true_count
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("true count should never overflow usize")
            .vortex_expect("true count should never be null"))
    }
}

pub trait PrimitiveArrayTrait: Array {
    fn ptype(&self) -> PType {
        let DType::Primitive(ptype, _) = self.dtype() else {
            vortex_panic!("Expected Primitive DType")
        };
        *ptype
    }

    /// Return the primitive value at the given index.
    fn value(&self, idx: usize) -> Option<PValue> {
        self.scalar_at(idx)
            .vortex_expect("scalar at index")
            .as_primitive()
            .pvalue()
    }
}

impl IndexOrd<Option<PValue>> for dyn PrimitiveArrayTrait + '_ {
    fn index_cmp(&self, idx: usize, elem: &Option<PValue>) -> Option<Ordering> {
        self.value(idx).partial_cmp(elem)
    }

    fn index_len(&self) -> usize {
        Array::len(self)
    }
}

pub trait Utf8ArrayTrait: Array {}

pub trait BinaryArrayTrait: Array {}

pub trait DecimalArrayTrait: Array {}

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

pub trait ListArrayTrait: Array {}

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
