//! This module defines extension functionality specific to each Vortex DType.
use std::cmp::Ordering;
use std::sync::Arc;

use vortex_dtype::{DType, ExtDType, FieldNames, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_scalar::PValue;

use crate::Array;
use crate::compute::sum;
use crate::search_sorted::IndexOrd;

impl dyn Array + '_ {
    /// Downcasts the array for null-specific behavior.
    pub fn as_null_typed(&self) -> NullTyped {
        matches!(self.dtype(), DType::Null)
            .then(|| NullTyped(self))
            .vortex_expect("Array does not have DType::Null")
    }

    /// Downcasts the array for bool-specific behavior.
    pub fn as_bool_typed(&self) -> BoolTyped {
        matches!(self.dtype(), DType::Bool(..))
            .then(|| BoolTyped(self))
            .vortex_expect("Array does not have DType::Bool")
    }

    /// Downcasts the array for primitive-specific behavior.
    pub fn as_primitive_typed(&self) -> PrimitiveTyped {
        matches!(self.dtype(), DType::Primitive(..))
            .then(|| PrimitiveTyped(self))
            .vortex_expect("Array does not have DType::Primitive")
    }

    /// Downcasts the array for decimal-specific behavior.
    pub fn as_decimal_typed(&self) -> DecimalTyped {
        matches!(self.dtype(), DType::Decimal(..))
            .then(|| DecimalTyped(self))
            .vortex_expect("Array does not have DType::Decimal")
    }

    /// Downcasts the array for utf8-specific behavior.
    pub fn as_utf8_typed(&self) -> Utf8Typed {
        matches!(self.dtype(), DType::Utf8(..))
            .then(|| Utf8Typed(self))
            .vortex_expect("Array does not have DType::Utf8")
    }

    /// Downcasts the array for binary-specific behavior.
    pub fn as_binary_typed(&self) -> BinaryTyped {
        matches!(self.dtype(), DType::Binary(..))
            .then(|| BinaryTyped(self))
            .vortex_expect("Array does not have DType::Binary")
    }

    /// Downcasts the array for struct-specific behavior.
    pub fn as_struct_typed(&self) -> StructTyped {
        matches!(self.dtype(), DType::Struct(..))
            .then(|| StructTyped(self))
            .vortex_expect("Array does not have DType::Struct")
    }

    /// Downcasts the array for list-specific behavior.
    pub fn as_list_typed(&self) -> ListTyped {
        matches!(self.dtype(), DType::List(..))
            .then(|| ListTyped(self))
            .vortex_expect("Array does not have DType::List")
    }

    /// Downcasts the array for extension-specific behavior.
    pub fn as_extension_typed(&self) -> ExtensionTyped {
        matches!(self.dtype(), DType::Extension(..))
            .then(|| ExtensionTyped(self))
            .vortex_expect("Array does not have DType::Extension")
    }
}

#[allow(dead_code)]
pub struct NullTyped<'a>(&'a dyn Array);

pub struct BoolTyped<'a>(&'a dyn Array);

impl BoolTyped<'_> {
    pub fn true_count(&self) -> VortexResult<usize> {
        let true_count = sum(self.0)?;
        Ok(true_count
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("true count should never overflow usize")
            .vortex_expect("true count should never be null"))
    }
}

pub struct PrimitiveTyped<'a>(&'a dyn Array);

impl PrimitiveTyped<'_> {
    fn ptype(&self) -> PType {
        let DType::Primitive(ptype, _) = self.0.dtype() else {
            vortex_panic!("Expected Primitive DType")
        };
        *ptype
    }

    /// Return the primitive value at the given index.
    fn value(&self, idx: usize) -> Option<PValue> {
        self.0
            .is_valid(idx)
            .vortex_expect("is valid")
            .then(|| self.value_unchecked(idx))
    }

    /// Return the primitive value at the given index, ignoring nullability.
    fn value_unchecked(&self, idx: usize) -> PValue {
        self.0
            .scalar_at(idx)
            .vortex_expect("scalar at index")
            .as_primitive()
            .pvalue()
            .unwrap_or_else(|| PValue::zero(self.ptype()))
    }
}

impl IndexOrd<Option<PValue>> for PrimitiveTyped<'_> {
    fn index_cmp(&self, idx: usize, elem: &Option<PValue>) -> Option<Ordering> {
        self.value(idx).partial_cmp(elem)
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

// TODO(ngates): add generics to the `value` function and implement this over T.
impl IndexOrd<PValue> for PrimitiveTyped<'_> {
    fn index_cmp(&self, idx: usize, elem: &PValue) -> Option<Ordering> {
        assert!(self.0.all_valid().vortex_expect("all valid"));
        self.value_unchecked(idx).partial_cmp(elem)
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

#[allow(dead_code)]
pub struct Utf8Typed<'a>(&'a dyn Array);

#[allow(dead_code)]
pub struct BinaryTyped<'a>(&'a dyn Array);

#[allow(dead_code)]
pub struct DecimalTyped<'a>(&'a dyn Array);

pub struct StructTyped<'a>(&'a dyn Array);

impl StructTyped<'_> {
    pub fn names(&self) -> &FieldNames {
        let DType::Struct(st, _) = self.0.dtype() else {
            unreachable!()
        };
        st.names()
    }

    pub fn dtypes(&self) -> Vec<DType> {
        let DType::Struct(st, _) = self.0.dtype() else {
            unreachable!()
        };
        st.fields().collect()
    }

    pub fn nfields(&self) -> usize {
        self.names().len()
    }
}

#[allow(dead_code)]
pub struct ListTyped<'a>(&'a dyn Array);

pub struct ExtensionTyped<'a>(&'a dyn Array);

impl ExtensionTyped<'_> {
    /// Returns the extension logical [`DType`].
    pub fn ext_dtype(&self) -> &Arc<ExtDType> {
        let DType::Extension(ext_dtype) = self.0.dtype() else {
            vortex_panic!("Expected ExtDType")
        };
        ext_dtype
    }
}
