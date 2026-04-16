// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines extension functionality specific to each Vortex DType.
use std::cmp::Ordering;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::aggregate_fn::fns::sum::sum;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::FieldNames;
use crate::dtype::PType;
use crate::dtype::extension::ExtDTypeRef;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::search_sorted::IndexOrd;

impl ArrayRef {
    /// Downcasts the array for null-specific behavior.
    pub fn as_null_typed(&self) -> NullTyped<'_> {
        matches!(self.dtype(), DType::Null)
            .then(|| NullTyped(self))
            .vortex_expect("Array does not have DType::Null")
    }

    /// Downcasts the array for bool-specific behavior.
    pub fn as_bool_typed(&self) -> BoolTyped<'_> {
        matches!(self.dtype(), DType::Bool(..))
            .then(|| BoolTyped(self))
            .vortex_expect("Array does not have DType::Bool")
    }

    /// Downcasts the array for primitive-specific behavior.
    pub fn as_primitive_typed(&self) -> PrimitiveTyped<'_> {
        matches!(self.dtype(), DType::Primitive(..))
            .then(|| PrimitiveTyped(self))
            .vortex_expect("Array does not have DType::Primitive")
    }

    /// Downcasts the array for decimal-specific behavior.
    pub fn as_decimal_typed(&self) -> DecimalTyped<'_> {
        matches!(self.dtype(), DType::Decimal(..))
            .then(|| DecimalTyped(self))
            .vortex_expect("Array does not have DType::Decimal")
    }

    /// Downcasts the array for utf8-specific behavior.
    pub fn as_utf8_typed(&self) -> Utf8Typed<'_> {
        matches!(self.dtype(), DType::Utf8(..))
            .then(|| Utf8Typed(self))
            .vortex_expect("Array does not have DType::Utf8")
    }

    /// Downcasts the array for binary-specific behavior.
    pub fn as_binary_typed(&self) -> BinaryTyped<'_> {
        matches!(self.dtype(), DType::Binary(..))
            .then(|| BinaryTyped(self))
            .vortex_expect("Array does not have DType::Binary")
    }

    /// Downcasts the array for struct-specific behavior.
    pub fn as_struct_typed(&self) -> StructTyped<'_> {
        matches!(self.dtype(), DType::Struct(..))
            .then(|| StructTyped(self))
            .vortex_expect("Array does not have DType::Struct")
    }

    /// Downcasts the array for list-specific behavior.
    pub fn as_list_typed(&self) -> ListTyped<'_> {
        matches!(self.dtype(), DType::List(..))
            .then(|| ListTyped(self))
            .vortex_expect("Array does not have DType::List")
    }

    /// Downcasts the array for extension-specific behavior.
    pub fn as_extension_typed(&self) -> ExtensionTyped<'_> {
        matches!(self.dtype(), DType::Extension(..))
            .then(|| ExtensionTyped(self))
            .vortex_expect("Array does not have DType::Extension")
    }

    pub fn try_to_mask_fill_null_false(&self, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
        if !matches!(self.dtype(), DType::Bool(_)) {
            vortex_bail!("mask must be bool array, has dtype {}", self.dtype());
        }

        // Convert nulls to false first in case this can be done cheaply by the encoding.
        let array = self
            .clone()
            .fill_null(Scalar::bool(false, self.dtype().nullability()))?;

        Ok(array
            .execute::<BoolArray>(ctx)?
            .to_mask_fill_null_false(ctx))
    }
}

#[expect(dead_code)]
pub struct NullTyped<'a>(&'a ArrayRef);

pub struct BoolTyped<'a>(&'a ArrayRef);

impl BoolTyped<'_> {
    pub fn true_count(&self) -> VortexResult<usize> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let true_count = sum(self.0, &mut ctx)?;
        Ok(true_count
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("true count should never be null"))
    }
}

pub struct PrimitiveTyped<'a>(&'a ArrayRef);

impl PrimitiveTyped<'_> {
    pub fn ptype(&self) -> PType {
        let DType::Primitive(ptype, _) = self.0.dtype() else {
            vortex_panic!("Expected Primitive DType")
        };
        *ptype
    }

    /// Return the primitive value at the given index.
    pub fn value(&self, idx: usize) -> VortexResult<Option<PValue>> {
        self.0
            .is_valid(idx)?
            .then(|| self.value_unchecked(idx))
            .transpose()
    }

    /// Return the primitive value at the given index, ignoring nullability.
    pub fn value_unchecked(&self, idx: usize) -> VortexResult<PValue> {
        Ok(self
            .0
            .scalar_at(idx)?
            .as_primitive()
            .pvalue()
            .unwrap_or_else(|| PValue::zero(&self.ptype())))
    }
}

impl IndexOrd<Option<PValue>> for PrimitiveTyped<'_> {
    fn index_cmp(&self, idx: usize, elem: &Option<PValue>) -> VortexResult<Option<Ordering>> {
        let value = self.value(idx)?;
        Ok(value.partial_cmp(elem))
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

// TODO(ngates): add generics to the `value` function and implement this over T.
impl IndexOrd<PValue> for PrimitiveTyped<'_> {
    fn index_cmp(&self, idx: usize, elem: &PValue) -> VortexResult<Option<Ordering>> {
        assert!(self.0.all_valid()?);
        let value = self.value_unchecked(idx)?;
        Ok(value.partial_cmp(elem))
    }

    fn index_len(&self) -> usize {
        self.0.len()
    }
}

#[expect(dead_code)]
pub struct Utf8Typed<'a>(&'a ArrayRef);

#[expect(dead_code)]
pub struct BinaryTyped<'a>(&'a ArrayRef);

#[expect(dead_code)]
pub struct DecimalTyped<'a>(&'a ArrayRef);

pub struct StructTyped<'a>(&'a ArrayRef);

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

#[expect(dead_code)]
pub struct ListTyped<'a>(&'a ArrayRef);

pub struct ExtensionTyped<'a>(&'a ArrayRef);

impl ExtensionTyped<'_> {
    /// Returns the extension logical [`DType`].
    pub fn ext_dtype(&self) -> &ExtDTypeRef {
        let DType::Extension(ext_dtype) = self.0.dtype() else {
            vortex_panic!("Expected ExtDType")
        };
        ext_dtype
    }
}
