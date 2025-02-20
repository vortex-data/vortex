use std::sync::Arc;

use vortex_dtype::DType;

use crate::variants::{
    BinaryArrayTrait, BoolArrayTrait, ExtensionArrayTrait, ListArrayTrait, NullArrayTrait,
    PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::Array;

pub trait ArrayVariants: Array {
    /// Downcasts the array for null-specific behavior.
    fn as_null_typed(&self) -> Option<&dyn NullArrayTrait> {
        matches!(self.dtype(), DType::Null)
            .then(|| ArrayVariantsImpl::_as_null_typed(self))
            .flatten()
    }

    /// Downcasts the array for bool-specific behavior.
    fn as_bool_typed(&self) -> Option<&dyn BoolArrayTrait> {
        matches!(self.dtype(), DType::Bool(..))
            .then(|| ArrayVariantsImpl::_as_bool_typed(self))
            .flatten()
    }

    /// Downcasts the array for primitive-specific behavior.
    fn as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        matches!(self.dtype(), DType::Primitive(..))
            .then(|| ArrayVariantsImpl::_as_primitive_typed(self))
            .flatten()
    }

    /// Downcasts the array for utf8-specific behavior.
    fn as_utf8_typed(&self) -> Option<&dyn Utf8ArrayTrait> {
        matches!(self.dtype(), DType::Utf8(..))
            .then(|| ArrayVariantsImpl::_as_utf8_typed(self))
            .flatten()
    }

    /// Downcasts the array for binary-specific behavior.
    fn as_binary_typed(&self) -> Option<&dyn BinaryArrayTrait> {
        matches!(self.dtype(), DType::Binary(..))
            .then(|| ArrayVariantsImpl::_as_binary_typed(self))
            .flatten()
    }

    /// Downcasts the array for struct-specific behavior.
    fn as_struct_typed(&self) -> Option<&dyn StructArrayTrait> {
        matches!(self.dtype(), DType::Struct(..))
            .then(|| ArrayVariantsImpl::_as_struct_typed(self))
            .flatten()
    }

    /// Downcasts the array for list-specific behavior.
    fn as_list_typed(&self) -> Option<&dyn ListArrayTrait> {
        matches!(self.dtype(), DType::List(..))
            .then(|| ArrayVariantsImpl::_as_list_typed(self))
            .flatten()
    }

    /// Downcasts the array for extension-specific behavior.
    fn as_extension_typed(&self) -> Option<&dyn ExtensionArrayTrait> {
        matches!(self.dtype(), DType::Extension(..))
            .then(|| ArrayVariantsImpl::_as_extension_typed(self))
            .flatten()
    }
}

impl<A: Array> ArrayVariants for A {}

/// Implementation trait for downcasting to type-specific traits.
pub trait ArrayVariantsImpl {
    /// Downcasts the array for null-specific behavior.
    fn _as_null_typed(&self) -> Option<&dyn NullArrayTrait> {
        None
    }

    /// Downcasts the array for bool-specific behavior.
    fn _as_bool_typed(&self) -> Option<&dyn BoolArrayTrait> {
        None
    }

    /// Downcasts the array for primitive-specific behavior.
    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        None
    }

    /// Downcasts the array for utf8-specific behavior.
    fn _as_utf8_typed(&self) -> Option<&dyn Utf8ArrayTrait> {
        None
    }

    /// Downcasts the array for binary-specific behavior.
    fn _as_binary_typed(&self) -> Option<&dyn BinaryArrayTrait> {
        None
    }

    /// Downcasts the array for struct-specific behavior.
    fn _as_struct_typed(&self) -> Option<&dyn StructArrayTrait> {
        None
    }

    /// Downcasts the array for list-specific behavior.
    fn _as_list_typed(&self) -> Option<&dyn ListArrayTrait> {
        None
    }

    /// Downcasts the array for extension-specific behavior.
    fn _as_extension_typed(&self) -> Option<&dyn ExtensionArrayTrait> {
        None
    }
}
