use std::sync::Arc;

use vortex_dtype::DType;

use crate::variants::{
    BinaryArrayTrait, BoolArrayTrait, DecimalArrayTrait, ExtensionArrayTrait, ListArrayTrait,
    NullArrayTrait, PrimitiveArrayTrait, StructArrayTrait, Utf8ArrayTrait,
};
use crate::{Array, ArrayImpl};

pub trait ArrayVariants {
    /// Downcasts the array for null-specific behavior.
    fn as_null_typed(&self) -> Option<&dyn NullArrayTrait>;

    /// Downcasts the array for bool-specific behavior.
    fn as_bool_typed(&self) -> Option<&dyn BoolArrayTrait>;

    /// Downcasts the array for primitive-specific behavior.
    fn as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait>;

    /// Downcasts the array for decimal-specific behavior.
    fn as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait>;

    /// Downcasts the array for utf8-specific behavior.
    fn as_utf8_typed(&self) -> Option<&dyn Utf8ArrayTrait>;

    /// Downcasts the array for binary-specific behavior.
    fn as_binary_typed(&self) -> Option<&dyn BinaryArrayTrait>;

    /// Downcasts the array for struct-specific behavior.
    fn as_struct_typed(&self) -> Option<&dyn StructArrayTrait>;

    /// Downcasts the array for list-specific behavior.
    fn as_list_typed(&self) -> Option<&dyn ListArrayTrait>;

    /// Downcasts the array for extension-specific behavior.
    fn as_extension_typed(&self) -> Option<&dyn ExtensionArrayTrait>;
}

impl ArrayVariants for Arc<dyn Array> {
    fn as_null_typed(&self) -> Option<&dyn NullArrayTrait> {
        self.as_ref().as_null_typed()
    }

    fn as_bool_typed(&self) -> Option<&dyn BoolArrayTrait> {
        self.as_ref().as_bool_typed()
    }

    fn as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        self.as_ref().as_primitive_typed()
    }

    fn as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait> {
        self.as_ref().as_decimal_typed()
    }

    fn as_utf8_typed(&self) -> Option<&dyn Utf8ArrayTrait> {
        self.as_ref().as_utf8_typed()
    }

    fn as_binary_typed(&self) -> Option<&dyn BinaryArrayTrait> {
        self.as_ref().as_binary_typed()
    }

    fn as_struct_typed(&self) -> Option<&dyn StructArrayTrait> {
        self.as_ref().as_struct_typed()
    }

    fn as_list_typed(&self) -> Option<&dyn ListArrayTrait> {
        self.as_ref().as_list_typed()
    }

    fn as_extension_typed(&self) -> Option<&dyn ExtensionArrayTrait> {
        self.as_ref().as_extension_typed()
    }
}

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

    /// Downcasts the array for decimal-specific behavior.
    fn _as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait> {
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

impl<A: ArrayImpl> ArrayVariants for A {
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

    /// Downcasts the array for decimal-specific behavior.
    fn as_decimal_typed(&self) -> Option<&dyn DecimalArrayTrait> {
        matches!(self.dtype(), DType::Decimal(..))
            .then(|| ArrayVariantsImpl::_as_decimal_typed(self))
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
