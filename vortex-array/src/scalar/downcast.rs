// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar downcasting methods to typed views.

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::scalar::BinaryScalar;
use crate::scalar::BoolScalar;
use crate::scalar::DecimalScalar;
use crate::scalar::DecimalValue;
use crate::scalar::ExtScalar;
use crate::scalar::ListScalar;
use crate::scalar::PValue;
use crate::scalar::PrimitiveScalar;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::scalar::StructScalar;
use crate::scalar::Utf8Scalar;
use crate::scalar::VariantScalar;

impl Scalar {
    /// Returns a view of the scalar as a boolean scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar does not have a [`Bool`](crate::dtype::DType::Bool) type.
    pub fn as_bool(&self) -> BoolScalar<'_> {
        self.as_bool_opt()
            .vortex_expect("Failed to convert scalar to bool")
    }

    /// Returns a view of the scalar as a boolean scalar if it has a boolean type.
    pub fn as_bool_opt(&self) -> Option<BoolScalar<'_>> {
        BoolScalar::try_new(self.dtype(), self.value()).ok()
    }

    /// Returns a view of the scalar as a primitive scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar does not have a [`Primitive`](crate::dtype::DType::Primitive) type.
    pub fn as_primitive(&self) -> PrimitiveScalar<'_> {
        self.as_primitive_opt()
            .vortex_expect("Failed to convert scalar to primitive")
    }

    /// Returns a view of the scalar as a primitive scalar if it has a primitive type.
    pub fn as_primitive_opt(&self) -> Option<PrimitiveScalar<'_>> {
        PrimitiveScalar::try_new(self.dtype(), self.value()).ok()
    }

    /// Returns a view of the scalar as a decimal scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar does not have a [`Decimal`](crate::dtype::DType::Decimal) type.
    pub fn as_decimal(&self) -> DecimalScalar<'_> {
        self.as_decimal_opt()
            .vortex_expect("Failed to convert scalar to decimal")
    }

    /// Returns a view of the scalar as a decimal scalar if it has a decimal type.
    pub fn as_decimal_opt(&self) -> Option<DecimalScalar<'_>> {
        DecimalScalar::try_new(self.dtype(), self.value()).ok()
    }

    /// Returns a view of the scalar as a UTF-8 string scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar does not have a [`Utf8`](crate::dtype::DType::Utf8) type.
    pub fn as_utf8(&self) -> Utf8Scalar<'_> {
        self.as_utf8_opt()
            .vortex_expect("Failed to convert scalar to utf8")
    }

    /// Returns a view of the scalar as a UTF-8 string scalar if it has a UTF-8 type.
    pub fn as_utf8_opt(&self) -> Option<Utf8Scalar<'_>> {
        Utf8Scalar::try_new(self.dtype(), self.value()).ok()
    }

    /// Returns a view of the scalar as a binary scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar does not have a [`Binary`](crate::dtype::DType::Binary) type.
    pub fn as_binary(&self) -> BinaryScalar<'_> {
        self.as_binary_opt()
            .vortex_expect("Failed to convert scalar to binary")
    }

    /// Returns a view of the scalar as a binary scalar if it has a binary type.
    pub fn as_binary_opt(&self) -> Option<BinaryScalar<'_>> {
        BinaryScalar::try_new(self.dtype(), self.value()).ok()
    }

    /// Returns a view of the scalar as a struct scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar does not have a [`Struct`](crate::dtype::DType::Struct) type.
    pub fn as_struct(&self) -> StructScalar<'_> {
        self.as_struct_opt()
            .vortex_expect("Failed to convert scalar to struct")
    }

    /// Returns a view of the scalar as a struct scalar if it has a struct type.
    pub fn as_struct_opt(&self) -> Option<StructScalar<'_>> {
        StructScalar::try_new(self.dtype(), self.value()).ok()
    }

    /// Returns a view of the scalar as a list scalar.
    ///
    /// Note that we use [`ListScalar`] to represent **both** [`List`](crate::dtype::DType::List)
    /// and [`FixedSizeList`](crate::dtype::DType::FixedSizeList).
    ///
    /// # Panics
    ///
    /// Panics if the scalar does not have a [`List`](crate::dtype::DType::List) or
    /// [`FixedSizeList`](crate::dtype::DType::FixedSizeList) type.
    pub fn as_list(&self) -> ListScalar<'_> {
        self.as_list_opt()
            .vortex_expect("Failed to convert scalar to list")
    }

    /// Returns a view of the scalar as a list scalar if it has a list type.
    ///
    /// Note that we use [`ListScalar`] to represent **both** [`List`](crate::dtype::DType::List)
    /// and [`FixedSizeList`](crate::dtype::DType::FixedSizeList).
    pub fn as_list_opt(&self) -> Option<ListScalar<'_>> {
        ListScalar::try_new(self.dtype(), self.value()).ok()
    }

    /// Returns a view of the scalar as an extension scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar does not have a [`Extension`](crate::dtype::DType::Extension) type.
    pub fn as_extension(&self) -> ExtScalar<'_> {
        self.as_extension_opt()
            .vortex_expect("Failed to convert scalar to extension")
    }

    /// Returns a view of the scalar as an extension scalar if it has an extension type.
    pub fn as_extension_opt(&self) -> Option<ExtScalar<'_>> {
        if !self.dtype().is_extension() {
            return None;
        }

        // SAFETY: Because we are a valid Scalar, we have already validated that the value is valid
        // for this extension type.
        Some(ExtScalar::new_unchecked(self.dtype(), self.value()))
    }

    /// Returns a view of the scalar as a variant scalar.
    ///
    /// # Panics
    ///
    /// Panics if the scalar does not have a [`Variant`](crate::dtype::DType::Variant) type.
    pub fn as_variant(&self) -> VariantScalar<'_> {
        self.as_variant_opt()
            .vortex_expect("Failed to convert scalar to variant")
    }

    /// Returns a view of the scalar as a variant scalar if it has a variant type.
    pub fn as_variant_opt(&self) -> Option<VariantScalar<'_>> {
        VariantScalar::try_new(self.dtype(), self.value()).ok()
    }
}

impl ScalarValue {
    /// Returns the boolean value, panicking if the value is not a [`Bool`](ScalarValue::Bool).
    pub fn as_bool(&self) -> bool {
        match self {
            ScalarValue::Bool(b) => *b,
            _ => vortex_panic!("ScalarValue is not a Bool"),
        }
    }

    /// Returns the primitive value, panicking if the value is not a
    /// [`Primitive`](ScalarValue::Primitive).
    pub fn as_primitive(&self) -> &PValue {
        match self {
            ScalarValue::Primitive(p) => p,
            _ => vortex_panic!("ScalarValue is not a Primitive"),
        }
    }

    /// Returns the decimal value, panicking if the value is not a
    /// [`Decimal`](ScalarValue::Decimal).
    pub fn as_decimal(&self) -> &DecimalValue {
        match self {
            ScalarValue::Decimal(d) => d,
            _ => vortex_panic!("ScalarValue is not a Decimal"),
        }
    }

    /// Returns the UTF-8 string value, panicking if the value is not a [`Utf8`](ScalarValue::Utf8).
    pub fn as_utf8(&self) -> &BufferString {
        match self {
            ScalarValue::Utf8(s) => s,
            _ => vortex_panic!("ScalarValue is not a Utf8"),
        }
    }

    /// Returns the binary value, panicking if the value is not a [`Binary`](ScalarValue::Binary).
    pub fn as_binary(&self) -> &ByteBuffer {
        match self {
            ScalarValue::Binary(b) => b,
            _ => vortex_panic!("ScalarValue is not a Binary"),
        }
    }

    /// Returns the tuple elements, panicking if the value is not a [`Tuple`](ScalarValue::Tuple).
    pub fn as_list(&self) -> &[Option<ScalarValue>] {
        match self {
            ScalarValue::Tuple(elements) => elements,
            _ => vortex_panic!("ScalarValue is not a Tuple"),
        }
    }

    /// Returns the boolean value, panicking if the value is not a [`Bool`](ScalarValue::Bool).
    pub fn into_bool(self) -> bool {
        match self {
            ScalarValue::Bool(b) => b,
            _ => vortex_panic!("ScalarValue is not a Bool"),
        }
    }

    /// Returns the primitive value, panicking if the value is not a
    /// [`Primitive`](ScalarValue::Primitive).
    pub fn into_primitive(self) -> PValue {
        match self {
            ScalarValue::Primitive(p) => p,
            _ => vortex_panic!("ScalarValue is not a Primitive"),
        }
    }

    /// Returns the decimal value, panicking if the value is not a
    /// [`Decimal`](ScalarValue::Decimal).
    pub fn into_decimal(self) -> DecimalValue {
        match self {
            ScalarValue::Decimal(d) => d,
            _ => vortex_panic!("ScalarValue is not a Decimal"),
        }
    }

    /// Returns the UTF-8 string value, panicking if the value is not a [`Utf8`](ScalarValue::Utf8).
    pub fn into_utf8(self) -> BufferString {
        match self {
            ScalarValue::Utf8(s) => s,
            _ => vortex_panic!("ScalarValue is not a Utf8"),
        }
    }

    /// Returns the binary value, panicking if the value is not a [`Binary`](ScalarValue::Binary).
    pub fn into_binary(self) -> ByteBuffer {
        match self {
            ScalarValue::Binary(b) => b,
            _ => vortex_panic!("ScalarValue is not a Binary"),
        }
    }

    /// Returns the tuple elements, panicking if the value is not a [`Tuple`](ScalarValue::Tuple).
    pub fn into_list(self) -> Vec<Option<ScalarValue>> {
        match self {
            ScalarValue::Tuple(elements) => elements,
            _ => vortex_panic!("ScalarValue is not a Tuple"),
        }
    }

    /// Returns the row-specific scalar wrapped by a variant, panicking if the value is not a
    /// variant.
    pub fn as_variant(&self) -> &Scalar {
        match self {
            ScalarValue::Variant(value) => value,
            _ => vortex_panic!("ScalarValue is not a Variant"),
        }
    }

    /// Returns the row-specific scalar wrapped by a variant, panicking if the value is not a
    /// variant.
    pub fn into_variant(self) -> Scalar {
        match self {
            ScalarValue::Variant(value) => *value,
            _ => vortex_panic!("ScalarValue is not a Variant"),
        }
    }
}
