// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex scalar values (`Scalar`).
//!
//! Replaces the hand-written `vx_scalar` box-wrapper and its many typed constructors and
//! accessors. A `VxScalar` represents a single value with an associated `DType`; its value is
//! either null or a `ScalarValue`.
//!
//! Notable ABI mapping differences:
//! - Diplomat auto-generates the destructor, so `vx_scalar_free` is gone. `vx_scalar_clone`
//!   becomes the `clone` method (returns `Box<VxScalar>`; the NULL-handle special case
//!   disappears since Diplomat `&self` is never null).
//! - `vx_scalar_dtype` returned a borrowed `vx_dtype`; `dtype` here returns an owned
//!   `Box<VxDType>` clone (the underlying `Arc<DType>` clone is cheap).
//! - The primitive constructors (`bool`, `u8`..`i64`, `f32`, `f64`) are infallible and map to
//!   named constructors.
//! - `f16` has no portable C ABI, so the C constructor took raw bits (`vx_scalar_new_f16_bits`);
//!   that is preserved as `new_f16_bits`.
//! - `utf8` takes `&DiplomatStr` (Diplomat validates/owns the bytes) instead of a
//!   `*const c_char` + length + `error_out`. `binary` takes a `&[u8]` slice.
//! - The decimal constructors that took 16-/32-byte little-endian buffers
//!   (`new_decimal_i128_le` / `new_decimal_i256_le`) keep that shape but accept `&[u8]`
//!   slices and return `Result`.
//! - The nested constructors (`list`, `fixed_size_list`, `struct`) took raw
//!   `*const *const vx_scalar` arrays plus a borrowed element/struct `vx_dtype`. Here they
//!   take `&[&VxScalar]` slices (children are cloned in) and a `&VxDType` (borrowed), and
//!   return `Result`.
//! - All `error_out` out-parameters become `Result<_, Box<VortexFfiError>>`.

pub use ffi::VxScalar;

#[diplomat::bridge]
pub mod ffi {
    use std::sync::Arc;

    use diplomat_runtime::DiplomatStr;
    use vortex::dtype::DType;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::Nullability;
    use vortex::dtype::half::f16;
    use vortex::dtype::i256;
    use vortex::scalar::DecimalValue;
    use vortex::scalar::Scalar;
    use vortex::scalar::ScalarValue;

    use crate::dtype::ffi::VxDType;
    use crate::error::ffi::VortexFfiError;

    /// A typed scalar value.
    ///
    /// Either null or a `ScalarValue`, interpreted via its associated `DType`. Replaces the C
    /// `vx_scalar` box-wrapper.
    #[diplomat::opaque]
    pub struct VxScalar(pub(crate) Scalar);

    impl VxScalar {
        /// Clone this scalar into a new owned handle.
        pub fn clone(&self) -> Box<VxScalar> {
            Box::new(VxScalar(self.0.clone()))
        }

        /// The data type of this scalar.
        ///
        /// Returns an owned clone (the C ABI returned a borrowed `vx_dtype`).
        #[diplomat::attr(auto, getter)]
        pub fn dtype(&self) -> Box<VxDType> {
            Box::new(VxDType(self.0.dtype().clone().into()))
        }

        /// Whether this scalar is a typed null value.
        #[diplomat::attr(auto, getter)]
        pub fn is_null(&self) -> bool {
            self.0.is_null()
        }

        /// Create a boolean scalar.
        #[diplomat::attr(auto, named_constructor = "bool")]
        pub fn new_bool(value: bool, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::bool(value, Nullability::from(is_nullable))))
        }

        /// Create an unsigned 8-bit integer scalar.
        #[diplomat::attr(auto, named_constructor = "u8")]
        pub fn new_u8(value: u8, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                value,
                Nullability::from(is_nullable),
            )))
        }

        /// Create an unsigned 16-bit integer scalar.
        #[diplomat::attr(auto, named_constructor = "u16")]
        pub fn new_u16(value: u16, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                value,
                Nullability::from(is_nullable),
            )))
        }

        /// Create an unsigned 32-bit integer scalar.
        #[diplomat::attr(auto, named_constructor = "u32")]
        pub fn new_u32(value: u32, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                value,
                Nullability::from(is_nullable),
            )))
        }

        /// Create an unsigned 64-bit integer scalar.
        #[diplomat::attr(auto, named_constructor = "u64")]
        pub fn new_u64(value: u64, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                value,
                Nullability::from(is_nullable),
            )))
        }

        /// Create a signed 8-bit integer scalar.
        #[diplomat::attr(auto, named_constructor = "i8")]
        pub fn new_i8(value: i8, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                value,
                Nullability::from(is_nullable),
            )))
        }

        /// Create a signed 16-bit integer scalar.
        #[diplomat::attr(auto, named_constructor = "i16")]
        pub fn new_i16(value: i16, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                value,
                Nullability::from(is_nullable),
            )))
        }

        /// Create a signed 32-bit integer scalar.
        #[diplomat::attr(auto, named_constructor = "i32")]
        pub fn new_i32(value: i32, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                value,
                Nullability::from(is_nullable),
            )))
        }

        /// Create a signed 64-bit integer scalar.
        #[diplomat::attr(auto, named_constructor = "i64")]
        pub fn new_i64(value: i64, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                value,
                Nullability::from(is_nullable),
            )))
        }

        /// Create a 32-bit floating point scalar.
        #[diplomat::attr(auto, named_constructor = "f32")]
        pub fn new_f32(value: f32, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                value,
                Nullability::from(is_nullable),
            )))
        }

        /// Create a 64-bit floating point scalar.
        #[diplomat::attr(auto, named_constructor = "f64")]
        pub fn new_f64(value: f64, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                value,
                Nullability::from(is_nullable),
            )))
        }

        /// Create a 16-bit floating point scalar from raw half-precision bits.
        ///
        /// C has no portable half-precision ABI, so the value is provided as `u16` bits.
        #[diplomat::attr(auto, named_constructor = "f16_bits")]
        pub fn new_f16_bits(bits: u16, is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::primitive(
                f16::from_bits(bits),
                Nullability::from(is_nullable),
            )))
        }

        /// Create a UTF-8 scalar from the given string.
        #[diplomat::attr(auto, named_constructor = "utf8")]
        pub fn new_utf8(value: &DiplomatStr, is_nullable: bool) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
            let value = std::str::from_utf8(value)
                .map_err(|e| VortexFfiError::new(format!("invalid utf-8: {e}")))?;
            Ok(Box::new(VxScalar(Scalar::utf8(
                value.to_owned(),
                Nullability::from(is_nullable),
            ))))
        }

        /// Create a binary scalar from the given bytes.
        #[diplomat::attr(auto, named_constructor = "binary")]
        pub fn new_binary(value: &[u8], is_nullable: bool) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::binary(
                value.to_vec(),
                Nullability::from(is_nullable),
            )))
        }

        /// Create a typed null scalar with a nullable copy of `dtype`.
        ///
        /// `dtype` is borrowed, not consumed.
        #[diplomat::attr(auto, named_constructor = "null")]
        pub fn new_null(dtype: &VxDType) -> Box<VxScalar> {
            Box::new(VxScalar(Scalar::null(dtype.0.as_nullable())))
        }

        /// Create a decimal scalar from a signed 8-bit unscaled value.
        #[diplomat::attr(auto, named_constructor = "decimal_i8")]
        pub fn new_decimal_i8(
            value: i8,
            precision: u8,
            scale: i8,
            is_nullable: bool,
        ) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
            decimal_scalar(DecimalValue::I8(value), precision, scale, is_nullable)
        }

        /// Create a decimal scalar from a signed 16-bit unscaled value.
        #[diplomat::attr(auto, named_constructor = "decimal_i16")]
        pub fn new_decimal_i16(
            value: i16,
            precision: u8,
            scale: i8,
            is_nullable: bool,
        ) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
            decimal_scalar(DecimalValue::I16(value), precision, scale, is_nullable)
        }

        /// Create a decimal scalar from a signed 32-bit unscaled value.
        #[diplomat::attr(auto, named_constructor = "decimal_i32")]
        pub fn new_decimal_i32(
            value: i32,
            precision: u8,
            scale: i8,
            is_nullable: bool,
        ) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
            decimal_scalar(DecimalValue::I32(value), precision, scale, is_nullable)
        }

        /// Create a decimal scalar from a signed 64-bit unscaled value.
        #[diplomat::attr(auto, named_constructor = "decimal_i64")]
        pub fn new_decimal_i64(
            value: i64,
            precision: u8,
            scale: i8,
            is_nullable: bool,
        ) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
            decimal_scalar(DecimalValue::I64(value), precision, scale, is_nullable)
        }

        /// Create a decimal scalar from a 16-byte little-endian signed unscaled value.
        #[diplomat::attr(auto, named_constructor = "decimal_i128_le")]
        pub fn new_decimal_i128_le(
            bytes16: &[u8],
            precision: u8,
            scale: i8,
            is_nullable: bool,
        ) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
            let bytes = fixed_bytes::<16>(bytes16, "decimal i128")?;
            decimal_scalar(
                DecimalValue::I128(i128::from_le_bytes(bytes)),
                precision,
                scale,
                is_nullable,
            )
        }

        /// Create a decimal scalar from a 32-byte little-endian signed unscaled value.
        #[diplomat::attr(auto, named_constructor = "decimal_i256_le")]
        pub fn new_decimal_i256_le(
            bytes32: &[u8],
            precision: u8,
            scale: i8,
            is_nullable: bool,
        ) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
            let bytes = fixed_bytes::<32>(bytes32, "decimal i256")?;
            decimal_scalar(
                DecimalValue::I256(i256::from_le_bytes(bytes)),
                precision,
                scale,
                is_nullable,
            )
        }

        /// Create a list scalar.
        ///
        /// `element_dtype` is borrowed; each child in `elements` is cloned into the list.
        #[diplomat::attr(auto, named_constructor = "list")]
        pub fn new_list(
            element_dtype: &VxDType,
            elements: &[&VxScalar],
            is_nullable: bool,
        ) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
            let dtype = DType::List(
                Arc::new(element_dtype.0.as_ref().clone()),
                Nullability::from(is_nullable),
            );
            let values = child_values(elements);
            Ok(Box::new(VxScalar(Scalar::try_new(
                dtype,
                Some(ScalarValue::Tuple(values)),
            )?)))
        }

        /// Create a fixed-size list scalar.
        ///
        /// `element_dtype` is borrowed; each child in `elements` is cloned in. The child count
        /// becomes the list width and must fit in a `u32`.
        #[diplomat::attr(auto, named_constructor = "fixed_size_list")]
        pub fn new_fixed_size_list(
            element_dtype: &VxDType,
            elements: &[&VxScalar],
            is_nullable: bool,
        ) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
            let size = u32::try_from(elements.len()).map_err(|_| {
                VortexFfiError::new(format!(
                    "fixed-size list length {} exceeds u32::MAX",
                    elements.len()
                ))
            })?;
            let dtype = DType::FixedSizeList(
                Arc::new(element_dtype.0.as_ref().clone()),
                size,
                Nullability::from(is_nullable),
            );
            let values = child_values(elements);
            Ok(Box::new(VxScalar(Scalar::try_new(
                dtype,
                Some(ScalarValue::Tuple(values)),
            )?)))
        }

        /// Create a struct scalar.
        ///
        /// `struct_dtype` is borrowed; each field in `fields` is cloned in. Field count and
        /// field logical types are validated against `struct_dtype`.
        #[diplomat::attr(auto, named_constructor = "struct")]
        pub fn new_struct(
            struct_dtype: &VxDType,
            fields: &[&VxScalar],
        ) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
            let values = child_values(fields);
            Ok(Box::new(VxScalar(Scalar::try_new(
                struct_dtype.0.as_ref().clone(),
                Some(ScalarValue::Tuple(values)),
            )?)))
        }
    }

    /// Build a decimal scalar from a [`DecimalValue`] and decimal metadata.
    fn decimal_scalar(
        value: DecimalValue,
        precision: u8,
        scale: i8,
        is_nullable: bool,
    ) -> Result<Box<VxScalar>, Box<VortexFfiError>> {
        let decimal_dtype = DecimalDType::try_new(precision, scale)?;
        Ok(Box::new(VxScalar(Scalar::try_new(
            DType::Decimal(decimal_dtype, Nullability::from(is_nullable)),
            Some(ScalarValue::Decimal(value)),
        )?)))
    }

    /// Clone borrowed child scalars into owned `ScalarValue`s for tuple construction.
    fn child_values(children: &[&VxScalar]) -> Vec<Option<ScalarValue>> {
        children
            .iter()
            .map(|child| child.0.clone().into_value())
            .collect()
    }

    /// Copy a fixed-width little-endian byte buffer, validating its length.
    fn fixed_bytes<const N: usize>(
        bytes: &[u8],
        label: &str,
    ) -> Result<[u8; N], Box<VortexFfiError>> {
        if bytes.len() != N {
            return Err(VortexFfiError::new(format!(
                "{label} expects {N} bytes, got {}",
                bytes.len()
            )));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(bytes);
        Ok(out)
    }
}
