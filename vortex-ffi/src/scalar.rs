// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;
use std::slice;
use std::sync::Arc;

use paste::paste;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::Nullability;
use vortex::dtype::half::f16;
use vortex::dtype::i256;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;

use crate::box_wrapper;
use crate::dtype::vx_dtype;
use crate::error::try_or;
use crate::error::vx_error;
use crate::string::vx_view;

box_wrapper!(
    /// A vx_scalar is a single value with an associated vx_dtype.
    ///
    /// Scalar value may be Null is vx_dtype is nullable.
    /// One example where you can get a Null scalar is vx_array_get_scalar
    /// where the element at some index is invalid/null.
    Scalar,
    vx_scalar
);

/// Clone a vx_scalar
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_clone(scalar: *const vx_scalar) -> *mut vx_scalar {
    vx_scalar::new(vx_scalar::as_ref(scalar).clone())
}

/// Return scalar's dtype.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_dtype(scalar: *const vx_scalar) -> *const vx_dtype {
    vx_dtype::new(vx_scalar::as_ref(scalar).dtype().clone())
}

/// Return whether scalar is a typed Null value.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_is_null(scalar: *const vx_scalar) -> bool {
    vx_scalar::as_ref(scalar).is_null()
}

/// Create a boolean scalar.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_new_bool(
    value: bool,
    is_nullable: bool,
) -> *mut vx_scalar {
    vx_scalar::new(Scalar::bool(value, Nullability::from(is_nullable)))
}

/// Return the boolean value stored in the scalar.
///
/// Panics if the scalar is not a Bool scalar, or is null.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_get_bool(scalar: *const vx_scalar) -> bool {
    vx_scalar::as_ref(scalar)
        .as_bool()
        .value()
        .vortex_expect("scalar is null or not a bool")
}

macro_rules! scalar_primitive {
    ($ptype:ident) => {
        paste! {
            #[doc = concat!(" Create a ", stringify!($ptype), " scalar.")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_scalar_new_ $ptype>](
                value: $ptype,
                is_nullable: bool,
            ) -> *mut vx_scalar {
                vx_scalar::new(Scalar::primitive(value, Nullability::from(is_nullable)))
            }

            #[doc = concat!(" Return ", stringify!($ptype), " value stored in scalar.")]
            ///
            /// Panics if scalar is not a primitive scalar of this type or is null.
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_scalar_get_ $ptype>](
                scalar: *const vx_scalar,
            ) -> $ptype {
                vx_scalar::as_ref(scalar)
                    .as_primitive()
                    .typed_value::<$ptype>()
                    .vortex_expect(concat!("scalar is null or not a ", stringify!($ptype)))
            }
        }
    };
}

scalar_primitive!(u8);
scalar_primitive!(u16);
scalar_primitive!(u32);
scalar_primitive!(u64);
scalar_primitive!(i8);
scalar_primitive!(i16);
scalar_primitive!(i32);
scalar_primitive!(i64);
scalar_primitive!(f32);
scalar_primitive!(f64);

/// Create a 16-bit floating point scalar.
/// The value is read from raw uint16_t.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_new_f16_bits(
    bits: u16,
    is_nullable: bool,
) -> *mut vx_scalar {
    vx_scalar::new(Scalar::primitive(
        f16::from_bits(bits),
        Nullability::from(is_nullable),
    ))
}

/// Return 16-bit floating point value stored in scalar.
/// The value is read into raw uint16_t.
///
/// Panics if scalar is not a primitive scalar of this type or is null.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_get_f16_bits(scalar: *const vx_scalar) -> u16 {
    let value = vx_scalar::as_ref(scalar)
        .as_primitive()
        .typed_value::<f16>()
        .vortex_expect("scalar is null or not a u16");
    f16::to_bits(value)
}

/// Create a UTF-8 scalar.
///
/// "value" bytes are copied into scalar.
/// Errors on invalid UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_new_utf8(
    value: vx_view,
    is_nullable: bool,
    err: *mut *mut vx_error,
) -> *mut vx_scalar {
    try_or(err, ptr::null_mut(), || {
        let value = unsafe { value.as_str() }?;
        Ok(vx_scalar::new(Scalar::utf8(
            value.to_owned(),
            Nullability::from(is_nullable),
        )))
    })
}

/// Create a binary scalar.
///
/// Byte range is copied into the scalar.
///
/// NULL "ptr" is allowed only when len == 0.
///
/// Returns NULL and sets "err" on error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_new_binary(
    ptr: *const u8,
    len: usize,
    is_nullable: bool,
    err: *mut *mut vx_error,
) -> *mut vx_scalar {
    try_or(err, ptr::null_mut(), || {
        let bytes = bytes_from_raw(ptr, len, "binary")?;
        Ok(vx_scalar::new(Scalar::binary(
            bytes.to_vec(),
            Nullability::from(is_nullable),
        )))
    })
}

/// Return UTF-8 string stored in scalar.
///
/// Returned view borrows the scalar and is valid as long as "scalar" is valid.
///
/// Panics if scalar is not a Utf8 scalar, or is null.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_get_utf8(scalar: *const vx_scalar) -> vx_view {
    let value = vx_scalar::as_ref(scalar)
        .as_utf8()
        .value()
        .vortex_expect("scalar is null or not a utf8");
    vx_view::from_str(value.as_str())
}

/// Return binary bytes stored in the scalar.
///
/// Returned view borrows scalar and is valid as long as "scalar" is valid.
///
/// Panics if scalar is not a Binary scalar, or is null.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_get_binary(scalar: *const vx_scalar) -> vx_view {
    let value = vx_scalar::as_ref(scalar)
        .as_binary()
        .value()
        .vortex_expect("scalar is null or not a binary");
    vx_view::from_bytes(value.as_slice())
}

/// Create a typed null scalar.
///
/// Returned scalar uses a nullable copy of that logical type, regardless of
/// the input type's top-level nullability.
///
/// Returns NULL and sets "err" on error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_new_null(
    dtype: *const vx_dtype,
    err: *mut *mut vx_error,
) -> *mut vx_scalar {
    try_or(err, ptr::null_mut(), || {
        Ok(vx_scalar::new(Scalar::null(
            vx_dtype::as_ref(dtype).as_nullable(),
        )))
    })
}

macro_rules! scalar_decimal {
    ($int:ident, $variant:ident) => {
        paste! {
            #[doc = concat!(" Create a decimal scalar from a signed ", stringify!($int), " unscaled value.")]
            ///
            /// Returns NULL and sets "err" on error.
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_scalar_new_decimal_ $int>](
                value: $int,
                precision: u8,
                scale: i8,
                is_nullable: bool,
                err: *mut *mut vx_error,
            ) -> *mut vx_scalar {
                try_or(err, ptr::null_mut(), || {
                    decimal_scalar_from_value(
                        DecimalValue::$variant(value),
                        precision,
                        scale,
                        is_nullable,
                    )
                })
            }

            #[doc = concat!(" Return the unscaled ", stringify!($int), " value of a decimal scalar.")]
            ///
            /// Panics if the scalar is not a decimal scalar, is null, or the
            #[doc = concat!(" unscaled value does not fit in ", stringify!($int), ".")]
            #[unsafe(no_mangle)]
            pub unsafe extern "C-unwind" fn [<vx_scalar_get_decimal_ $int>](
                scalar: *const vx_scalar,
            ) -> $int {
                vx_scalar::as_ref(scalar)
                    .as_decimal()
                    .decimal_value()
                    .and_then(|value| value.cast::<$int>())
                    .vortex_expect(concat!(
                        "scalar is null or its decimal value does not fit in ",
                        stringify!($int)
                    ))
            }
        }
    };
}

scalar_decimal!(i8, I8);
scalar_decimal!(i16, I16);
scalar_decimal!(i32, I32);
scalar_decimal!(i64, I64);

/// Create a decimal scalar.
///
/// The unscaled value is read from a 16-byte little-endian signed integer
/// buffer.
///
/// Returns NULL and sets "err" on error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_new_decimal_i128_le(
    bytes16: *const u8,
    precision: u8,
    scale: i8,
    is_nullable: bool,
    err: *mut *mut vx_error,
) -> *mut vx_scalar {
    try_or(err, ptr::null_mut(), || {
        let bytes = fixed_bytes_from_raw::<16>(bytes16, "decimal i128")?;
        decimal_scalar_from_value(
            DecimalValue::I128(i128::from_le_bytes(bytes)),
            precision,
            scale,
            is_nullable,
        )
    })
}

/// Create a decimal scalar.
///
/// The unscaled value is read from a 32-byte little-endian signed integer
/// buffer.
///
/// Returns NULL and sets "err" on error.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_new_decimal_i256_le(
    bytes32: *const u8,
    precision: u8,
    scale: i8,
    is_nullable: bool,
    err: *mut *mut vx_error,
) -> *mut vx_scalar {
    try_or(err, ptr::null_mut(), || {
        let bytes = fixed_bytes_from_raw::<32>(bytes32, "decimal i256")?;
        decimal_scalar_from_value(
            DecimalValue::I256(i256::from_le_bytes(bytes)),
            precision,
            scale,
            is_nullable,
        )
    })
}

/// Create a list scalar.
///
/// NULL "elements" are allowed only if len == 0.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_new_list(
    element_dtype: *const vx_dtype,
    elements: *const *const vx_scalar,
    len: usize,
    is_nullable: bool,
    err: *mut *mut vx_error,
) -> *mut vx_scalar {
    try_or(err, ptr::null_mut(), || {
        let dtype = DType::List(
            Arc::new(vx_dtype::as_ref(element_dtype).clone()),
            Nullability::from(is_nullable),
        );
        let values = scalar_values_from_raw(elements, len)?;
        Ok(vx_scalar::new(Scalar::try_new(
            dtype,
            Some(ScalarValue::Tuple(values)),
        )?))
    })
}

/// Create a fixed-size list scalar.
///
/// NULL "elements" are allowed only if len == 0.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_new_fixed_size_list(
    element_dtype: *const vx_dtype,
    elements: *const *const vx_scalar,
    len: u32,
    is_nullable: bool,
    err: *mut *mut vx_error,
) -> *mut vx_scalar {
    try_or(err, ptr::null_mut(), || {
        let dtype = DType::FixedSizeList(
            Arc::new(vx_dtype::as_ref(element_dtype).clone()),
            len,
            Nullability::from(is_nullable),
        );
        let values = scalar_values_from_raw(elements, len as usize)?;
        Ok(vx_scalar::new(Scalar::try_new(
            dtype,
            Some(ScalarValue::Tuple(values)),
        )?))
    })
}

/// Create a struct scalar.
///
/// NULL "fields" are allowed only if len == 0.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_scalar_new_struct(
    struct_dtype: *const vx_dtype,
    fields: *const *const vx_scalar,
    len: usize,
    err: *mut *mut vx_error,
) -> *mut vx_scalar {
    try_or(err, ptr::null_mut(), || {
        vortex_ensure!(!struct_dtype.is_null(), "struct dtype is null");
        let values = scalar_values_from_raw(fields, len)?;
        Ok(vx_scalar::new(Scalar::try_new(
            vx_dtype::as_ref(struct_dtype).clone(),
            Some(ScalarValue::Tuple(values)),
        )?))
    })
}

fn decimal_scalar_from_value(
    value: DecimalValue,
    precision: u8,
    scale: i8,
    is_nullable: bool,
) -> VortexResult<*mut vx_scalar> {
    let decimal_dtype = DecimalDType::try_new(precision, scale)?;
    Ok(vx_scalar::new(Scalar::try_new(
        DType::Decimal(decimal_dtype, Nullability::from(is_nullable)),
        Some(ScalarValue::Decimal(value)),
    )?))
}

fn scalar_values_from_raw(
    values: *const *const vx_scalar,
    len: usize,
) -> VortexResult<Vec<Option<ScalarValue>>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    vortex_ensure!(!values.is_null(), "scalar pointer array is null");

    unsafe { slice::from_raw_parts(values, len) }
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if value.is_null() {
                vortex_bail!("scalar pointer at index {idx} is null");
            }
            Ok(vx_scalar::as_ref(*value).clone().into_value())
        })
        .collect()
}

fn bytes_from_raw<'a>(ptr: *const u8, len: usize, label: &str) -> VortexResult<&'a [u8]> {
    if len == 0 {
        return Ok(&[]);
    }
    vortex_ensure!(!ptr.is_null(), "{label} data pointer is null");
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

fn fixed_bytes_from_raw<const N: usize>(ptr: *const u8, label: &str) -> VortexResult<[u8; N]> {
    vortex_ensure!(!ptr.is_null(), "{label} data pointer is null");
    let mut bytes = [0u8; N];
    bytes.copy_from_slice(unsafe { slice::from_raw_parts(ptr, N) });
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use std::sync::Arc;

    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::dtype::DType;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::dtype::StructFields;
    use vortex::dtype::half::f16;
    use vortex::scalar::DecimalValue;
    use vortex::scalar::Scalar;

    use crate::array::*;
    use crate::dtype::vx_dtype;
    use crate::dtype::vx_dtype_free;
    use crate::dtype::vx_dtype_new_bool;
    use crate::dtype::vx_dtype_new_primitive;
    use crate::ptype::vx_ptype;
    use crate::scalar::*;
    use crate::session::vx_session;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;
    use crate::string::vx_view;
    use crate::tests::assert_error;
    use crate::tests::assert_no_error;

    fn assert_scalar(ptr: *mut vx_scalar, expected: Scalar) {
        assert!(!ptr.is_null());
        assert_eq!(vx_scalar::as_ref(ptr), &expected);
        unsafe { vx_scalar_free(ptr) };
    }

    #[test]
    fn test_primitive_scalar_constructors() {
        unsafe {
            assert_scalar(
                vx_scalar_new_bool(true, true),
                Scalar::bool(true, Nullability::Nullable),
            );
            assert_scalar(
                vx_scalar_new_u8(42, false),
                Scalar::primitive(42u8, Nullability::NonNullable),
            );
            assert_scalar(
                vx_scalar_new_u16(42, true),
                Scalar::primitive(42u16, Nullability::Nullable),
            );
            assert_scalar(
                vx_scalar_new_u32(42, false),
                Scalar::primitive(42u32, Nullability::NonNullable),
            );
            assert_scalar(
                vx_scalar_new_u64(42, true),
                Scalar::primitive(42u64, Nullability::Nullable),
            );
            assert_scalar(
                vx_scalar_new_i8(-42, false),
                Scalar::primitive(-42i8, Nullability::NonNullable),
            );
            assert_scalar(
                vx_scalar_new_i16(-42, true),
                Scalar::primitive(-42i16, Nullability::Nullable),
            );
            assert_scalar(
                vx_scalar_new_i32(-42, true),
                Scalar::primitive(-42i32, Nullability::Nullable),
            );
            assert_scalar(
                vx_scalar_new_i64(-42, false),
                Scalar::primitive(-42i64, Nullability::NonNullable),
            );

            let f16_value = f16::from_f32(1.5);
            assert_scalar(
                vx_scalar_new_f16_bits(f16_value.to_bits(), false),
                Scalar::primitive(f16_value, Nullability::NonNullable),
            );
            assert_scalar(
                vx_scalar_new_f32(1.5, true),
                Scalar::primitive(1.5f32, Nullability::Nullable),
            );
            assert_scalar(
                vx_scalar_new_f64(1.5, false),
                Scalar::primitive(1.5f64, Nullability::NonNullable),
            );
        }
    }

    #[test]
    fn test_utf8_binary_and_null_scalar_constructors() {
        unsafe {
            let mut error = ptr::null_mut();
            let value = "literal";
            assert_scalar(
                vx_scalar_new_utf8(vx_view::from_str(value), false, &raw mut error),
                Scalar::utf8(value, Nullability::NonNullable),
            );
            assert_no_error(error);

            let invalid_utf8 = [0xffu8];
            let scalar =
                vx_scalar_new_utf8(vx_view::from_bytes(&invalid_utf8), false, &raw mut error);
            assert!(scalar.is_null());
            assert_error(error);

            let bytes = b"\xde\xad\xbe\xef";
            assert_scalar(
                vx_scalar_new_binary(bytes.as_ptr(), bytes.len(), true, &raw mut error),
                Scalar::binary(bytes.to_vec(), Nullability::Nullable),
            );
            assert_no_error(error);

            let dtype = vx_dtype_new_primitive(vx_ptype::PTYPE_I32, false);
            let null_scalar = vx_scalar_new_null(dtype, &raw mut error);
            vx_dtype_free(dtype);
            assert_no_error(error);
            assert!(vx_scalar_is_null(null_scalar));
            let scalar_dtype = vx_scalar_dtype(null_scalar);
            assert_eq!(
                vx_dtype::as_ref(scalar_dtype),
                &DType::Primitive(PType::I32, Nullability::Nullable)
            );
            vx_dtype_free(scalar_dtype);
            vx_scalar_free(null_scalar);
        }
    }

    #[test]
    fn test_scalar_clone() {
        unsafe {
            let scalar = vx_scalar_new_u8(7, false);
            let cloned = vx_scalar_clone(scalar);
            assert_eq!(vx_scalar::as_ref(cloned), vx_scalar::as_ref(scalar));
            vx_scalar_free(cloned);
            vx_scalar_free(scalar);
        }
    }

    #[test]
    fn test_decimal_scalar_constructors() {
        unsafe {
            let mut error = ptr::null_mut();
            assert_scalar(
                vx_scalar_new_decimal_i16(999, 3, 0, false, &raw mut error),
                Scalar::decimal(
                    DecimalValue::I16(999),
                    DecimalDType::new(3, 0),
                    Nullability::NonNullable,
                ),
            );
            assert_no_error(error);

            assert_scalar(
                vx_scalar_new_decimal_i32(999, 3, 0, true, &raw mut error),
                Scalar::decimal(
                    DecimalValue::I32(999),
                    DecimalDType::new(3, 0),
                    Nullability::Nullable,
                ),
            );
            assert_no_error(error);

            assert_scalar(
                vx_scalar_new_decimal_i64(999, 3, 0, false, &raw mut error),
                Scalar::decimal(
                    DecimalValue::I64(999),
                    DecimalDType::new(3, 0),
                    Nullability::NonNullable,
                ),
            );
            assert_no_error(error);

            let scalar = vx_scalar_new_decimal_i8(100, 2, 0, false, &raw mut error);
            assert!(scalar.is_null());
            assert_error(error);

            let i128_value = 12345i128;
            assert_scalar(
                vx_scalar_new_decimal_i128_le(
                    i128_value.to_le_bytes().as_ptr(),
                    10,
                    2,
                    true,
                    &raw mut error,
                ),
                Scalar::decimal(
                    DecimalValue::I128(i128_value),
                    DecimalDType::new(10, 2),
                    Nullability::Nullable,
                ),
            );
            assert_no_error(error);

            let i256_value = i256::from_i128(12345);
            assert_scalar(
                vx_scalar_new_decimal_i256_le(
                    i256_value.to_le_bytes().as_ptr(),
                    10,
                    2,
                    false,
                    &raw mut error,
                ),
                Scalar::decimal(
                    DecimalValue::I256(i256_value),
                    DecimalDType::new(10, 2),
                    Nullability::NonNullable,
                ),
            );
            assert_no_error(error);
        }
    }

    #[test]
    fn test_nested_scalar_constructors() {
        unsafe {
            let mut error = ptr::null_mut();

            let element_dtype = vx_dtype_new_primitive(vx_ptype::PTYPE_I32, false);
            let child0 = vx_scalar_new_i32(1, false);
            let child1 = vx_scalar_new_i32(2, false);
            let children = [child0.cast_const(), child1.cast_const()];

            assert_scalar(
                vx_scalar_new_list(
                    element_dtype,
                    children.as_ptr(),
                    children.len(),
                    true,
                    &raw mut error,
                ),
                Scalar::list(
                    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                    vec![
                        Scalar::primitive(1i32, Nullability::NonNullable),
                        Scalar::primitive(2i32, Nullability::NonNullable),
                    ],
                    Nullability::Nullable,
                ),
            );
            assert_no_error(error);

            let len = u32::try_from(children.len()).unwrap();
            assert_scalar(
                vx_scalar_new_fixed_size_list(
                    element_dtype,
                    children.as_ptr(),
                    len,
                    false,
                    &raw mut error,
                ),
                Scalar::fixed_size_list(
                    Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                    vec![
                        Scalar::primitive(1i32, Nullability::NonNullable),
                        Scalar::primitive(2i32, Nullability::NonNullable),
                    ],
                    Nullability::NonNullable,
                ),
            );
            assert_no_error(error);

            let wrong_child = vx_scalar_new_bool(true, false);
            let wrong_children = [wrong_child.cast_const()];
            let wrong = vx_scalar_new_list(
                element_dtype,
                wrong_children.as_ptr(),
                wrong_children.len(),
                false,
                &raw mut error,
            );
            assert!(wrong.is_null());
            assert_error(error);

            let struct_dtype = vx_dtype::new(DType::Struct(
                StructFields::new(
                    ["flag", "value"].into(),
                    vec![
                        DType::Bool(Nullability::NonNullable),
                        DType::Primitive(PType::I32, Nullability::NonNullable),
                    ],
                ),
                Nullability::NonNullable,
            ));
            let flag = vx_scalar_new_bool(true, false);
            let value = vx_scalar_new_i32(10, false);
            let fields = [flag.cast_const(), value.cast_const()];
            assert_scalar(
                vx_scalar_new_struct(struct_dtype, fields.as_ptr(), fields.len(), &raw mut error),
                Scalar::struct_(
                    DType::Struct(
                        StructFields::new(
                            ["flag", "value"].into(),
                            vec![
                                DType::Bool(Nullability::NonNullable),
                                DType::Primitive(PType::I32, Nullability::NonNullable),
                            ],
                        ),
                        Nullability::NonNullable,
                    ),
                    vec![
                        Scalar::bool(true, Nullability::NonNullable),
                        Scalar::primitive(10i32, Nullability::NonNullable),
                    ],
                ),
            );
            assert_no_error(error);

            let missing_field = vx_scalar_new_struct(
                struct_dtype,
                fields.as_ptr(),
                fields.len() - 1,
                &raw mut error,
            );
            assert!(missing_field.is_null());
            assert_error(error);

            vx_dtype_free(element_dtype);
            vx_dtype_free(struct_dtype);
            vx_scalar_free(child0);
            vx_scalar_free(child1);
            vx_scalar_free(wrong_child);
            vx_scalar_free(flag);
            vx_scalar_free(value);
        }
    }

    #[test]
    fn test_nested_null_inputs() {
        unsafe {
            let mut error = ptr::null_mut();
            let dtype = vx_dtype_new_bool(false);
            assert!(vx_scalar_new_list(dtype, ptr::null(), 1, false, &raw mut error).is_null());
            assert_error(error);

            let empty = vx_scalar_new_list(dtype, ptr::null(), 0, false, &raw mut error);
            assert_no_error(error);
            assert!(!empty.is_null());
            vx_scalar_free(empty);
            vx_dtype_free(dtype);
        }
    }

    #[test]
    // TODO(joe): enable once this is fixed https://github.com/Amanieu/parking_lot/issues/477
    #[cfg_attr(miri, ignore)]
    fn test_array_scalar_getters() {
        unsafe fn get_i32(session: *const vx_session, array: *const vx_array, index: usize) -> i32 {
            let mut error = ptr::null_mut();
            let scalar = unsafe { vx_array_get_scalar(session, array, index, &raw mut error) };
            assert_no_error(error);
            let value = unsafe { vx_scalar_get_i32(scalar) };
            unsafe { vx_scalar_free(scalar.cast_mut()) };
            value
        }

        unsafe fn get_f64(session: *const vx_session, array: *const vx_array, index: usize) -> f64 {
            let mut error = ptr::null_mut();
            let scalar = unsafe { vx_array_get_scalar(session, array, index, &raw mut error) };
            assert_no_error(error);
            let value = unsafe { vx_scalar_get_f64(scalar) };
            unsafe { vx_scalar_free(scalar.cast_mut()) };
            value
        }

        unsafe {
            let session = vx_session_new();

            let i32_array =
                PrimitiveArray::new(buffer![i32::MAX, i32::MIN, 0], Validity::NonNullable)
                    .into_array();
            let ffi_i32 = vx_array::new(i32_array);
            assert!(vx_array_is_primitive(ffi_i32, vx_ptype::PTYPE_I32));
            assert_eq!(get_i32(session, ffi_i32, 0), i32::MAX);
            assert_eq!(get_i32(session, ffi_i32, 1), i32::MIN);
            assert_eq!(get_i32(session, ffi_i32, 2), 0);
            vx_array_free(ffi_i32);

            let f64_array = PrimitiveArray::new(
                buffer![f64::NEG_INFINITY, 0.0f64, f64::NAN],
                Validity::NonNullable,
            )
            .into_array();
            let ffi_f64 = vx_array::new(f64_array);
            assert_eq!(get_f64(session, ffi_f64, 0), f64::NEG_INFINITY);
            assert_eq!(get_f64(session, ffi_f64, 1), 0.0);
            assert!(get_f64(session, ffi_f64, 2).is_nan());
            vx_array_free(ffi_f64);

            vx_session_free(session);
        }
    }

    #[test]
    fn test_scalar_primitive_getters() {
        unsafe {
            let s = vx_scalar_new_i32(-42, false);
            assert_eq!(vx_scalar_get_i32(s), -42);
            vx_scalar_free(s);

            let s = vx_scalar_new_u64(u64::MAX, true);
            assert_eq!(vx_scalar_get_u64(s), u64::MAX);
            vx_scalar_free(s);

            let s = vx_scalar_new_f64(1.5, false);
            assert_eq!(vx_scalar_get_f64(s), 1.5);
            vx_scalar_free(s);

            let s = vx_scalar_new_bool(true, false);
            assert!(vx_scalar_get_bool(s));
            vx_scalar_free(s);
        }
    }

    #[test]
    fn test_scalar_string_getters() {
        unsafe {
            let mut error = ptr::null_mut();

            let value = "hello";
            let s = vx_scalar_new_utf8(vx_view::from_str(value), false, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_scalar_get_utf8(s).as_str().unwrap(), value);
            vx_scalar_free(s);

            let bytes = b"\xde\xad\xbe\xef";
            let s = vx_scalar_new_binary(bytes.as_ptr(), bytes.len(), false, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_scalar_get_binary(s).as_bytes().unwrap(), bytes);
            vx_scalar_free(s);
        }
    }

    #[test]
    fn test_scalar_decimal_getters() {
        unsafe {
            let mut error = ptr::null_mut();

            let s = vx_scalar_new_decimal_i32(1234, 5, 2, false, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_scalar_get_decimal_i32(s), 1234);
            vx_scalar_free(s);

            let s = vx_scalar_new_decimal_i64(99999, 12, 3, false, &raw mut error);
            assert_no_error(error);
            assert_eq!(vx_scalar_get_decimal_i64(s), 99999);
            vx_scalar_free(s);
        }
    }
}
