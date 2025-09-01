// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::{BufferString, ByteBuffer};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, NativePType, Nullability};

/// A trait for types that can be used as scalar values.
///
/// This trait is implemented by native Rust types that can be converted
/// to and from Vortex scalar values.
///
/// Unless the type is an `Option<T>`, the `DType` that is returned by `dtype()` should **ALWAYS**
/// be [`NonNullable`](Nullability::NonNullable).
pub trait ScalarType {
    /// Returns the Vortex data type for this scalar type.
    fn dtype() -> DType;
}

/// It is common to represent a nullable type `T` as an `Option<T>`, so we implement a blanket
/// implementation for all `Option<T>` to simply be a nullable `T`.
impl<T> ScalarType for Option<T>
where
    T: ScalarType,
{
    fn dtype() -> DType {
        T::dtype().as_nullable()
    }
}

impl<T> ScalarType for Vec<T>
where
    T: ScalarType,
{
    fn dtype() -> DType {
        DType::List(Arc::new(T::dtype()), Nullability::NonNullable)
    }
}

impl ScalarType for bool {
    fn dtype() -> DType {
        DType::Bool(Nullability::NonNullable)
    }
}

/// We manually implement `ScalarType` for the primitive types because doing a blanket
/// implementation would cause a conflict.
///
/// If you don't believe this, see for yourself! Try uncommenting this:
///
/// ```ignore
/// impl<T> ScalarType for T
/// where
///     T: NativePType,
/// {
///     fn dtype() -> DType {
///         DType::Primitive(T::PTYPE, Nullability::NonNullable)
///     }
/// }
/// ```
macro_rules! scalar_type_for_native_ptype {
    ($T:ty) => {
        impl ScalarType for $T {
            fn dtype() -> DType {
                DType::Primitive(<$T>::PTYPE, Nullability::NonNullable)
            }
        }
    };
}

scalar_type_for_native_ptype!(u8); // Vec<u8> could be either Binary or List(U8)
scalar_type_for_native_ptype!(u16);
scalar_type_for_native_ptype!(u32);
scalar_type_for_native_ptype!(u64);
scalar_type_for_native_ptype!(i8);
scalar_type_for_native_ptype!(i16);
scalar_type_for_native_ptype!(i32);
scalar_type_for_native_ptype!(i64);
scalar_type_for_native_ptype!(f16);
scalar_type_for_native_ptype!(f32);
scalar_type_for_native_ptype!(f64);

impl ScalarType for String {
    fn dtype() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }
}

impl ScalarType for BufferString {
    fn dtype() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }
}

impl ScalarType for ByteBuffer {
    fn dtype() -> DType {
        DType::Binary(Nullability::NonNullable)
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::PType;

    use super::*;

    #[test]
    fn test_bool_scalar_type() {
        let dtype = bool::dtype();
        assert_eq!(dtype, DType::Bool(Nullability::NonNullable));
    }

    #[test]
    fn test_primitive_scalar_types() {
        assert_eq!(
            u8::dtype(),
            DType::Primitive(PType::U8, Nullability::NonNullable)
        );
        assert_eq!(
            u16::dtype(),
            DType::Primitive(PType::U16, Nullability::NonNullable)
        );
        assert_eq!(
            u32::dtype(),
            DType::Primitive(PType::U32, Nullability::NonNullable)
        );
        assert_eq!(
            u64::dtype(),
            DType::Primitive(PType::U64, Nullability::NonNullable)
        );

        assert_eq!(
            i8::dtype(),
            DType::Primitive(PType::I8, Nullability::NonNullable)
        );
        assert_eq!(
            i16::dtype(),
            DType::Primitive(PType::I16, Nullability::NonNullable)
        );
        assert_eq!(
            i32::dtype(),
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(
            i64::dtype(),
            DType::Primitive(PType::I64, Nullability::NonNullable)
        );

        assert_eq!(
            f32::dtype(),
            DType::Primitive(PType::F32, Nullability::NonNullable)
        );
        assert_eq!(
            f64::dtype(),
            DType::Primitive(PType::F64, Nullability::NonNullable)
        );
        assert_eq!(
            f16::dtype(),
            DType::Primitive(PType::F16, Nullability::NonNullable)
        );
    }

    #[test]
    fn test_string_scalar_types() {
        assert_eq!(String::dtype(), DType::Utf8(Nullability::NonNullable));
        assert_eq!(BufferString::dtype(), DType::Utf8(Nullability::NonNullable));
    }

    #[test]
    fn test_byte_buffer_scalar_type() {
        assert_eq!(ByteBuffer::dtype(), DType::Binary(Nullability::NonNullable));
    }

    #[test]
    fn test_vec_option_scalar_type() {
        // Test that Vec<Option<T>> has non-nullable List dtype even though elements are nullable.
        assert_eq!(
            Vec::<Option<i32>>::dtype(),
            DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
                Nullability::NonNullable
            )
        );

        assert_eq!(
            Vec::<Option<String>>::dtype(),
            DType::List(
                Arc::new(DType::Utf8(Nullability::Nullable)),
                Nullability::NonNullable
            )
        );

        assert_eq!(
            Vec::<Option<bool>>::dtype(),
            DType::List(
                Arc::new(DType::Bool(Nullability::Nullable)),
                Nullability::NonNullable
            )
        );
    }

    #[test]
    fn test_vec_scalar_types() {
        // Test Vec<primitive> types
        assert_eq!(
            Vec::<u16>::dtype(),
            DType::List(
                Arc::new(DType::Primitive(PType::U16, Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );

        assert_eq!(
            Vec::<i32>::dtype(),
            DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );

        assert_eq!(
            Vec::<f64>::dtype(),
            DType::List(
                Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );

        assert_eq!(
            Vec::<f16>::dtype(),
            DType::List(
                Arc::new(DType::Primitive(PType::F16, Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );
    }

    #[test]
    fn test_vec_string_scalar_type() {
        assert_eq!(
            Vec::<String>::dtype(),
            DType::List(
                Arc::new(DType::Utf8(Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );

        assert_eq!(
            Vec::<BufferString>::dtype(),
            DType::List(
                Arc::new(DType::Utf8(Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );
    }

    #[test]
    fn test_vec_byte_buffer_scalar_type() {
        assert_eq!(
            Vec::<ByteBuffer>::dtype(),
            DType::List(
                Arc::new(DType::Binary(Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );
    }
}
