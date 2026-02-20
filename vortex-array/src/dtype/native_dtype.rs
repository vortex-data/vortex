// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::ByteBuffer;

use crate::dtype::DType;
use crate::dtype::NativeDType;
use crate::dtype::Nullability;

/// It is common to represent a nullable type `T` as an `Option<T>`, so we implement a blanket
/// implementation for all `Option<T>` to simply be a nullable `T`.
impl<T> NativeDType for Option<T>
where
    T: NativeDType,
{
    fn dtype() -> DType {
        T::dtype().as_nullable()
    }
}

impl<T> NativeDType for Vec<T>
where
    T: NativeDType,
{
    fn dtype() -> DType {
        DType::List(Arc::new(T::dtype()), Nullability::NonNullable)
    }
}

impl NativeDType for bool {
    fn dtype() -> DType {
        DType::Bool(Nullability::NonNullable)
    }
}

impl NativeDType for String {
    fn dtype() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }
}

impl NativeDType for &str {
    fn dtype() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }
}

impl NativeDType for &[u8] {
    fn dtype() -> DType {
        DType::Binary(Nullability::NonNullable)
    }
}

impl NativeDType for ByteBuffer {
    fn dtype() -> DType {
        DType::Binary(Nullability::NonNullable)
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::*;
    use crate::dtype::PType;

    #[test]
    fn test_bool_native_dtype() {
        let dtype = bool::dtype();
        assert_eq!(dtype, DType::Bool(Nullability::NonNullable));
    }

    #[test]
    fn test_primitive_native_dtypes() {
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
    fn test_string_native_dtypes() {
        assert_eq!(String::dtype(), DType::Utf8(Nullability::NonNullable));
    }

    #[test]
    fn test_vec_option_native_dtype() {
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
    fn test_vec_native_dtypes() {
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
    fn test_vec_string_native_dtype() {
        assert_eq!(
            Vec::<String>::dtype(),
            DType::List(
                Arc::new(DType::Utf8(Nullability::NonNullable)),
                Nullability::NonNullable
            )
        );
    }

    #[test]
    fn test_str_native_dtype() {
        assert_eq!(
            <&str as NativeDType>::dtype(),
            DType::Utf8(Nullability::NonNullable)
        );
    }
}
