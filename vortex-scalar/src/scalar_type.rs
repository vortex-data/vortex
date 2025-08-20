// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::{BufferString, ByteBuffer};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, NativePType, Nullability, PType};

/// A trait for types that can be used as scalar values.
///
/// This trait is implemented by native Rust types that can be converted
/// to and from Vortex scalar values.
pub trait ScalarType {
    /// Returns the Vortex data type for this scalar type.
    fn dtype() -> DType;
}

impl ScalarType for bool {
    fn dtype() -> DType {
        DType::Bool(Nullability::NonNullable)
    }
}

macro_rules! scalar_type_for_vec {
    ($T:ty) => {
        impl ScalarType for Vec<$T> {
            fn dtype() -> DType {
                DType::List(Arc::new(<$T>::dtype()), Nullability::NonNullable)
            }
        }
    };
}

macro_rules! scalar_type_for_native_ptype {
    ($T:ty,without_vec) => {
        impl ScalarType for $T {
            fn dtype() -> DType {
                DType::Primitive(<$T>::PTYPE, Nullability::NonNullable)
            }
        }
    };
    ($T:ty,with_vec) => {
        scalar_type_for_native_ptype!($T, without_vec);
        scalar_type_for_vec!($T);
    };
}

scalar_type_for_native_ptype!(u8, without_vec); // Vec<u8> could be either Binary or List(U8)
scalar_type_for_native_ptype!(u16, with_vec);
scalar_type_for_native_ptype!(u32, with_vec);
scalar_type_for_native_ptype!(u64, with_vec);
scalar_type_for_native_ptype!(i8, with_vec);
scalar_type_for_native_ptype!(i16, with_vec);
scalar_type_for_native_ptype!(i32, with_vec);
scalar_type_for_native_ptype!(i64, with_vec);
scalar_type_for_native_ptype!(f32, with_vec);
scalar_type_for_native_ptype!(f64, with_vec);

impl ScalarType for f16 {
    fn dtype() -> DType {
        DType::Primitive(PType::F16, Nullability::NonNullable)
    }
}

scalar_type_for_vec!(f16);

impl ScalarType for String {
    fn dtype() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }
}

scalar_type_for_vec!(String);

impl ScalarType for BufferString {
    fn dtype() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }
}

scalar_type_for_vec!(BufferString);

impl ScalarType for ByteBuffer {
    fn dtype() -> DType {
        DType::Binary(Nullability::NonNullable)
    }
}

scalar_type_for_vec!(ByteBuffer);

#[cfg(test)]
mod tests {
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
