use std::sync::Arc;

use vortex_buffer::{Buffer, BufferString};
use vortex_dtype::half::f16;
use vortex_dtype::{DType, NativePType, Nullability, PType};

pub trait ScalarType {
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

impl ScalarType for usize {
    fn dtype() -> DType {
        DType::Primitive(PType::U64, Nullability::NonNullable)
    }
}

scalar_type_for_vec!(usize);

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

impl ScalarType for bytes::Bytes {
    fn dtype() -> DType {
        DType::Binary(Nullability::NonNullable)
    }
}

scalar_type_for_vec!(bytes::Bytes);

impl ScalarType for Buffer {
    fn dtype() -> DType {
        DType::Binary(Nullability::NonNullable)
    }
}

scalar_type_for_vec!(Buffer);
