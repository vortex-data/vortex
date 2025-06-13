use std::fmt::{Debug, Formatter};

use crate::{cpp, wrapper};

wrapper!(
    LogicalType,
    cpp::duckdb_logical_type,
    cpp::duckdb_destroy_logical_type
);

/// `LogicalType` is Send, as the wrapped pointer and bool are Send.
unsafe impl Send for LogicalType {}

impl LogicalType {
    pub fn new(dtype: cpp::DUCKDB_TYPE) -> Self {
        unsafe { Self::own(cpp::duckdb_create_logical_type(dtype)) }
    }

    pub fn as_type_id(&self) -> cpp::DUCKDB_TYPE {
        unsafe { cpp::duckdb_get_type_id(self.as_ptr()) }
    }

    pub fn varchar() -> Self {
        Self::new(cpp::DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR)
    }
}

impl Debug for LogicalType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let debug = unsafe { cpp::duckdb_vx_logical_type_stringify(self.as_ptr()) };
        write!(f, "{}", unsafe {
            std::ffi::CStr::from_ptr(debug).to_string_lossy()
        })?;
        unsafe { cpp::duckdb_free(debug.cast()) };
        Ok(())
    }
}

/// A trait representing the DuckDB logical types.
pub trait DuckDBType {}

macro_rules! duckdb_type {
    ($name:ident) => {
        pub struct $name;
        impl DuckDBType for $name {}
    };
}

/// Fixed-width primitive types in DuckDB.
pub trait PrimitiveType: DuckDBType {
    type NATIVE;
}

macro_rules! primitive_type {
    ($name:ident, $native:ty) => {
        duckdb_type!($name);
        impl PrimitiveType for $name {
            type NATIVE = $native;
        }
    };
}

/// Integer types in DuckDB.
pub trait IntegerType: PrimitiveType {}

macro_rules! integer_type {
    ($name:ident, $native:ty) => {
        primitive_type!($name, $native);
        impl IntegerType for $name {}
    };
}

integer_type!(TinyInt, i8);
integer_type!(SmallInt, i16);
integer_type!(Integer, i32);
integer_type!(BigInt, i64);
integer_type!(HugeInt, i128);
integer_type!(UTinyInt, u8);
integer_type!(USmallInt, u16);
integer_type!(UInteger, u32);
integer_type!(UBigInt, u64);
integer_type!(UHugeInt, u128);

#[macro_export]
macro_rules! match_each_primitive_type {
    ($self:expr, | $type:ident | $body:block) => {{
        use $crate::duckdb::LogicalType;
        match $self.as_type_id() {
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => {
                let $type = <$crate::duckdb::TinyInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => {
                let $type = <$crate::duckdb::SmallInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER => {
                let $type = <$crate::duckdb::Integer as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_BIGINT => {
                let $type = <$crate::duckdb::BigInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_HUGEINT => {
                let $type = <$crate::duckdb::HugeInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT => {
                let $type = <$crate::duckdb::UTinyInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT => {
                let $type = <$crate::duckdb::USmallInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER => {
                let $type = <$crate::duckdb::UInteger as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT => {
                let $type = <$crate::duckdb::UBigInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_UHUGEINT => {
                let $type = <$crate::duckdb::UHugeInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => {
                let $type = <$crate::duckdb::Float as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => {
                let $type = <$crate::duckdb::Double as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            _ => vortex_panic!(
                "Unexpected type for match_each_primitive_type: {:?}",
                $self.as_type_id()
            ),
        }
    }};
}

/// Floating point types in DuckDB.
pub trait FloatingType: PrimitiveType {}

macro_rules! floating_type {
    ($name:ident, $native:ty) => {
        primitive_type!($name, $native);
        impl FloatingType for $name {}
    };
}

floating_type!(Float, f32);
floating_type!(Double, f64);
