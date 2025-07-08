// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};

use crate::cpp::*;
use crate::wrapper;

wrapper!(
    LogicalType,
    duckdb_logical_type,
    duckdb_destroy_logical_type
);

/// `LogicalType` is Send, as the wrapped pointer and bool are Send.
unsafe impl Send for LogicalType {}

impl Clone for LogicalType {
    fn clone(&self) -> Self {
        unsafe { Self::own(duckdb_vx_logical_type_copy(self.as_ptr())) }
    }
}

impl LogicalType {
    pub fn new(dtype: DUCKDB_TYPE) -> Self {
        unsafe { Self::own(duckdb_create_logical_type(dtype)) }
    }

    pub fn as_type_id(&self) -> DUCKDB_TYPE {
        unsafe { duckdb_get_type_id(self.as_ptr()) }
    }

    pub fn varchar() -> Self {
        Self::new(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR)
    }

    pub fn as_decimal(&self) -> (u8, u8) {
        unsafe {
            (
                duckdb_decimal_width(self.as_ptr()),
                duckdb_decimal_scale(self.as_ptr()),
            )
        }
    }

    pub fn array_child_type(&self) -> Self {
        unsafe { LogicalType::own(duckdb_array_type_child_type(self.as_ptr())) }
    }

    pub fn list_child_type(&self) -> Self {
        unsafe { LogicalType::own(duckdb_list_type_child_type(self.as_ptr())) }
    }

    pub fn map_key_type(&self) -> Self {
        unsafe { LogicalType::own(duckdb_map_type_key_type(self.as_ptr())) }
    }

    pub fn map_value_type(&self) -> Self {
        unsafe { LogicalType::own(duckdb_map_type_value_type(self.as_ptr())) }
    }

    pub fn struct_child_type(&self, idx: idx_t) -> Self {
        unsafe { LogicalType::own(duckdb_struct_type_child_type(self.as_ptr(), idx)) }
    }

    pub fn union_member_type(&self, idx: idx_t) -> Self {
        unsafe { LogicalType::own(duckdb_union_type_member_type(self.as_ptr(), idx)) }
    }
}

impl Debug for LogicalType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let debug = unsafe { duckdb_vx_logical_type_stringify(self.as_ptr()) };
        write!(f, "{}", unsafe {
            std::ffi::CStr::from_ptr(debug).to_string_lossy()
        })?;
        unsafe { duckdb_free(debug.cast()) };
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
            DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => {
                let $type = <$crate::duckdb::TinyInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => {
                let $type = <$crate::duckdb::SmallInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_INTEGER => {
                let $type = <$crate::duckdb::Integer as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_BIGINT => {
                let $type = <$crate::duckdb::BigInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_HUGEINT => {
                let $type = <$crate::duckdb::HugeInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT => {
                let $type = <$crate::duckdb::UTinyInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT => {
                let $type = <$crate::duckdb::USmallInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER => {
                let $type = <$crate::duckdb::UInteger as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT => {
                let $type = <$crate::duckdb::UBigInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UHUGEINT => {
                let $type = <$crate::duckdb::UHugeInt as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => {
                let $type = <$crate::duckdb::Float as $crate::duckdb::PrimitiveType>::NATIVE;
                $body
            }
            DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => {
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

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::duckdb::LogicalType;

    #[test]
    fn test_clone_logical_type() {
        for variant in (DUCKDB_TYPE::DUCKDB_TYPE_INVALID as u32
            ..=DUCKDB_TYPE::DUCKDB_TYPE_INTEGER_LITERAL as u32)
            .map(|variant| unsafe { std::mem::transmute::<u32, DUCKDB_TYPE>(variant) })
            .filter(|&variant| {
                // `LogicalType::new` calls the DuckDB C API function
                // `duckdb_create_logical_type` with just the type enum.
                //
                // Though complex types require additional parameters:
                let excluded_types = [
                    // Needs width and scale parameters
                    DUCKDB_TYPE::DUCKDB_TYPE_DECIMAL,
                    // Needs the enum dictionary/values
                    DUCKDB_TYPE::DUCKDB_TYPE_ENUM,
                    // Needs the child element type
                    DUCKDB_TYPE::DUCKDB_TYPE_LIST,
                    // Needs field names and their type
                    DUCKDB_TYPE::DUCKDB_TYPE_STRUCT,
                    //  Needs key and value types
                    DUCKDB_TYPE::DUCKDB_TYPE_MAP,
                    // Needs member types
                    DUCKDB_TYPE::DUCKDB_TYPE_UNION,
                    //  Needs element type and array size
                    DUCKDB_TYPE::DUCKDB_TYPE_ARRAY,
                ];
                !excluded_types.contains(&variant)
            })
        {
            assert_eq!(LogicalType::new(variant).clone().as_type_id(), variant);
        }
    }

    #[test]
    fn test_clone_decimal_logical_type() {
        let decimal_type = unsafe { LogicalType::own(duckdb_create_decimal_type(10, 2)) };
        #[allow(clippy::redundant_clone)]
        let cloned = decimal_type.clone();

        assert_eq!(decimal_type.as_type_id(), cloned.as_type_id());

        // Further verify the parameters are preserved.
        let original_width = unsafe { duckdb_decimal_width(decimal_type.as_ptr()) };
        let original_scale = unsafe { duckdb_decimal_scale(decimal_type.as_ptr()) };

        let cloned_width = unsafe { duckdb_decimal_width(cloned.as_ptr()) };
        let cloned_scale = unsafe { duckdb_decimal_scale(cloned.as_ptr()) };

        assert_eq!(original_width, cloned_width);
        assert_eq!(original_scale, cloned_scale);
    }

    #[test]
    fn test_clone_list_logical_type() {
        // Create a list of integers
        let int_type = unsafe {
            LogicalType::own(duckdb_create_logical_type(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER))
        };
        let list_type = unsafe { LogicalType::own(duckdb_create_list_type(int_type.as_ptr())) };

        #[allow(clippy::redundant_clone)]
        let cloned = list_type.clone();

        assert_eq!(list_type.as_type_id(), cloned.as_type_id());
        assert_eq!(list_type.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_LIST);

        // Verify the child type is preserved
        let original_child = list_type.list_child_type();
        let cloned_child = cloned.list_child_type();

        let original_child_type_id = unsafe { duckdb_get_type_id(original_child.as_ptr()) };
        let cloned_child_type_id = unsafe { duckdb_get_type_id(cloned_child.as_ptr()) };

        assert_eq!(original_child_type_id, cloned_child_type_id);
        assert_eq!(original_child_type_id, DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
    }

    #[test]
    fn test_clone_array_logical_type() {
        // Create an array of strings with size 5
        let varchar_type = unsafe {
            LogicalType::own(duckdb_create_logical_type(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR))
        };
        let array_type =
            unsafe { LogicalType::own(duckdb_create_array_type(varchar_type.as_ptr(), 5)) };
        #[allow(clippy::redundant_clone)]
        let cloned = array_type.clone();

        assert_eq!(array_type.as_type_id(), cloned.as_type_id());
        assert_eq!(array_type.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_ARRAY);

        // Verify the child type is preserved
        let original_child = array_type.array_child_type();
        let cloned_child = cloned.array_child_type();

        let original_child_type_id = unsafe { duckdb_get_type_id(original_child.as_ptr()) };
        let cloned_child_type_id = unsafe { duckdb_get_type_id(cloned_child.as_ptr()) };

        assert_eq!(original_child_type_id, cloned_child_type_id);
        assert_eq!(original_child_type_id, DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);

        // Verify the array size is preserved
        let original_size = unsafe { duckdb_array_type_array_size(array_type.as_ptr()) };
        let cloned_size = unsafe { duckdb_array_type_array_size(cloned.as_ptr()) };

        assert_eq!(original_size, cloned_size);
        assert_eq!(original_size, 5);
    }

    #[test]
    fn test_clone_map_logical_type() {
        // Create a map of string -> integer
        let key_type = unsafe {
            LogicalType::own(duckdb_create_logical_type(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR))
        };
        let value_type = unsafe {
            LogicalType::own(duckdb_create_logical_type(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER))
        };
        let map_type = unsafe {
            LogicalType::own(duckdb_create_map_type(
                key_type.as_ptr(),
                value_type.as_ptr(),
            ))
        };

        #[allow(clippy::redundant_clone)]
        let cloned = map_type.clone();

        assert_eq!(map_type.as_type_id(), cloned.as_type_id());
        assert_eq!(map_type.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_MAP);

        // Verify the key and value types are preserved
        let original_key = map_type.map_key_type();
        let original_value = map_type.map_value_type();
        let cloned_key = cloned.map_key_type();
        let cloned_value = cloned.map_value_type();

        assert_eq!(original_key.as_type_id(), cloned_key.as_type_id());
        assert_eq!(original_value.as_type_id(), cloned_value.as_type_id());
        assert_eq!(original_key.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);
        assert_eq!(
            original_value.as_type_id(),
            DUCKDB_TYPE::DUCKDB_TYPE_INTEGER
        );
    }

    #[test]
    fn test_clone_struct_logical_type() {
        // Create a struct with two fields: {name: VARCHAR, age: INTEGER}
        let name_type = unsafe {
            LogicalType::own(duckdb_create_logical_type(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR))
        };
        let age_type = unsafe {
            LogicalType::own(duckdb_create_logical_type(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER))
        };

        let mut member_types = vec![name_type.as_ptr(), age_type.as_ptr()];
        let name_cstr = std::ffi::CString::new("name").unwrap();
        let age_cstr = std::ffi::CString::new("age").unwrap();
        let mut member_names = vec![name_cstr.as_ptr(), age_cstr.as_ptr()];

        let struct_type = unsafe {
            LogicalType::own(duckdb_create_struct_type(
                member_types.as_mut_ptr(),
                member_names.as_mut_ptr(),
                2,
            ))
        };

        #[allow(clippy::redundant_clone)]
        let cloned = struct_type.clone();

        assert_eq!(struct_type.as_type_id(), cloned.as_type_id());
        assert_eq!(struct_type.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_STRUCT);

        // Verify the child count is preserved
        let original_count = unsafe { duckdb_struct_type_child_count(struct_type.as_ptr()) };
        let cloned_count = unsafe { duckdb_struct_type_child_count(cloned.as_ptr()) };
        assert_eq!(original_count, cloned_count);
        assert_eq!(original_count, 2);

        // Verify each field
        for idx in 0..original_count {
            let original_child_type = struct_type.struct_child_type(idx);
            let cloned_child_type = cloned.struct_child_type(idx);
            let original_child_name =
                unsafe { duckdb_struct_type_child_name(struct_type.as_ptr(), idx) };
            let cloned_child_name = unsafe { duckdb_struct_type_child_name(cloned.as_ptr(), idx) };

            assert_eq!(
                original_child_type.as_type_id(),
                cloned_child_type.as_type_id()
            );

            let original_name = unsafe { std::ffi::CStr::from_ptr(original_child_name) };
            let cloned_name = unsafe { std::ffi::CStr::from_ptr(cloned_child_name) };
            assert_eq!(original_name, cloned_name);

            // Free strings
            unsafe {
                duckdb_free(original_child_name.cast());
                duckdb_free(cloned_child_name.cast());
            }
        }
    }

    #[test]
    fn test_clone_union_logical_type() {
        // Create a union with two members: {str: VARCHAR, num: INTEGER}
        let str_type = unsafe {
            LogicalType::own(duckdb_create_logical_type(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR))
        };
        let num_type = unsafe {
            LogicalType::own(duckdb_create_logical_type(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER))
        };

        let mut member_types = vec![str_type.as_ptr(), num_type.as_ptr()];
        let str_cstr = std::ffi::CString::new("str").unwrap();
        let num_cstr = std::ffi::CString::new("num").unwrap();
        let mut member_names = vec![str_cstr.as_ptr(), num_cstr.as_ptr()];

        let union_type = unsafe {
            LogicalType::own(duckdb_create_union_type(
                member_types.as_mut_ptr(),
                member_names.as_mut_ptr(),
                2,
            ))
        };

        #[allow(clippy::redundant_clone)]
        let cloned = union_type.clone();

        assert_eq!(union_type.as_type_id(), cloned.as_type_id());
        assert_eq!(union_type.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_UNION);

        // Verify the member count is preserved
        let original_count = unsafe { duckdb_union_type_member_count(union_type.as_ptr()) };
        let cloned_count = unsafe { duckdb_union_type_member_count(cloned.as_ptr()) };
        assert_eq!(original_count, cloned_count);
        assert_eq!(original_count, 2);

        // Verify each member
        for idx in 0..original_count {
            let original_member_type = union_type.union_member_type(idx);
            let cloned_member_type = cloned.union_member_type(idx);
            let original_member_name =
                unsafe { duckdb_union_type_member_name(union_type.as_ptr(), idx) };
            let cloned_member_name = unsafe { duckdb_union_type_member_name(cloned.as_ptr(), idx) };

            assert_eq!(
                original_member_type.as_type_id(),
                cloned_member_type.as_type_id(),
            );

            assert_eq!(
                unsafe { std::ffi::CStr::from_ptr(original_member_name) },
                unsafe { std::ffi::CStr::from_ptr(cloned_member_name) }
            );

            // Free strings
            unsafe {
                duckdb_free(original_member_name.cast());
                duckdb_free(cloned_member_name.cast());
            }
        }
    }
}
