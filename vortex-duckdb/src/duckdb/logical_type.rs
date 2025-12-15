// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::CString;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex::dtype::ExtDType;
use vortex::dtype::FieldName;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::VortexUnwrap;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::cpp::*;
use crate::wrapper;

wrapper!(
    LogicalType,
    duckdb_logical_type,
    duckdb_destroy_logical_type
);

/// `LogicalType` is Send, as the wrapped pointer and bool are Send.
unsafe impl Send for LogicalType {}
unsafe impl Sync for LogicalType {}

impl Clone for LogicalType {
    fn clone(&self) -> Self {
        unsafe { Self::own(duckdb_vx_logical_type_copy(self.as_ptr())) }
    }
}

impl LogicalType {
    pub fn new(dtype: DUCKDB_TYPE) -> Self {
        unsafe { Self::own(duckdb_create_logical_type(dtype)) }
    }

    /// Creates a DuckDB struct logical type from child types and field names.
    pub fn struct_type<T, N>(child_types: T, child_names: N) -> VortexResult<LogicalType>
    where
        T: IntoIterator<Item = LogicalType>,
        N: IntoIterator<Item = CString>,
    {
        let child_types: Vec<LogicalType> = child_types.into_iter().collect();
        let child_names: Vec<CString> = child_names.into_iter().collect();

        let mut child_type_ptrs: Vec<duckdb_logical_type> =
            child_types.iter().map(|lt| lt.as_ptr()).collect();

        let mut child_name_ptrs: Vec<*const std::ffi::c_char> =
            child_names.iter().map(|name| name.as_ptr()).collect();

        let struct_type_ptr = unsafe {
            duckdb_create_struct_type(
                child_type_ptrs.as_mut_ptr(),
                child_name_ptrs.as_mut_ptr(),
                child_types.len() as _,
            )
        };

        if struct_type_ptr.is_null() {
            vortex_bail!("Failed to create struct logical type");
        }

        Ok(unsafe { Self::own(struct_type_ptr) })
    }

    /// Creates a DuckDB decimal logical type with the specified precision and scale.
    pub fn decimal_type(precision: u8, scale: u8) -> VortexResult<Self> {
        assert!(
            precision <= 38,
            "DuckDB decimal type precision must be <= 38. precision: {precision}"
        );

        let ptr = unsafe { duckdb_create_decimal_type(precision, scale) };
        if ptr.is_null() {
            vortex_bail!("Failed to create decimal type");
        }
        Ok(unsafe { Self::own(ptr) })
    }

    /// Creates a DuckDB list logical type with the specified element type.
    pub fn list_type(element_type: LogicalType) -> VortexResult<Self> {
        let ptr = unsafe { duckdb_create_list_type(element_type.as_ptr()) };

        if ptr.is_null() {
            vortex_bail!("Failed to create list type");
        }
        Ok(unsafe { Self::own(ptr) })
    }

    /// Creates a DuckDB fixed-size list logical type with the specified element type and list size.
    ///
    /// Note that DuckDB calls what we call a fixed-size list the ARRAY type.
    pub fn array_type(element_type: LogicalType, list_size: u32) -> VortexResult<Self> {
        // SAFETY: We trust that DuckDB correctly gives us a valid pointer or `NULL`.
        let ptr = unsafe { duckdb_create_array_type(element_type.as_ptr(), list_size as idx_t) };

        if ptr.is_null() {
            vortex_bail!("Failed to create fixed-size list (array) type");
        }

        // SAFETY: This pointer came directly from DuckDB, and we checked that it was not `NULL`.
        Ok(unsafe { Self::own(ptr) })
    }

    /// Converts temporal extension types to corresponding DuckDB types.
    ///
    /// # Arguments
    ///
    /// * `ext_dtype` - A reference to the extension data type containing temporal metadata.
    ///
    /// # Supported Temporal Types
    ///
    /// - **Date**: Must use `TimeUnit::D`
    /// - **Time**: Must use `TimeUnit::Us`
    /// - **Timestamp**: Supports `TimeUnit::Ns`, `Us`, `Ms`, `S`
    pub fn temporal_type(ext_dtype: &ExtDType) -> VortexResult<Self> {
        use vortex::dtype::datetime::TemporalMetadata;
        use vortex::dtype::datetime::TimeUnit;

        let temporal_metadata = TemporalMetadata::try_from(ext_dtype)
            .map_err(|e| vortex_err!("Failed to extract temporal metadata: {}", e))?;

        let duckdb_type = match temporal_metadata {
            TemporalMetadata::Date(TimeUnit::Days) => DUCKDB_TYPE::DUCKDB_TYPE_DATE,
            TemporalMetadata::Date(time_unit) => {
                vortex_bail!("Invalid TimeUnit {} for date", time_unit);
            }
            TemporalMetadata::Time(TimeUnit::Microseconds) => DUCKDB_TYPE::DUCKDB_TYPE_TIME,
            TemporalMetadata::Time(time_unit) => {
                vortex_bail!("Invalid TimeUnit {} for time", time_unit);
            }
            TemporalMetadata::Timestamp(time_unit, tz) => match time_unit {
                TimeUnit::Nanoseconds => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS,
                TimeUnit::Microseconds => {
                    if let Some(tz) = tz {
                        if tz != "UTC" {
                            vortex_bail!("Invalid timezone for timestamp: {tz}");
                        }
                        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_TZ
                    } else {
                        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP
                    }
                }
                TimeUnit::Milliseconds => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS,
                TimeUnit::Seconds => DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S,
                _ => vortex_bail!("Invalid TimeUnit {} for timestamp", time_unit),
            },
        };

        Ok(Self::new(duckdb_type))
    }

    pub fn new_array(element_dtype: DUCKDB_TYPE, array_size: u32) -> Self {
        let element_dtype = Self::new(element_dtype);

        // SAFETY: The element_dtype is created by `Self::new` which ensures it is valid.
        unsafe {
            Self::own(duckdb_create_array_type(
                element_dtype.as_ptr(),
                array_size as idx_t,
            ))
        }
    }

    pub fn as_type_id(&self) -> DUCKDB_TYPE {
        unsafe { duckdb_get_type_id(self.as_ptr()) }
    }

    pub fn varchar() -> Self {
        Self::new(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR)
    }

    pub fn blob() -> Self {
        Self::new(DUCKDB_TYPE::DUCKDB_TYPE_BLOB)
    }

    pub fn int64() -> Self {
        Self::new(DUCKDB_TYPE::DUCKDB_TYPE_BIGINT)
    }

    pub fn uint64() -> Self {
        Self::new(DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT)
    }

    pub fn int32() -> Self {
        Self::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER)
    }

    pub fn bool() -> Self {
        Self::new(DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN)
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

    pub fn array_type_array_size(&self) -> u32 {
        u32::try_from(unsafe { duckdb_array_type_array_size(self.as_ptr()) })
            .vortex_expect("Array size must fit in u32")
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

    pub fn struct_child_type(&self, idx: usize) -> Self {
        unsafe { LogicalType::own(duckdb_struct_type_child_type(self.as_ptr(), idx as idx_t)) }
    }

    pub fn struct_child_name(&self, idx: usize) -> DDBString {
        unsafe { DDBString::own(duckdb_struct_type_child_name(self.as_ptr(), idx as idx_t)) }
    }

    pub fn struct_type_child_count(&self) -> usize {
        usize::try_from(unsafe { duckdb_struct_type_child_count(self.as_ptr()) })
            .vortex_expect("Struct type child count must fit in usize")
    }

    pub fn union_member_type(&self, idx: usize) -> Self {
        unsafe { LogicalType::own(duckdb_union_type_member_type(self.as_ptr(), idx as idx_t)) }
    }

    pub fn union_member_name(&self, idx: usize) -> DDBString {
        unsafe { DDBString::own(duckdb_union_type_member_name(self.as_ptr(), idx as idx_t)) }
    }

    pub fn union_member_count(&self) -> usize {
        usize::try_from(unsafe { duckdb_union_type_member_count(self.as_ptr()) })
            .vortex_expect("Union member count must fit in usize")
    }
}

impl Debug for LogicalType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let debug = unsafe { DDBString::own(duckdb_vx_logical_type_stringify(self.as_ptr())) };
        write!(f, "{}", debug)
    }
}

wrapper!(
    #[derive(Debug)]
    DDBString,
    *mut std::ffi::c_char,
    |ptr: *mut std::ffi::c_char| {
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .map_err(|e| vortex_err!("Failed to convert C string to str: {e}"))
            .vortex_unwrap()
    },
    |ptr: &mut *mut std::ffi::c_char| unsafe { duckdb_free((*ptr).cast()) }
);

impl Display for DDBString {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

impl AsRef<str> for DDBString {
    fn as_ref(&self) -> &str {
        // SAFETY: The string have been validated on construction.
        unsafe { str::from_utf8_unchecked(CStr::from_ptr(self.ptr).to_bytes()) }
    }
}

impl PartialEq for DDBString {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl PartialEq<str> for DDBString {
    fn eq(&self, other: &str) -> bool {
        self.as_ref() == other
    }
}

impl From<DDBString> for FieldName {
    fn from(value: DDBString) -> Self {
        FieldName::from(value.as_ref())
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
            _ => vortex::error::vortex_panic!(
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
        let decimal_type =
            LogicalType::decimal_type(10, 2).vortex_expect("Failed to create decimal type");
        #[allow(clippy::redundant_clone)]
        let cloned = decimal_type.clone();

        assert_eq!(decimal_type.as_type_id(), cloned.as_type_id());

        // Further verify the parameters are preserved.
        let (original_width, original_scale) = decimal_type.as_decimal();

        let (cloned_width, cloned_scale) = cloned.as_decimal();

        assert_eq!(original_width, cloned_width);
        assert_eq!(original_scale, cloned_scale);
    }

    #[test]
    fn test_clone_list_logical_type() {
        // Create a list of integers
        let int_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
        let list_type =
            LogicalType::list_type(int_type).vortex_expect("Failed to create list type");

        #[allow(clippy::redundant_clone)]
        let cloned = list_type.clone();

        assert_eq!(list_type.as_type_id(), cloned.as_type_id());
        assert_eq!(list_type.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_LIST);

        // Verify the child type is preserved
        let original_child = list_type.list_child_type();
        let cloned_child = cloned.list_child_type();

        let original_child_type_id = original_child.as_type_id();
        let cloned_child_type_id = cloned_child.as_type_id();

        assert_eq!(original_child_type_id, cloned_child_type_id);
        assert_eq!(original_child_type_id, DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
    }

    #[test]
    fn test_clone_array_logical_type() {
        // Create an array of strings with size 5
        let array_type =
            LogicalType::array_type(LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR), 5)
                .vortex_expect("Failed to create array type");
        #[allow(clippy::redundant_clone)]
        let cloned = array_type.clone();

        assert_eq!(array_type.as_type_id(), cloned.as_type_id());
        assert_eq!(array_type.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_ARRAY);

        // Verify the child type is preserved
        let original_child = array_type.array_child_type();
        let cloned_child = cloned.array_child_type();

        let original_child_type_id = original_child.as_type_id();
        let cloned_child_type_id = cloned_child.as_type_id();

        assert_eq!(original_child_type_id, cloned_child_type_id);
        assert_eq!(original_child_type_id, DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);

        // Verify the array size is preserved
        let original_size = array_type.array_type_array_size();
        let cloned_size = cloned.array_type_array_size();

        assert_eq!(original_size, cloned_size);
        assert_eq!(original_size, 5);
    }

    #[test]
    fn test_clone_map_logical_type() {
        // Create a map of string -> integer
        let key_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);
        let value_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);
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
        let name_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);
        let age_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);

        let member_types = vec![name_type, age_type];
        let member_names = vec![CString::new("name").unwrap(), CString::new("age").unwrap()];

        let struct_type = LogicalType::struct_type(member_types, member_names)
            .vortex_expect("Failed to create struct type");

        #[allow(clippy::redundant_clone)]
        let cloned = struct_type.clone();

        assert_eq!(struct_type.as_type_id(), cloned.as_type_id());
        assert_eq!(struct_type.as_type_id(), DUCKDB_TYPE::DUCKDB_TYPE_STRUCT);

        // Verify the child count is preserved
        let original_count = struct_type.struct_type_child_count();
        let cloned_count = cloned.struct_type_child_count();
        assert_eq!(original_count, cloned_count);
        assert_eq!(original_count, 2);

        // Verify each field
        for idx in 0..original_count {
            let original_child_type = struct_type.struct_child_type(idx);
            let cloned_child_type = cloned.struct_child_type(idx);
            let original_child_name = struct_type.struct_child_name(idx);
            let cloned_child_name = cloned.struct_child_name(idx);

            assert_eq!(
                original_child_type.as_type_id(),
                cloned_child_type.as_type_id()
            );

            assert_eq!(original_child_name, cloned_child_name);
        }
    }

    #[test]
    fn test_clone_union_logical_type() {
        // Create a union with two members: {str: VARCHAR, num: INTEGER}
        let str_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR);
        let num_type = LogicalType::new(DUCKDB_TYPE::DUCKDB_TYPE_INTEGER);

        let mut member_types = vec![str_type.as_ptr(), num_type.as_ptr()];
        let str_cstr = CString::new("str").unwrap();
        let num_cstr = CString::new("num").unwrap();
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
        let original_count = union_type.union_member_count();
        let cloned_count = cloned.union_member_count();
        assert_eq!(original_count, cloned_count);
        assert_eq!(original_count, 2);

        // Verify each member
        for idx in 0..original_count {
            let original_member_type = union_type.union_member_type(idx);
            let cloned_member_type = cloned.union_member_type(idx);
            let original_member_name = union_type.union_member_name(idx);
            let cloned_member_name = cloned.union_member_name(idx);

            assert_eq!(
                original_member_type.as_type_id(),
                cloned_member_type.as_type_id(),
            );

            assert_eq!(original_member_name, cloned_member_name);
        }
    }
}
