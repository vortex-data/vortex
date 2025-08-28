// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::fmt::{Debug, Display, Formatter};

use num_traits::AsPrimitive;
use vortex::buffer::{BufferString, ByteBuffer};
use vortex::error::{VortexExpect, vortex_err, vortex_panic};

use crate::cpp::DUCKDB_TYPE;
use crate::duckdb::LogicalType;
use crate::{cpp, wrapper};

wrapper!(Value, cpp::duckdb_value, cpp::duckdb_destroy_value);

impl Value {
    pub fn null() -> Self {
        unsafe { Self::own(cpp::duckdb_create_null_value()) }
    }

    /// Note the lifetime of logical type if tied to &self
    pub fn logical_type(&self) -> LogicalType {
        unsafe { LogicalType::borrow(cpp::duckdb_get_value_type(self.as_ptr())) }
    }

    pub fn new_decimal(precision: u8, scale: i8, value: i128) -> Self {
        unsafe {
            Self::own(cpp::duckdb_create_decimal(cpp::duckdb_decimal {
                width: precision,
                scale: scale.cast_unsigned(),
                value: cpp::duckdb_hugeint {
                    // We want to truncate
                    #[allow(clippy::cast_possible_truncation)]
                    lower: value as u64,
                    upper: (value >> 64) as i64,
                },
            }))
        }
    }

    pub fn new_timestamp_ns(nanos: i64) -> Self {
        unsafe {
            Self::own(cpp::duckdb_create_timestamp_ns(cpp::duckdb_timestamp_ns {
                nanos,
            }))
        }
    }

    pub fn new_timestamp_us(micros: i64) -> Self {
        unsafe {
            Self::own(cpp::duckdb_create_timestamp(cpp::duckdb_timestamp {
                micros,
            }))
        }
    }

    pub fn new_timestamp_ms(millis: i64) -> Self {
        unsafe {
            Self::own(cpp::duckdb_create_timestamp_ms(cpp::duckdb_timestamp_ms {
                millis,
            }))
        }
    }

    pub fn new_timestamp_s(seconds: i64) -> Self {
        unsafe {
            Self::own(cpp::duckdb_create_timestamp_s(cpp::duckdb_timestamp_s {
                seconds,
            }))
        }
    }

    pub fn new_time(micros: i64) -> Self {
        unsafe { Self::own(cpp::duckdb_create_time(cpp::duckdb_time { micros })) }
    }

    pub fn new_date(days: i32) -> Self {
        unsafe { Self::own(cpp::duckdb_create_date(cpp::duckdb_date { days })) }
    }

    pub fn as_string(&self) -> BufferString {
        let Val::Varchar(string) = self.extract() else {
            vortex_panic!("Value is not a string");
        };
        string
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let debug = unsafe { cpp::duckdb_vx_value_to_string(self.as_ptr()) };
        let str = unsafe { CStr::from_ptr(debug) }
            .to_string_lossy()
            .to_string();
        f.write_str(&str)?;
        Ok(())
    }
}

#[inline]
pub fn i128_from_parts(high: i64, low: u64) -> i128 {
    ((high as i128) << 64) | (low as i128)
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string())
    }
}

impl<T> From<Option<T>> for Value
where
    T: Into<Value>,
{
    fn from(value: Option<T>) -> Self {
        match value {
            Some(v) => v.into(),
            None => Value::null(),
        }
    }
}

impl From<i8> for Value {
    fn from(value: i8) -> Self {
        unsafe { Self::own(cpp::duckdb_create_int8(value)) }
    }
}

impl From<i16> for Value {
    fn from(value: i16) -> Self {
        unsafe { Self::own(cpp::duckdb_create_int16(value)) }
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        unsafe { Self::own(cpp::duckdb_create_int32(value)) }
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        unsafe { Self::own(cpp::duckdb_create_int64(value)) }
    }
}

impl From<u8> for Value {
    fn from(value: u8) -> Self {
        unsafe { Self::own(cpp::duckdb_create_uint8(value)) }
    }
}

impl From<u16> for Value {
    fn from(value: u16) -> Self {
        unsafe { Self::own(cpp::duckdb_create_uint16(value)) }
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        unsafe { Self::own(cpp::duckdb_create_uint32(value)) }
    }
}

impl From<u64> for Value {
    fn from(value: u64) -> Self {
        unsafe { Self::own(cpp::duckdb_create_uint64(value)) }
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        unsafe { Self::own(cpp::duckdb_create_float(value)) }
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        unsafe { Self::own(cpp::duckdb_create_double(value)) }
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        unsafe { Self::own(cpp::duckdb_create_bool(value)) }
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        unsafe {
            Self::own(cpp::duckdb_create_varchar_length(
                value.as_ptr().cast(),
                value.len() as _,
            ))
        }
    }
}

impl From<&[u8]> for Value {
    fn from(value: &[u8]) -> Self {
        unsafe { Self::own(cpp::duckdb_create_blob(value.as_ptr(), value.len() as _)) }
    }
}

/// An enum for extracting the underlying typed value from a `Value`.
pub enum Val {
    Null,
    TinyInt(i8),
    SmallInt(i16),
    Integer(i32),
    BigInt(i64),
    HugeInt(i128),
    UTinyInt(u8),
    USmallInt(u16),
    UInteger(u32),
    UBigInt(u64),
    Float(f32),
    Double(f64),
    Boolean(bool),
    Varchar(BufferString),
    Blob(ByteBuffer),
    Date(i32),
    Time(i64),
    TimestampNs(i64),
    Timestamp(i64),
    TimestampMs(i64),
    TimestampS(i64),
    Decimal(u8, i8, i128),
}

impl Value {
    /// Extracts the value from the DuckDB `Value` into a `ValueRef`.
    pub fn extract(&self) -> Val {
        if unsafe { cpp::duckdb_is_null_value(self.as_ptr()) } {
            return Val::Null;
        }
        match self.logical_type().as_type_id() {
            DUCKDB_TYPE::DUCKDB_TYPE_INVALID => vortex_panic!("Invalid type for DuckDB value"),
            DUCKDB_TYPE::DUCKDB_TYPE_SQLNULL => Val::Null,
            DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN => {
                Val::Boolean(unsafe { cpp::duckdb_get_bool(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => {
                Val::TinyInt(unsafe { cpp::duckdb_get_int8(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => {
                Val::SmallInt(unsafe { cpp::duckdb_get_int16(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_INTEGER => {
                Val::Integer(unsafe { cpp::duckdb_get_int32(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_BIGINT => {
                Val::BigInt(unsafe { cpp::duckdb_get_int64(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT => {
                Val::UTinyInt(unsafe { cpp::duckdb_get_uint8(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT => {
                Val::USmallInt(unsafe { cpp::duckdb_get_uint16(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER => {
                Val::UInteger(unsafe { cpp::duckdb_get_uint32(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT => {
                Val::UBigInt(unsafe { cpp::duckdb_get_uint64(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => {
                Val::Float(unsafe { cpp::duckdb_get_float(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => {
                Val::Double(unsafe { cpp::duckdb_get_double(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_VARCHAR => {
                let ptr = unsafe { cpp::duckdb_get_varchar(self.as_ptr()) };
                let cstr = unsafe { CStr::from_ptr(ptr) };
                let string = BufferString::from(
                    cstr.to_str()
                        .map_err(|e| vortex_err!("Invalid UTF-8 string from DuckDB: {e}"))
                        .vortex_expect("Invalid UTF-8 string from DuckDB"),
                );
                unsafe { cpp::duckdb_free(ptr.cast()) };
                Val::Varchar(string)
            }
            DUCKDB_TYPE::DUCKDB_TYPE_BLOB => {
                // TODO(ngates): for blobs and strings, we could write our own C functions to
                //  get the values by reference, avoiding a double copy since these C functions
                //  also copy on the CPP side.
                let blob = unsafe { cpp::duckdb_get_blob(self.as_ptr()) };
                let slice =
                    unsafe { std::slice::from_raw_parts(blob.data.cast::<u8>(), blob.size.as_()) };
                let bytes = ByteBuffer::copy_from(slice);
                unsafe { cpp::duckdb_free(blob.data) };
                Val::Blob(bytes)
            }
            DUCKDB_TYPE::DUCKDB_TYPE_DATE => {
                Val::Date(unsafe { cpp::duckdb_get_date(self.as_ptr()).days })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIME => {
                Val::Time(unsafe { cpp::duckdb_get_time(self.as_ptr()).micros })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS => {
                Val::TimestampNs(unsafe { cpp::duckdb_get_timestamp_ns(self.as_ptr()).nanos })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP => {
                Val::Timestamp(unsafe { cpp::duckdb_get_timestamp(self.as_ptr()).micros })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS => {
                Val::TimestampMs(unsafe { cpp::duckdb_get_timestamp_ms(self.as_ptr()).millis })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S => {
                Val::TimestampS(unsafe { cpp::duckdb_get_timestamp_s(self.as_ptr()).seconds })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_DECIMAL => {
                let decimal = unsafe { cpp::duckdb_get_decimal(self.as_ptr()) };
                let value = i128_from_parts(decimal.value.upper, decimal.value.lower);
                Val::Decimal(
                    decimal.width,
                    i8::try_from(decimal.scale).vortex_expect("invalid scale"),
                    value,
                )
            }
            // ...other types remain unimplemented...
            _ => vortex_panic!("Unsupported DuckDB value type {:?}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::duckdb::i128_from_parts;

    #[test]
    fn test_huge_int_from_parts() {
        assert_eq!(i128_from_parts(0, 0), 0i128);
        assert_eq!(i128_from_parts(0, 34534912), 34534912i128);
        assert_eq!(i128_from_parts(i64::MIN, 0), i128::MIN);
        assert_eq!(i128_from_parts(i64::MAX, u64::MAX), i128::MAX);

        assert_eq!(i128_from_parts(0, u64::MAX), u64::MAX as i128);
        assert_eq!(
            i128_from_parts(1, u64::MAX),
            (1i128 << 64) + (u64::MAX as i128)
        );
    }
}
