// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

use num_traits::AsPrimitive;
use vortex::buffer::BufferString;
use vortex::buffer::ByteBuffer;
use vortex::dtype::NativeDType;
use vortex::error::VortexError;
use vortex::error::VortexExpect;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;

use crate::cpp;
use crate::cpp::DUCKDB_TYPE;
use crate::cpp::idx_t;
use crate::duckdb::LogicalType;
use crate::duckdb::LogicalTypeRef;
use crate::lifetime_wrapper;

lifetime_wrapper!(Value, cpp::duckdb_value, cpp::duckdb_destroy_value);

impl ValueRef {
    pub fn logical_type(&self) -> &LogicalTypeRef {
        unsafe { LogicalType::borrow(cpp::duckdb_get_value_type(self.as_ptr())) }
    }

    pub fn as_string(&self) -> BufferString {
        let ExtractedValue::Varchar(string) = self.extract() else {
            vortex_panic!("ValueRef is not a string");
        };
        string
    }

    /// Extracts the value from the DuckDB `Value` into a `ExtractedValue`.
    pub fn extract(&self) -> ExtractedValue {
        if unsafe { cpp::duckdb_is_null_value(self.as_ptr()) } {
            return ExtractedValue::Null;
        }
        match self.logical_type().as_type_id() {
            DUCKDB_TYPE::DUCKDB_TYPE_INVALID => vortex_panic!("Invalid type for DuckDB value"),
            DUCKDB_TYPE::DUCKDB_TYPE_SQLNULL => ExtractedValue::Null,
            DUCKDB_TYPE::DUCKDB_TYPE_BOOLEAN => {
                ExtractedValue::Boolean(unsafe { cpp::duckdb_get_bool(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TINYINT => {
                ExtractedValue::TinyInt(unsafe { cpp::duckdb_get_int8(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_SMALLINT => {
                ExtractedValue::SmallInt(unsafe { cpp::duckdb_get_int16(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_INTEGER => {
                ExtractedValue::Integer(unsafe { cpp::duckdb_get_int32(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_BIGINT => {
                ExtractedValue::BigInt(unsafe { cpp::duckdb_get_int64(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_HUGEINT => {
                let huge_int = unsafe { cpp::duckdb_get_hugeint(self.as_ptr()) };
                ExtractedValue::HugeInt(i128_from_parts(huge_int.upper, huge_int.lower))
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UTINYINT => {
                ExtractedValue::UTinyInt(unsafe { cpp::duckdb_get_uint8(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_USMALLINT => {
                ExtractedValue::USmallInt(unsafe { cpp::duckdb_get_uint16(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UINTEGER => {
                ExtractedValue::UInteger(unsafe { cpp::duckdb_get_uint32(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UBIGINT => {
                ExtractedValue::UBigInt(unsafe { cpp::duckdb_get_uint64(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_UHUGEINT => {
                let huge_uint = unsafe { cpp::duckdb_get_uhugeint(self.as_ptr()) };
                ExtractedValue::UHugeInt(u128_from_parts(huge_uint.upper, huge_uint.lower))
            }
            DUCKDB_TYPE::DUCKDB_TYPE_FLOAT => {
                ExtractedValue::Float(unsafe { cpp::duckdb_get_float(self.as_ptr()) })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_DOUBLE => {
                ExtractedValue::Double(unsafe { cpp::duckdb_get_double(self.as_ptr()) })
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
                ExtractedValue::Varchar(string)
            }
            DUCKDB_TYPE::DUCKDB_TYPE_BLOB => {
                let blob = unsafe { cpp::duckdb_get_blob(self.as_ptr()) };
                let slice =
                    unsafe { std::slice::from_raw_parts(blob.data.cast::<u8>(), blob.size.as_()) };
                let bytes = ByteBuffer::copy_from(slice);
                unsafe { cpp::duckdb_free(blob.data) };
                ExtractedValue::Blob(bytes)
            }
            DUCKDB_TYPE::DUCKDB_TYPE_DATE => {
                ExtractedValue::Date(unsafe { cpp::duckdb_get_date(self.as_ptr()).days })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIME => {
                ExtractedValue::Time(unsafe { cpp::duckdb_get_time(self.as_ptr()).micros })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIME_NS => {
                ExtractedValue::Time(unsafe { cpp::duckdb_get_time_ns(self.as_ptr()).nanos })
            }
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS => ExtractedValue::TimestampNs(unsafe {
                cpp::duckdb_get_timestamp_ns(self.as_ptr()).nanos
            }),
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP => ExtractedValue::Timestamp(unsafe {
                cpp::duckdb_get_timestamp(self.as_ptr()).micros
            }),
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS => ExtractedValue::TimestampMs(unsafe {
                cpp::duckdb_get_timestamp_ms(self.as_ptr()).millis
            }),
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S => ExtractedValue::TimestampS(unsafe {
                cpp::duckdb_get_timestamp_s(self.as_ptr()).seconds
            }),
            DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_TZ => ExtractedValue::TimestampS(unsafe {
                cpp::duckdb_get_timestamp_tz(self.as_ptr()).micros
            }),
            DUCKDB_TYPE::DUCKDB_TYPE_DECIMAL => {
                let decimal = unsafe { cpp::duckdb_get_decimal(self.as_ptr()) };
                let value = i128_from_parts(decimal.value.upper, decimal.value.lower);
                ExtractedValue::Decimal(
                    decimal.width,
                    i8::try_from(decimal.scale).vortex_expect("invalid scale"),
                    value,
                )
            }
            DUCKDB_TYPE::DUCKDB_TYPE_LIST => {
                let elem_count =
                    usize::try_from(unsafe { cpp::duckdb_get_list_size(self.as_ptr()) })
                        .vortex_expect("List size must fit usize");
                ExtractedValue::List(
                    (0..elem_count)
                        .map(|i| unsafe {
                            Value::own(cpp::duckdb_get_list_child(self.as_ptr(), i as idx_t))
                        })
                        .collect::<Vec<_>>(),
                )
            }
            DUCKDB_TYPE::DUCKDB_TYPE_STRUCT => ExtractedValue::List(
                (0..self.logical_type().struct_type_child_count())
                    .map(|i| unsafe {
                        Value::own(cpp::duckdb_get_struct_child(self.as_ptr(), i as idx_t))
                    })
                    .collect::<Vec<_>>(),
            ),
            // ...other types remain unimplemented..
            other => vortex_panic!("Unsupported DuckDB value type {other:?}"),
        }
    }
}

impl Debug for ValueRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ptr = unsafe { cpp::duckdb_value_to_string(self.as_ptr()) };
        write!(f, "{}", unsafe { CStr::from_ptr(ptr).to_string_lossy() })?;
        unsafe { cpp::duckdb_free(ptr.cast()) };
        Ok(())
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl Value {
    pub fn sql_null() -> Self {
        unsafe { Self::own(cpp::duckdb_create_null_value()) }
    }

    pub fn null(logical_type: &LogicalTypeRef) -> Self {
        unsafe { Self::own(cpp::duckdb_vx_value_create_null(logical_type.as_ptr())) }
    }

    pub fn new_decimal(precision: u8, scale: i8, value: i128) -> Self {
        unsafe {
            Self::own(cpp::duckdb_create_decimal(cpp::duckdb_decimal {
                width: precision,
                scale: scale.cast_unsigned(),
                value: cpp::duckdb_hugeint {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "intentional truncation to lower 64 bits"
                    )]
                    lower: value as u64,
                    upper: (value >> 64) as i64,
                },
            }))
        }
    }

    pub fn new_timestamp_tz(micros: i64) -> Self {
        unsafe {
            Self::own(cpp::duckdb_create_timestamp_tz(cpp::duckdb_timestamp {
                micros,
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
}

impl Display for ValueRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ptr = unsafe { cpp::duckdb_vx_value_to_string(self.as_ptr()) };
        write!(f, "{}", unsafe { CStr::from_ptr(ptr) }.to_string_lossy())?;
        unsafe { cpp::duckdb_free(ptr.cast()) };
        Ok(())
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&**self, f)
    }
}

#[inline]
pub fn i128_from_parts(high: i64, low: u64) -> i128 {
    ((high as i128) << 64) | (low as i128)
}

#[inline]
pub fn u128_from_parts(high: u64, low: u64) -> u128 {
    ((high as u128) << 64) | (low as u128)
}

impl<T> TryFrom<Option<T>> for Value
where
    T: Into<Value> + NativeDType,
{
    type Error = VortexError;

    fn try_from(value: Option<T>) -> Result<Self, Self::Error> {
        match value {
            Some(v) => Ok(v.into()),
            None => {
                let lt = LogicalType::try_from(&T::dtype())?;
                Ok(Value::null(&lt))
            }
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
pub enum ExtractedValue {
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
    UHugeInt(u128),
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
    List(Vec<Value>),
}

#[cfg(test)]
mod tests {
    use crate::duckdb::i128_from_parts;
    use crate::duckdb::u128_from_parts;

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

    #[test]
    fn test_uhuge_int_from_parts() {
        assert_eq!(u128_from_parts(0, 0), 0u128);
        assert_eq!(u128_from_parts(0, 34534912), 34534912u128);
        assert_eq!(u128_from_parts(0, u64::MAX), u64::MAX as u128);
        assert_eq!(u128_from_parts(u64::MAX, u64::MAX), u128::MAX);
        assert_eq!(
            u128_from_parts(1, u64::MAX),
            (1u128 << 64) + (u64::MAX as u128)
        );
        assert_eq!(u128_from_parts(1, 0), 1u128 << 64);
    }
}
