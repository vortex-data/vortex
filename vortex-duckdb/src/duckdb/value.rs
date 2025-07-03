use std::ffi::CStr;
use std::fmt::{Debug, Formatter};

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

    pub fn as_string(&self) -> String {
        unsafe {
            let ptr = cpp::duckdb_get_varchar(self.as_ptr());
            let cstr = CStr::from_ptr(ptr);
            let result = cstr.to_string_lossy().into_owned();
            cpp::duckdb_free(ptr.cast());
            result
        }
    }

    pub fn as_u8(&self) -> u8 {
        unsafe { cpp::duckdb_get_uint8(self.ptr) }
    }

    pub fn as_i32(&self) -> i32 {
        unsafe { cpp::duckdb_get_int32(self.ptr) }
    }

    pub fn as_i64(&self) -> i64 {
        unsafe { cpp::duckdb_get_int64(self.ptr) }
    }

    pub fn as_i128(&self) -> i128 {
        let huge = unsafe { cpp::duckdb_get_hugeint(self.ptr) };
        i128_from_parts(huge.upper, huge.lower)
    }

    pub fn as_date(&self) -> i32 {
        unsafe { cpp::duckdb_get_date(self.ptr).days }
    }

    pub fn as_time(&self) -> i64 {
        unsafe { cpp::duckdb_get_time(self.ptr).micros }
    }

    pub fn as_decimal(&self) -> cpp::duckdb_decimal {
        unsafe { cpp::duckdb_get_decimal(self.ptr) }
    }
}

#[inline]
pub fn i128_from_parts(high: i64, low: u64) -> i128 {
    ((high as i128) << 64) | (low as i128)
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let debug = unsafe { cpp::duckdb_value_to_string(self.as_ptr()) };
        write!(f, "{}", unsafe { CStr::from_ptr(debug).to_string_lossy() })?;
        unsafe { cpp::duckdb_free(debug.cast()) };
        Ok(())
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
