// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt::Display;
use std::cmp::Ordering;
use std::hash::Hash;
use std::hash::Hasher;

use num_traits::NumCast;
use num_traits::ToPrimitive;
use paste::paste;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::ToBytes;
use vortex_dtype::half::f16;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

/// A primitive value that can represent any primitive type supported by Vortex.
///
/// `PValue` is used to store primitive scalar values in a type-erased manner,
/// supporting all primitive types (integers, floats) with various bit widths.
#[derive(Debug, Clone, Copy)]
pub enum PValue {
    /// Unsigned 8-bit integer.
    U8(u8),
    /// Unsigned 16-bit integer.
    U16(u16),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 8-bit integer.
    I8(i8),
    /// Signed 16-bit integer.
    I16(i16),
    /// Signed 32-bit integer.
    I32(i32),
    /// Signed 64-bit integer.
    I64(i64),
    /// 16-bit floating point.
    F16(f16),
    /// 32-bit floating point.
    F32(f32),
    /// 64-bit floating point.
    F64(f64),
}

impl PartialEq for PValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::U8(s), o) => o.as_u64().vortex_expect("upcast") == *s as u64,
            (Self::U16(s), o) => o.as_u64().vortex_expect("upcast") == *s as u64,
            (Self::U32(s), o) => o.as_u64().vortex_expect("upcast") == *s as u64,
            (Self::U64(s), o) => o.as_u64().vortex_expect("upcast") == *s,
            (Self::I8(s), o) => o.as_i64().vortex_expect("upcast") == *s as i64,
            (Self::I16(s), o) => o.as_i64().vortex_expect("upcast") == *s as i64,
            (Self::I32(s), o) => o.as_i64().vortex_expect("upcast") == *s as i64,
            (Self::I64(s), o) => o.as_i64().vortex_expect("upcast") == *s,
            (Self::F16(s), Self::F16(o)) => s.is_eq(*o),
            (Self::F32(s), Self::F32(o)) => s.is_eq(*o),
            (Self::F64(s), Self::F64(o)) => s.is_eq(*o),
            (..) => false,
        }
    }
}

impl Eq for PValue {}

impl PartialOrd for PValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::U8(s), o) => Some((*s as u64).cmp(&o.as_u64().vortex_expect("upcast"))),
            (Self::U16(s), o) => Some((*s as u64).cmp(&o.as_u64().vortex_expect("upcast"))),
            (Self::U32(s), o) => Some((*s as u64).cmp(&o.as_u64().vortex_expect("upcast"))),
            (Self::U64(s), o) => Some((*s).cmp(&o.as_u64().vortex_expect("upcast"))),
            (Self::I8(s), o) => Some((*s as i64).cmp(&o.as_i64().vortex_expect("upcast"))),
            (Self::I16(s), o) => Some((*s as i64).cmp(&o.as_i64().vortex_expect("upcast"))),
            (Self::I32(s), o) => Some((*s as i64).cmp(&o.as_i64().vortex_expect("upcast"))),
            (Self::I64(s), o) => Some((*s).cmp(&o.as_i64().vortex_expect("upcast"))),
            (Self::F16(s), Self::F16(o)) => Some(s.total_compare(*o)),
            (Self::F32(s), Self::F32(o)) => Some(s.total_compare(*o)),
            (Self::F64(s), Self::F64(o)) => Some(s.total_compare(*o)),
            (..) => None,
        }
    }
}

impl Hash for PValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            PValue::U8(_) | PValue::U16(_) | PValue::U32(_) | PValue::U64(_) => {
                self.as_u64().vortex_expect("upcast").hash(state)
            }
            PValue::I8(_) | PValue::I16(_) | PValue::I32(_) | PValue::I64(_) => {
                self.as_i64().vortex_expect("upcast").hash(state)
            }
            PValue::F16(v) => v.to_le_bytes().hash(state),
            PValue::F32(v) => v.to_le_bytes().hash(state),
            PValue::F64(v) => v.to_le_bytes().hash(state),
        }
    }
}

impl ToBytes for PValue {
    fn to_le_bytes(&self) -> &[u8] {
        match self {
            PValue::U8(v) => v.to_le_bytes(),
            PValue::U16(v) => v.to_le_bytes(),
            PValue::U32(v) => v.to_le_bytes(),
            PValue::U64(v) => v.to_le_bytes(),
            PValue::I8(v) => v.to_le_bytes(),
            PValue::I16(v) => v.to_le_bytes(),
            PValue::I32(v) => v.to_le_bytes(),
            PValue::I64(v) => v.to_le_bytes(),
            PValue::F16(v) => v.to_le_bytes(),
            PValue::F32(v) => v.to_le_bytes(),
            PValue::F64(v) => v.to_le_bytes(),
        }
    }
}

macro_rules! as_primitive {
    ($T:ty, $PT:tt) => {
        paste! {
            #[doc = "Access PValue as `" $T "`, returning `None` if conversion is unsuccessful"]
            pub fn [<as_ $T>](self) -> Option<$T> {
                <$T>::try_from(self).ok()
            }
        }
    };
}

impl PValue {
    /// Creates a zero value for the given primitive type.
    pub fn zero(ptype: PType) -> PValue {
        match ptype {
            PType::U8 => PValue::U8(0),
            PType::U16 => PValue::U16(0),
            PType::U32 => PValue::U32(0),
            PType::U64 => PValue::U64(0),
            PType::I8 => PValue::I8(0),
            PType::I16 => PValue::I16(0),
            PType::I32 => PValue::I32(0),
            PType::I64 => PValue::I64(0),
            PType::F16 => PValue::F16(f16::ZERO),
            PType::F32 => PValue::F32(0.0),
            PType::F64 => PValue::F64(0.0),
        }
    }

    /// Returns the primitive type of this value.
    pub fn ptype(&self) -> PType {
        match self {
            Self::U8(_) => PType::U8,
            Self::U16(_) => PType::U16,
            Self::U32(_) => PType::U32,
            Self::U64(_) => PType::U64,
            Self::I8(_) => PType::I8,
            Self::I16(_) => PType::I16,
            Self::I32(_) => PType::I32,
            Self::I64(_) => PType::I64,
            Self::F16(_) => PType::F16,
            Self::F32(_) => PType::F32,
            Self::F64(_) => PType::F64,
        }
    }

    /// Returns true if this value is of the given primitive type.
    pub fn is_instance_of(&self, ptype: &PType) -> bool {
        &self.ptype() == ptype
    }

    /// Converts this value to a specific native primitive type.
    ///
    /// Panics if the conversion is not supported or would overflow.
    #[inline]
    pub fn cast<T: NativePType>(&self) -> T {
        self.cast_opt::<T>().vortex_expect("as_primitive")
    }

    /// Converts this value to a specific native primitive type.
    ///
    /// Returns `None` if the conversion is not supported or would overflow.
    #[inline]
    pub fn cast_opt<T: NativePType>(&self) -> Option<T> {
        match *self {
            PValue::U8(u) => T::from_u8(u),
            PValue::U16(u) => T::from_u16(u),
            PValue::U32(u) => T::from_u32(u),
            PValue::U64(u) => T::from_u64(u),
            PValue::I8(i) => T::from_i8(i),
            PValue::I16(i) => T::from_i16(i),
            PValue::I32(i) => T::from_i32(i),
            PValue::I64(i) => T::from_i64(i),
            PValue::F16(f) => T::from_f16(f),
            PValue::F32(f) => T::from_f32(f),
            PValue::F64(f) => T::from_f64(f),
        }
    }

    /// Returns true if the value of float type and is NaN.
    pub fn is_nan(&self) -> bool {
        match self {
            PValue::F16(f) => f.is_nan(),
            PValue::F32(f) => f.is_nan(),
            PValue::F64(f) => f.is_nan(),
            _ => false,
        }
    }

    /// Reinterprets the bits of this value as a different primitive type.
    ///
    /// This performs a bitwise cast between types of the same width.
    ///
    /// # Panics
    ///
    /// Panics if the target type has a different byte width than this value.
    pub fn reinterpret_cast(&self, ptype: PType) -> Self {
        if ptype == self.ptype() {
            return *self;
        }

        assert_eq!(
            ptype.byte_width(),
            self.ptype().byte_width(),
            "Cannot reinterpret cast between types of different widths"
        );

        match self {
            PValue::U8(v) => u8::cast_signed(*v).into(),
            PValue::U16(v) => match ptype {
                PType::I16 => u16::cast_signed(*v).into(),
                PType::F16 => f16::from_bits(*v).into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::U32(v) => match ptype {
                PType::I32 => u32::cast_signed(*v).into(),
                PType::F32 => f32::from_bits(*v).into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::U64(v) => match ptype {
                PType::I64 => u64::cast_signed(*v).into(),
                PType::F64 => f64::from_bits(*v).into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::I8(v) => i8::cast_unsigned(*v).into(),
            PValue::I16(v) => match ptype {
                PType::U16 => i16::cast_unsigned(*v).into(),
                PType::F16 => f16::from_bits(v.cast_unsigned()).into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::I32(v) => match ptype {
                PType::U32 => i32::cast_unsigned(*v).into(),
                PType::F32 => f32::from_bits(i32::cast_unsigned(*v)).into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::I64(v) => match ptype {
                PType::U64 => i64::cast_unsigned(*v).into(),
                PType::F64 => f64::from_bits(i64::cast_unsigned(*v)).into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::F16(v) => match ptype {
                PType::U16 => v.to_bits().into(),
                PType::I16 => v.to_bits().cast_signed().into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::F32(v) => match ptype {
                PType::U32 => f32::to_bits(*v).into(),
                PType::I32 => f32::to_bits(*v).cast_signed().into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::F64(v) => match ptype {
                PType::U64 => f64::to_bits(*v).into(),
                PType::I64 => f64::to_bits(*v).cast_signed().into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
        }
    }

    as_primitive!(i8, I8);
    as_primitive!(i16, I16);
    as_primitive!(i32, I32);
    as_primitive!(i64, I64);
    as_primitive!(u8, U8);
    as_primitive!(u16, U16);
    as_primitive!(u32, U32);
    as_primitive!(u64, U64);
    as_primitive!(f16, F16);
    as_primitive!(f32, F32);
    as_primitive!(f64, F64);
}

macro_rules! int_pvalue {
    ($T:ty, $PT:tt) => {
        impl TryFrom<PValue> for $T {
            type Error = VortexError;

            fn try_from(value: PValue) -> Result<Self, Self::Error> {
                match value {
                    PValue::U8(_)
                    | PValue::U16(_)
                    | PValue::U32(_)
                    | PValue::U64(_)
                    | PValue::I8(_)
                    | PValue::I16(_)
                    | PValue::I32(_)
                    | PValue::I64(_) => Some(value),
                    _ => None,
                }
                .and_then(|v| PValue::cast_opt(&v))
                .ok_or_else(|| {
                    vortex_err!("Cannot read primitive value {:?} as {}", value, PType::$PT)
                })
            }
        }
    };
}

macro_rules! float_pvalue {
    ($T:ty, $PT:tt) => {
        impl TryFrom<PValue> for $T {
            type Error = VortexError;

            fn try_from(value: PValue) -> Result<Self, Self::Error> {
                value.cast_opt().ok_or_else(|| {
                    vortex_err!("Cannot read primitive value {:?} as {}", value, PType::$PT)
                })
            }
        }
    };
}

impl TryFrom<PValue> for usize {
    type Error = VortexError;

    fn try_from(value: PValue) -> Result<Self, Self::Error> {
        value
            .cast_opt::<u64>()
            .and_then(|v| v.to_usize())
            .ok_or_else(|| vortex_err!("Cannot read primitive value {:?} as usize", value))
    }
}

int_pvalue!(u8, U8);
int_pvalue!(u16, U16);
int_pvalue!(u32, U32);
int_pvalue!(u64, U64);
int_pvalue!(i8, I8);
int_pvalue!(i16, I16);
int_pvalue!(i32, I32);
int_pvalue!(i64, I64);

float_pvalue!(f16, F16);
float_pvalue!(f32, F32);
float_pvalue!(f64, F64);

macro_rules! impl_pvalue {
    ($T:ty, $PT:tt) => {
        impl From<$T> for PValue {
            fn from(value: $T) -> Self {
                PValue::$PT(value)
            }
        }
    };
}

impl_pvalue!(u8, U8);
impl_pvalue!(u16, U16);
impl_pvalue!(u32, U32);
impl_pvalue!(u64, U64);
impl_pvalue!(i8, I8);
impl_pvalue!(i16, I16);
impl_pvalue!(i32, I32);
impl_pvalue!(i64, I64);
impl_pvalue!(f16, F16);
impl_pvalue!(f32, F32);
impl_pvalue!(f64, F64);

impl From<usize> for PValue {
    #[inline]
    fn from(value: usize) -> PValue {
        PValue::U64(value as u64)
    }
}

impl Display for PValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U8(v) => write!(f, "{v}u8"),
            Self::U16(v) => write!(f, "{v}u16"),
            Self::U32(v) => write!(f, "{v}u32"),
            Self::U64(v) => write!(f, "{v}u64"),
            Self::I8(v) => write!(f, "{v}i8"),
            Self::I16(v) => write!(f, "{v}i16"),
            Self::I32(v) => write!(f, "{v}i32"),
            Self::I64(v) => write!(f, "{v}i64"),
            Self::F16(v) => write!(f, "{v}f16"),
            Self::F32(v) => write!(f, "{v}f32"),
            Self::F64(v) => write!(f, "{v}f64"),
        }
    }
}

pub(super) trait CoercePValue: Sized {
    /// Coerce value from a compatible bit representation using into given type.
    ///
    /// Integers can be widened from narrower type
    /// Floats stored as integers will be reinterpreted as bit representation of the float
    fn coerce(value: PValue) -> VortexResult<Self>;
}

macro_rules! int_coerce {
    ($T:ty) => {
        impl CoercePValue for $T {
            #[inline]
            fn coerce(value: PValue) -> VortexResult<Self> {
                Self::try_from(value)
            }
        }
    };
}

int_coerce!(u8);
int_coerce!(u16);
int_coerce!(u32);
int_coerce!(u64);
int_coerce!(i8);
int_coerce!(i16);
int_coerce!(i32);
int_coerce!(i64);

impl CoercePValue for f16 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "truncation is intentional and checked where needed"
    )]
    fn coerce(value: PValue) -> VortexResult<Self> {
        // F16 coercion behavior:
        // - U8/U16/U32/U64: Interpreted as the bit representation of an f16 value.
        //   Only the lower 16 bits are used, allowing compact storage of f16 values
        //   as integers when the full type information is preserved externally.
        // - F16: Passthrough
        // - F32/F64: Numeric conversion with potential precision loss
        // - Other types: Not supported
        //
        // Note: This bit-pattern interpretation means that integer value 0x3C00u16
        // would be interpreted as f16(1.0), not as f16(15360.0).
        match value {
            PValue::U8(u) => Ok(Self::from_bits(u as u16)),
            PValue::U16(u) => Ok(Self::from_bits(u)),
            PValue::U32(u) => {
                vortex_ensure!(
                    u <= u16::MAX as u32,
                    "Cannot coerce U32 value to f16: value out of range"
                );
                Ok(Self::from_bits(u as u16))
            }
            PValue::U64(u) => {
                vortex_ensure!(
                    u <= u16::MAX as u64,
                    "Cannot coerce U64 value to f16: value out of range"
                );
                Ok(Self::from_bits(u as u16))
            }
            PValue::F16(u) => Ok(u),
            PValue::F32(f) => {
                <Self as NumCast>::from(f).ok_or_else(|| vortex_err!("Cannot convert f32 to f16"))
            }
            PValue::F64(f) => {
                <Self as NumCast>::from(f).ok_or_else(|| vortex_err!("Cannot convert f64 to f16"))
            }
            PValue::I8(_) | PValue::I16(_) | PValue::I32(_) | PValue::I64(_) => {
                vortex_bail!("Cannot coerce {value:?} to f16: type not supported for coercion")
            }
        }
    }
}

impl CoercePValue for f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "truncation is intentional and checked where needed"
    )]
    fn coerce(value: PValue) -> VortexResult<Self> {
        // F32 coercion: U32 values are interpreted as bit patterns, not numeric conversions
        match value {
            PValue::U8(u) => Ok(Self::from_bits(u as u32)),
            PValue::U16(u) => Ok(Self::from_bits(u as u32)),
            PValue::U32(u) => Ok(Self::from_bits(u)),
            PValue::U64(u) => {
                vortex_ensure!(
                    u <= u32::MAX as u64,
                    "Cannot coerce U64 value to f32: value out of range"
                );
                Ok(Self::from_bits(u as u32))
            }
            PValue::F16(f) => {
                <Self as NumCast>::from(f).ok_or_else(|| vortex_err!("Cannot convert f16 to f32"))
            }
            PValue::F32(f) => Ok(f),
            PValue::F64(f) => {
                <Self as NumCast>::from(f).ok_or_else(|| vortex_err!("Cannot convert f64 to f32"))
            }
            PValue::I8(_) | PValue::I16(_) | PValue::I32(_) | PValue::I64(_) => {
                vortex_bail!("Unsupported PValue {value:?} type for f32")
            }
        }
    }
}

impl CoercePValue for f64 {
    fn coerce(value: PValue) -> VortexResult<Self> {
        // F64 coercion: U64 values are interpreted as bit patterns, not numeric conversions
        match value {
            PValue::U8(u) => Ok(Self::from_bits(u as u64)),
            PValue::U16(u) => Ok(Self::from_bits(u as u64)),
            PValue::U32(u) => Ok(Self::from_bits(u as u64)),
            PValue::U64(u) => Ok(Self::from_bits(u)),
            PValue::F16(f) => {
                <Self as NumCast>::from(f).ok_or_else(|| vortex_err!("Cannot convert f16 to f64"))
            }
            PValue::F32(f) => {
                <Self as NumCast>::from(f).ok_or_else(|| vortex_err!("Cannot convert f32 to f64"))
            }
            PValue::F64(f) => Ok(f),
            PValue::I8(_) | PValue::I16(_) | PValue::I32(_) | PValue::I64(_) => {
                vortex_bail!("Unsupported PValue {value:?} type for f64")
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::cmp::Ordering;

    use vortex_dtype::FromPrimitiveOrF16;
    use vortex_dtype::PType;
    use vortex_dtype::ToBytes;
    use vortex_dtype::half::f16;
    use vortex_utils::aliases::hash_set::HashSet;

    use crate::PValue;
    use crate::pvalue::CoercePValue;

    #[test]
    pub fn test_is_instance_of() {
        assert!(PValue::U8(10).is_instance_of(&PType::U8));
        assert!(!PValue::U8(10).is_instance_of(&PType::U16));
        assert!(!PValue::U8(10).is_instance_of(&PType::I8));
        assert!(!PValue::U8(10).is_instance_of(&PType::F16));

        assert!(PValue::I8(10).is_instance_of(&PType::I8));
        assert!(!PValue::I8(10).is_instance_of(&PType::I16));
        assert!(!PValue::I8(10).is_instance_of(&PType::U8));
        assert!(!PValue::I8(10).is_instance_of(&PType::F16));

        assert!(PValue::F16(f16::from_f32(10.0)).is_instance_of(&PType::F16));
        assert!(!PValue::F16(f16::from_f32(10.0)).is_instance_of(&PType::F32));
        assert!(!PValue::F16(f16::from_f32(10.0)).is_instance_of(&PType::U16));
        assert!(!PValue::F16(f16::from_f32(10.0)).is_instance_of(&PType::I16));
    }

    #[test]
    fn test_compare_different_types() {
        assert_eq!(
            PValue::I8(4).partial_cmp(&PValue::I8(5)),
            Some(Ordering::Less)
        );
        assert_eq!(
            PValue::I8(4).partial_cmp(&PValue::I64(5)),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn test_hash() {
        let set = HashSet::from([
            PValue::U8(1),
            PValue::U16(1),
            PValue::U32(1),
            PValue::U64(1),
            PValue::I8(1),
            PValue::I16(1),
            PValue::I32(1),
            PValue::I64(1),
            PValue::I8(-1),
            PValue::I16(-1),
            PValue::I32(-1),
            PValue::I64(-1),
        ]);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_zero_values() {
        assert_eq!(PValue::zero(PType::U8), PValue::U8(0));
        assert_eq!(PValue::zero(PType::U16), PValue::U16(0));
        assert_eq!(PValue::zero(PType::U32), PValue::U32(0));
        assert_eq!(PValue::zero(PType::U64), PValue::U64(0));
        assert_eq!(PValue::zero(PType::I8), PValue::I8(0));
        assert_eq!(PValue::zero(PType::I16), PValue::I16(0));
        assert_eq!(PValue::zero(PType::I32), PValue::I32(0));
        assert_eq!(PValue::zero(PType::I64), PValue::I64(0));
        assert_eq!(PValue::zero(PType::F16), PValue::F16(f16::from_f32(0.0)));
        assert_eq!(PValue::zero(PType::F32), PValue::F32(0.0));
        assert_eq!(PValue::zero(PType::F64), PValue::F64(0.0));
    }

    #[test]
    fn test_ptype() {
        assert_eq!(PValue::U8(10).ptype(), PType::U8);
        assert_eq!(PValue::U16(10).ptype(), PType::U16);
        assert_eq!(PValue::U32(10).ptype(), PType::U32);
        assert_eq!(PValue::U64(10).ptype(), PType::U64);
        assert_eq!(PValue::I8(10).ptype(), PType::I8);
        assert_eq!(PValue::I16(10).ptype(), PType::I16);
        assert_eq!(PValue::I32(10).ptype(), PType::I32);
        assert_eq!(PValue::I64(10).ptype(), PType::I64);
        assert_eq!(PValue::F16(f16::from_f32(10.0)).ptype(), PType::F16);
        assert_eq!(PValue::F32(10.0).ptype(), PType::F32);
        assert_eq!(PValue::F64(10.0).ptype(), PType::F64);
    }

    #[test]
    fn test_reinterpret_cast_same_type() {
        let value = PValue::U32(42);
        assert_eq!(value.reinterpret_cast(PType::U32), value);
    }

    #[test]
    fn test_reinterpret_cast_u8_i8() {
        let value = PValue::U8(255);
        let casted = value.reinterpret_cast(PType::I8);
        assert_eq!(casted, PValue::I8(-1));
    }

    #[test]
    fn test_reinterpret_cast_u16_types() {
        let value = PValue::U16(12345);

        // U16 -> I16
        let as_i16 = value.reinterpret_cast(PType::I16);
        assert_eq!(as_i16, PValue::I16(12345));

        // U16 -> F16
        let as_f16 = value.reinterpret_cast(PType::F16);
        assert_eq!(as_f16, PValue::F16(f16::from_bits(12345)));
    }

    #[test]
    fn test_reinterpret_cast_u32_types() {
        let value = PValue::U32(0x3f800000); // 1.0 in float bits

        // U32 -> F32
        let as_f32 = value.reinterpret_cast(PType::F32);
        assert_eq!(as_f32, PValue::F32(1.0));

        // U32 -> I32
        let value2 = PValue::U32(0x80000000);
        let as_i32 = value2.reinterpret_cast(PType::I32);
        assert_eq!(as_i32, PValue::I32(i32::MIN));
    }

    #[test]
    fn test_reinterpret_cast_f32_to_u32() {
        let value = PValue::F32(1.0);
        let as_u32 = value.reinterpret_cast(PType::U32);
        assert_eq!(as_u32, PValue::U32(0x3f800000));
    }

    #[test]
    fn test_reinterpret_cast_f64_to_i64() {
        let value = PValue::F64(1.0);
        let as_i64 = value.reinterpret_cast(PType::I64);
        assert_eq!(as_i64, PValue::I64(0x3ff0000000000000_i64));
    }

    #[test]
    #[should_panic(expected = "Cannot reinterpret cast between types of different widths")]
    fn test_reinterpret_cast_different_widths() {
        let value = PValue::U8(42);
        let _ = value.reinterpret_cast(PType::U16);
    }

    #[test]
    fn test_as_primitive_conversions() {
        // Test as_u8
        assert_eq!(PValue::U8(42).as_u8(), Some(42));
        assert_eq!(PValue::I8(42).as_u8(), Some(42));
        assert_eq!(PValue::U16(255).as_u8(), Some(255));
        assert_eq!(PValue::U16(256).as_u8(), None); // Overflow

        // Test as_i32
        assert_eq!(PValue::I32(42).as_i32(), Some(42));
        assert_eq!(PValue::U32(42).as_i32(), Some(42));
        assert_eq!(PValue::I64(42).as_i32(), Some(42));
        assert_eq!(PValue::U64(u64::MAX).as_i32(), None); // Overflow

        // Test as_f64
        assert_eq!(PValue::F64(42.5).as_f64(), Some(42.5));
        assert_eq!(PValue::F32(42.5).as_f64(), Some(42.5f64));
        assert_eq!(PValue::I32(42).as_f64(), Some(42.0));
    }

    #[test]
    fn test_try_from_pvalue_integers() {
        // Test u8 conversion
        assert_eq!(u8::try_from(PValue::U8(42)).unwrap(), 42);
        assert_eq!(u8::try_from(PValue::I8(42)).unwrap(), 42);
        assert!(u8::try_from(PValue::I8(-1)).is_err());
        assert!(u8::try_from(PValue::U16(256)).is_err());

        // Test i32 conversion
        assert_eq!(i32::try_from(PValue::I32(42)).unwrap(), 42);
        assert_eq!(i32::try_from(PValue::I16(-100)).unwrap(), -100);
        assert!(i32::try_from(PValue::U64(u64::MAX)).is_err());

        // Float to int should fail
        assert!(i32::try_from(PValue::F32(42.5)).is_err());
    }

    #[test]
    fn test_try_from_pvalue_floats() {
        // Test f32 conversion
        assert_eq!(f32::try_from(PValue::F32(42.5)).unwrap(), 42.5);
        assert_eq!(f32::try_from(PValue::I32(42)).unwrap(), 42.0);
        assert_eq!(f32::try_from(PValue::U8(255)).unwrap(), 255.0);

        // Test f64 conversion
        assert_eq!(f64::try_from(PValue::F64(42.5)).unwrap(), 42.5);
        assert_eq!(f64::try_from(PValue::F32(42.5)).unwrap(), 42.5f64);
        assert_eq!(f64::try_from(PValue::I64(-100)).unwrap(), -100.0);
    }

    #[test]
    fn test_from_usize() {
        let value: PValue = 42usize.into();
        assert_eq!(value, PValue::U64(42));

        let max_value: PValue = usize::MAX.into();
        assert_eq!(max_value, PValue::U64(usize::MAX as u64));
    }

    #[test]
    fn test_equality_cross_types() {
        // Same numeric value, different types
        assert_eq!(PValue::U8(42), PValue::U16(42));
        assert_eq!(PValue::U8(42), PValue::U32(42));
        assert_eq!(PValue::U8(42), PValue::U64(42));
        assert_eq!(PValue::I8(42), PValue::I16(42));
        assert_eq!(PValue::I8(42), PValue::I32(42));
        assert_eq!(PValue::I8(42), PValue::I64(42));

        // Unsigned vs signed with same value (they compare equal even though different categories)
        assert_eq!(PValue::U8(42), PValue::I8(42));
        assert_eq!(PValue::U32(42), PValue::I32(42));

        // Float equality
        assert_eq!(PValue::F32(42.0), PValue::F32(42.0));
        assert_eq!(PValue::F64(42.0), PValue::F64(42.0));
        assert_ne!(PValue::F32(42.0), PValue::F64(42.0)); // Different types

        // Float vs int should not be equal
        assert_ne!(PValue::F32(42.0), PValue::I32(42));
    }

    #[test]
    fn test_partial_ord_cross_types() {
        // Unsigned comparisons
        assert_eq!(
            PValue::U8(10).partial_cmp(&PValue::U16(20)),
            Some(Ordering::Less)
        );
        assert_eq!(
            PValue::U32(30).partial_cmp(&PValue::U8(20)),
            Some(Ordering::Greater)
        );

        // Signed comparisons
        assert_eq!(
            PValue::I8(-10).partial_cmp(&PValue::I64(0)),
            Some(Ordering::Less)
        );
        assert_eq!(
            PValue::I32(10).partial_cmp(&PValue::I16(10)),
            Some(Ordering::Equal)
        );

        // Float comparisons (same type only)
        assert_eq!(
            PValue::F32(1.0).partial_cmp(&PValue::F32(2.0)),
            Some(Ordering::Less)
        );
        assert_eq!(
            PValue::F64(2.0).partial_cmp(&PValue::F64(1.0)),
            Some(Ordering::Greater)
        );

        // Cross-category comparisons - unsigned vs signed work, float vs int don't
        assert_eq!(
            PValue::U32(42).partial_cmp(&PValue::I32(42)),
            Some(Ordering::Equal)
        ); // Actually works
        assert_eq!(PValue::F32(42.0).partial_cmp(&PValue::I32(42)), None);
        assert_eq!(PValue::F32(42.0).partial_cmp(&PValue::F64(42.0)), None);
    }

    #[test]
    fn test_to_le_bytes() {
        assert_eq!(PValue::U8(0x12).to_le_bytes(), &[0x12]);
        assert_eq!(PValue::U16(0x1234).to_le_bytes(), &[0x34, 0x12]);
        assert_eq!(
            PValue::U32(0x12345678).to_le_bytes(),
            &[0x78, 0x56, 0x34, 0x12]
        );

        assert_eq!(PValue::I8(-1).to_le_bytes(), &[0xFF]);
        assert_eq!(PValue::I16(-1).to_le_bytes(), &[0xFF, 0xFF]);

        let f32_bytes = PValue::F32(1.0).to_le_bytes();
        assert_eq!(f32_bytes.len(), 4);

        let f64_bytes = PValue::F64(1.0).to_le_bytes();
        assert_eq!(f64_bytes.len(), 8);
    }

    #[test]
    fn test_f16_special_values() {
        // Test F16 NaN handling
        let nan = f16::NAN;
        let nan_value = PValue::F16(nan);
        assert!(nan_value.as_f16().unwrap().is_nan());

        // Test F16 infinity
        let inf = f16::INFINITY;
        let inf_value = PValue::F16(inf);
        assert!(inf_value.as_f16().unwrap().is_infinite());

        // Test F16 comparison with NaN
        assert_eq!(
            PValue::F16(nan).partial_cmp(&PValue::F16(nan)),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn test_coerce_pvalue() {
        // Test integer coercion
        assert_eq!(u32::coerce(PValue::U16(42)).unwrap(), 42u32);
        assert_eq!(i64::coerce(PValue::I32(-42)).unwrap(), -42i64);

        // Test float coercion from bits
        assert_eq!(f32::coerce(PValue::U32(0x3f800000)).unwrap(), 1.0f32);
        assert_eq!(
            f64::coerce(PValue::U64(0x3ff0000000000000)).unwrap(),
            1.0f64
        );
    }

    #[test]
    fn test_coerce_f16_beyond_u16_max() {
        // Test U32 to f16 coercion within valid range
        assert!(f16::coerce(PValue::U32(u16::MAX as u32)).is_ok());
        assert_eq!(
            f16::coerce(PValue::U32(0x3C00)).unwrap(),
            f16::from_bits(0x3C00) // 1.0 in f16
        );

        // Test U32 to f16 coercion beyond u16::MAX - should fail
        assert!(f16::coerce(PValue::U32((u16::MAX as u32) + 1)).is_err());
        assert!(f16::coerce(PValue::U32(u32::MAX)).is_err());

        // Test U64 to f16 coercion within valid range
        assert!(f16::coerce(PValue::U64(u16::MAX as u64)).is_ok());
        assert_eq!(
            f16::coerce(PValue::U64(0x3C00)).unwrap(),
            f16::from_bits(0x3C00) // 1.0 in f16
        );

        // Test U64 to f16 coercion beyond u16::MAX - should fail
        assert!(f16::coerce(PValue::U64((u16::MAX as u64) + 1)).is_err());
        assert!(f16::coerce(PValue::U64(u32::MAX as u64)).is_err());
        assert!(f16::coerce(PValue::U64(u64::MAX)).is_err());
    }

    #[test]
    fn test_coerce_f32_beyond_u32_max() {
        // Test U64 to f32 coercion within valid range
        assert!(f32::coerce(PValue::U64(u32::MAX as u64)).is_ok());
        assert_eq!(
            f32::coerce(PValue::U64(0x3f800000)).unwrap(),
            1.0f32 // 0x3f800000 is 1.0 in f32
        );

        // Test U64 to f32 coercion beyond u32::MAX - should fail
        assert!(f32::coerce(PValue::U64((u32::MAX as u64) + 1)).is_err());
        assert!(f32::coerce(PValue::U64(u64::MAX)).is_err());

        // Test smaller types still work
        assert!(f32::coerce(PValue::U8(255)).is_ok());
        assert!(f32::coerce(PValue::U16(u16::MAX)).is_ok());
        assert!(f32::coerce(PValue::U32(u32::MAX)).is_ok());
    }

    #[test]
    fn test_coerce_f64_all_unsigned() {
        // Test f64 can accept all unsigned integer values as bit patterns
        assert!(f64::coerce(PValue::U8(u8::MAX)).is_ok());
        assert!(f64::coerce(PValue::U16(u16::MAX)).is_ok());
        assert!(f64::coerce(PValue::U32(u32::MAX)).is_ok());
        assert!(f64::coerce(PValue::U64(u64::MAX)).is_ok());

        // Verify specific bit patterns
        assert_eq!(
            f64::coerce(PValue::U64(0x3ff0000000000000)).unwrap(),
            1.0f64 // 0x3ff0000000000000 is 1.0 in f64
        );
    }

    #[test]
    fn test_f16_nans_equal() {
        let nan1 = f16::from_le_bytes([154, 253]);
        assert!(nan1.is_nan());
        let nan3 = f16::from_f16(nan1).unwrap();
        assert_eq!(nan1.to_bits(), nan3.to_bits(),);
    }
}
