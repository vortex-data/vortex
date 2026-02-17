// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PValue`] enum representing a typed primitive value.

use core::fmt::Display;
use std::cmp::Ordering;
use std::hash::Hash;
use std::hash::Hasher;

use num_traits::NumCast;
use num_traits::ToPrimitive;
use num_traits::Zero;
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

/// Generates an `as_<type>` accessor method on [`PValue`].
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
    /// Returns true if this decimal value is zero.
    pub fn is_zero(&self) -> bool {
        matches!(
            self,
            PValue::U8(0)
                | PValue::U16(0)
                | PValue::U32(0)
                | PValue::U64(0)
                | PValue::I8(0)
                | PValue::I16(0)
                | PValue::I32(0)
                | PValue::I64(0)
        ) || matches!(self, PValue::F16(f) if f.to_f32().is_some_and(|f| f.is_zero()))
            || matches!(self, PValue::F32(f) if f.is_zero())
            || matches!(self, PValue::F64(f) if f.is_zero())
    }

    /// Creates a zero value for the given primitive type.
    pub fn zero(ptype: &PType) -> PValue {
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
    /// # Errors
    /// Returns `VortexError` if the conversion is not supported or would overflow.
    #[inline]
    pub fn cast<T: NativePType>(&self) -> VortexResult<T> {
        let res = match *self {
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
        };
        let to = T::PTYPE;
        res.ok_or_else(|| vortex_err!("Cannot cast {self} to {to}"))
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
                if matches!(
                    value,
                    PValue::U8(_)
                        | PValue::U16(_)
                        | PValue::U32(_)
                        | PValue::U64(_)
                        | PValue::I8(_)
                        | PValue::I16(_)
                        | PValue::I32(_)
                        | PValue::I64(_)
                ) {
                    PValue::cast(&value)
                } else {
                    vortex_bail!("Cannot read primitive value {:?} as {}", value, PType::$PT)
                }
            }
        }
    };
}

/// Implements [`TryFrom<PValue>`] for a floating-point type.
macro_rules! float_pvalue {
    ($T:ty, $PT:tt) => {
        impl TryFrom<PValue> for $T {
            type Error = VortexError;

            fn try_from(value: PValue) -> Result<Self, Self::Error> {
                value.cast()
            }
        }
    };
}

impl TryFrom<PValue> for usize {
    type Error = VortexError;

    fn try_from(value: PValue) -> Result<Self, Self::Error> {
        value
            .cast::<u64>()?
            .to_usize()
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

/// Implements [`From<T>`] for [`PValue`].
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

/// Coercion trait for widening or reinterpreting a [`PValue`] into a concrete type.
pub(super) trait CoercePValue: Sized {
    /// Coerce value from a compatible bit representation using into given type.
    ///
    /// Integers can be widened from narrower type
    /// Floats stored as integers will be reinterpreted as bit representation of the float
    fn coerce(value: PValue) -> VortexResult<Self>;
}

/// Implements [`CoercePValue`] for an integer type.
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
