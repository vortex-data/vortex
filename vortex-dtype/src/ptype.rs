//! Physical type definitions and behavior.

use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::panic::RefUnwindSafe;

use num_traits::{FromPrimitive, Num, NumCast};
use vortex_error::{vortex_err, VortexError, VortexResult};

use crate::half::f16;
use crate::nullability::Nullability::NonNullable;
use crate::DType;
use crate::DType::*;

/// Physical type enum, represents the in-memory physical layout but might represent a different logical type.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
pub enum PType {
    /// An 8-bit unsigned integer
    U8,
    /// A 16-bit unsigned integer
    U16,
    /// A 32-bit unsigned integer
    U32,
    /// A 64-bit unsigned integer
    U64,
    /// An 8-bit signed integer
    I8,
    /// A 16-bit signed integer
    I16,
    /// A 32-bit signed integer
    I32,
    /// A 64-bit signed integer
    I64,
    /// A 16-bit floating point number
    F16,
    /// A 32-bit floating point number
    F32,
    /// A 64-bit floating point number
    F64,
}

/// A trait for native Rust types that correspond 1:1 to a PType
pub trait NativePType:
    Send
    + Sync
    + Clone
    + Copy
    + Debug
    + Display
    + PartialEq
    + PartialOrd
    + Default
    + RefUnwindSafe
    + Num
    + NumCast
    + FromPrimitive
    + ToBytes
    + TryFromBytes
{
    /// The PType that corresponds to this native type
    const PTYPE: PType;

    /// Whether this instance (`self`) is NaN
    /// For integer types, this is always `false`
    fn is_nan(self) -> bool;

    /// Compare another instance of this type to `self`
    fn compare(self, other: Self) -> Ordering;

    /// Whether another instance of this type (`other`) is bitwise equal to `self`
    fn is_eq(self, other: Self) -> bool;
}

macro_rules! native_ptype {
    ($T:ty, $ptype:tt) => {
        impl NativePType for $T {
            const PTYPE: PType = PType::$ptype;

            fn is_nan(self) -> bool {
                false
            }

            fn compare(self, other: Self) -> Ordering {
                self.cmp(&other)
            }

            fn is_eq(self, other: Self) -> bool {
                self == other
            }
        }
    };
}

macro_rules! native_float_ptype {
    ($T:ty, $ptype:tt) => {
        impl NativePType for $T {
            const PTYPE: PType = PType::$ptype;

            fn is_nan(self) -> bool {
                <$T>::is_nan(self)
            }

            fn compare(self, other: Self) -> Ordering {
                self.total_cmp(&other)
            }

            fn is_eq(self, other: Self) -> bool {
                self.to_bits() == other.to_bits()
            }
        }
    };
}

native_ptype!(u8, U8);
native_ptype!(u16, U16);
native_ptype!(u32, U32);
native_ptype!(u64, U64);
native_ptype!(i8, I8);
native_ptype!(i16, I16);
native_ptype!(i32, I32);
native_ptype!(i64, I64);
native_float_ptype!(f16, F16);
native_float_ptype!(f32, F32);
native_float_ptype!(f64, F64);

/// Macro to match over each PType, binding the corresponding native type (from `NativePType`)
#[macro_export]
macro_rules! match_each_native_ptype {
    ($self:expr, | $_:tt $enc:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $enc:ident ) => ( $($body)* )}
        use $crate::PType;
        use $crate::half::f16;
        match $self {
            PType::I8 => __with__! { i8 },
            PType::I16 => __with__! { i16 },
            PType::I32 => __with__! { i32 },
            PType::I64 => __with__! { i64 },
            PType::U8 => __with__! { u8 },
            PType::U16 => __with__! { u16 },
            PType::U32 => __with__! { u32 },
            PType::U64 => __with__! { u64 },
            PType::F16 => __with__! { f16 },
            PType::F32 => __with__! { f32 },
            PType::F64 => __with__! { f64 },
        }
    })
}

/// Macro to match over each integer PType, binding the corresponding native type (from `NativePType`)
#[macro_export]
macro_rules! match_each_integer_ptype {
    ($self:expr, | $_:tt $enc:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $enc:ident ) => ( $($body)* )}
        use $crate::PType;
        match $self {
            PType::I8 => __with__! { i8 },
            PType::I16 => __with__! { i16 },
            PType::I32 => __with__! { i32 },
            PType::I64 => __with__! { i64 },
            PType::U8 => __with__! { u8 },
            PType::U16 => __with__! { u16 },
            PType::U32 => __with__! { u32 },
            PType::U64 => __with__! { u64 },
            PType::F16 =>  panic!("Unsupported ptype f16"),
            PType::F32 =>  panic!("Unsupported ptype f32"),
            PType::F64 =>  panic!("Unsupported ptype f64"),
        }
    })
}

/// Macro to match over each unsigned integer type, binding the corresponding native type (from `NativePType`)
#[macro_export]
macro_rules! match_each_unsigned_integer_ptype {
    ($self:expr, | $_:tt $enc:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $enc:ident ) => ( $($body)* )}
        use $crate::PType;
        match $self {
            PType::U8 => __with__! { u8 },
            PType::U16 => __with__! { u16 },
            PType::U32 => __with__! { u32 },
            PType::U64 => __with__! { u64 },
            _ => panic!("Unsupported ptype {}", $self),
        }
    })
}

/// Macro to match over each floating point type, binding the corresponding native type (from `NativePType`)
#[macro_export]
macro_rules! match_each_float_ptype {
    ($self:expr, | $_:tt $enc:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $enc:ident ) => ( $($body)* )}
        use $crate::PType;
        use vortex_dtype::half::f16;
        match $self {
            PType::F16 => __with__! { f16 },
            PType::F32 => __with__! { f32 },
            PType::F64 => __with__! { f64 },
            _ => panic!("Unsupported ptype {}", $self),
        }
    })
}

impl PType {
    /// Returns `true` iff this PType is an unsigned integer type
    pub const fn is_unsigned_int(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    /// Returns `true` iff this PType is a signed integer type
    pub const fn is_signed_int(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// Returns `true` iff this PType is an integer type
    /// Equivalent to `self.is_unsigned_int() || self.is_signed_int()`
    pub const fn is_int(self) -> bool {
        self.is_unsigned_int() || self.is_signed_int()
    }

    /// Returns `true` iff this PType is a floating point type
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    /// Returns the number of bytes in this PType
    pub const fn byte_width(&self) -> usize {
        match_each_native_ptype!(self, |$T| std::mem::size_of::<$T>())
    }

    /// Returns the number of bits in this PType
    pub const fn bit_width(&self) -> usize {
        self.byte_width() * 8
    }

    /// Returns the maximum value of this PType if it is an integer type
    /// For floating point types, this will panic
    pub const fn max_value(&self) -> u64 {
        match_each_integer_ptype!(self, |$T| $T::MAX as u64)
    }

    /// Returns the PType that corresponds to the signed version of this PType
    pub const fn to_signed(self) -> Self {
        match self {
            Self::U8 => Self::I8,
            Self::U16 => Self::I16,
            Self::U32 => Self::I32,
            Self::U64 => Self::I64,
            _ => self,
        }
    }

    /// Returns the PType that corresponds to the unsigned version of this PType
    /// For floating point types, this will simply return `self`
    pub const fn to_unsigned(self) -> Self {
        match self {
            Self::I8 => Self::U8,
            Self::I16 => Self::U16,
            Self::I32 => Self::U32,
            Self::I64 => Self::U64,
            _ => self,
        }
    }
}

impl Display for PType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U8 => write!(f, "u8"),
            Self::U16 => write!(f, "u16"),
            Self::U32 => write!(f, "u32"),
            Self::U64 => write!(f, "u64"),
            Self::I8 => write!(f, "i8"),
            Self::I16 => write!(f, "i16"),
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::F16 => write!(f, "f16"),
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
        }
    }
}

impl TryFrom<&DType> for PType {
    type Error = VortexError;

    fn try_from(value: &DType) -> VortexResult<Self> {
        match value {
            Primitive(p, _) => Ok(*p),
            _ => Err(vortex_err!("Cannot convert DType {} into PType", value)),
        }
    }
}

impl From<PType> for &DType {
    fn from(item: PType) -> Self {
        // We expand this match statement so that we can return a static reference.
        match item {
            PType::I8 => &Primitive(PType::I8, NonNullable),
            PType::I16 => &Primitive(PType::I16, NonNullable),
            PType::I32 => &Primitive(PType::I32, NonNullable),
            PType::I64 => &Primitive(PType::I64, NonNullable),
            PType::U8 => &Primitive(PType::U8, NonNullable),
            PType::U16 => &Primitive(PType::U16, NonNullable),
            PType::U32 => &Primitive(PType::U32, NonNullable),
            PType::U64 => &Primitive(PType::U64, NonNullable),
            PType::F16 => &Primitive(PType::F16, NonNullable),
            PType::F32 => &Primitive(PType::F32, NonNullable),
            PType::F64 => &Primitive(PType::F64, NonNullable),
        }
    }
}

impl From<PType> for DType {
    fn from(item: PType) -> Self {
        Primitive(item, NonNullable)
    }
}

pub trait ToBytes: Sized {
    fn to_le_bytes(&self) -> &[u8];
}

pub trait TryFromBytes: Sized {
    fn try_from_le_bytes(bytes: &[u8]) -> VortexResult<Self>;
}

macro_rules! try_from_bytes {
    ($T:ty) => {
        impl ToBytes for $T {
            #[inline]
            #[allow(clippy::size_of_in_element_count)]
            fn to_le_bytes(&self) -> &[u8] {
                // NOTE(ngates): this assumes the platform is little-endian. Currently enforced
                //  with a flag cfg(target_endian = "little")
                let raw_ptr = self as *const $T as *const u8;
                unsafe { std::slice::from_raw_parts(raw_ptr, std::mem::size_of::<$T>()) }
            }
        }

        impl TryFromBytes for $T {
            fn try_from_le_bytes(bytes: &[u8]) -> VortexResult<Self> {
                Ok(<$T>::from_le_bytes(bytes.try_into()?))
            }
        }
    };
}

try_from_bytes!(u8);
try_from_bytes!(u16);
try_from_bytes!(u32);
try_from_bytes!(u64);
try_from_bytes!(i8);
try_from_bytes!(i16);
try_from_bytes!(i32);
try_from_bytes!(i64);
try_from_bytes!(f16);
try_from_bytes!(f32);
try_from_bytes!(f64);
