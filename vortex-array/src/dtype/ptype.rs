// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Physical type definitions and behavior.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::ops::AddAssign;
use std::panic::RefUnwindSafe;

use num_traits::AsPrimitive;
use num_traits::Bounded;
use num_traits::Num;
use num_traits::NumCast;
use num_traits::PrimInt;
use num_traits::ToPrimitive;
use num_traits::Unsigned;
use num_traits::bounds::UpperBounded;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::FromPrimitiveOrF16;
use crate::dtype::half::f16;
use crate::dtype::nullability::Nullability::NonNullable;

/// Physical type enum, represents the in-memory physical layout but might represent a different logical type.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Hash, prost::Enumeration)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "lowercase"))]
#[repr(u8)]
pub enum PType {
    /// An 8-bit unsigned integer
    U8 = 0,
    /// A 16-bit unsigned integer
    U16 = 1,
    /// A 32-bit unsigned integer
    U32 = 2,
    /// A 64-bit unsigned integer
    U64 = 3,
    /// An 8-bit signed integer
    I8 = 4,
    /// A 16-bit signed integer
    I16 = 5,
    /// A 32-bit signed integer
    I32 = 6,
    /// A 64-bit signed integer
    I64 = 7,
    /// A 16-bit floating point number
    F16 = 8,
    /// A 32-bit floating point number
    F32 = 9,
    /// A 64-bit floating point number
    F64 = 10,
}

/// Trait for integer primitive types that can be used as indices, offsets, or codes.
///
/// Includes all signed and unsigned integer types (u8, u16, u32, u64, i8, i16, i32, i64).
///
/// You can use the `match_each_integer_ptype` macro to help with writing "generic" code over
/// dynamically typed code.
pub trait IntegerPType:
    NativePType + PrimInt + ToPrimitive + Bounded + AddAssign + AsPrimitive<usize>
{
    /// Returns the maximum offset value that can be represented by this type.
    fn max_value_as_u64() -> u64 {
        Self::PTYPE.max_value_as_u64()
    }
}

/// Implements [`IntegerPType`] for all possible `T` that have the correct bounds.
impl<T> IntegerPType for T where
    T: NativePType + PrimInt + ToPrimitive + Bounded + AddAssign + AsPrimitive<usize>
{
}

/// Trait for unsigned integer primitive types used where non-negative values are required.
///
/// Includes only unsigned integer types (u8, u16, u32, u64).
///
/// You can use the `match_each_unsigned_integer_ptype` macro to help with writing "generic" code
/// over dynamically typed code.
pub trait UnsignedPType: IntegerPType + Unsigned {}

/// Implements [`UnsignedPType`] for all possible `T` that have the correct bounds.
impl<T> UnsignedPType for T where T: IntegerPType + Unsigned {}

/// A trait for native Rust types that correspond 1:1 to a PType.
///
/// You can use the `match_each_native_ptype` macro to help with writing "generic" code over
/// dynamically typed code.
pub trait NativePType:
    Send
    + Sync
    + Clone
    + Copy
    + Debug
    + Display
    + Default
    + RefUnwindSafe
    + Num
    + NumCast
    + FromPrimitiveOrF16
    + ToBytes
    + TryFromBytes
    + private::Sealed
    + 'static
{
    /// The PType that corresponds to this native type
    const PTYPE: PType;

    /// Whether this instance (`self`) is NaN
    /// For integer types, this is always `false`
    fn is_nan(self) -> bool;

    /// Whether this instance (`self`) is Infinite
    /// For integer types, this is always `false`
    fn is_infinite(self) -> bool;

    /// Compare another instance of this type to `self`, providing a total ordering
    fn total_compare(self, other: Self) -> Ordering;

    /// Test whether self is less than or equal to the other
    #[inline]
    fn is_le(self, other: Self) -> bool {
        self.total_compare(other).is_le()
    }

    /// Test whether self is less than the other
    #[inline]
    fn is_lt(self, other: Self) -> bool {
        self.total_compare(other).is_lt()
    }

    /// Test whether self is greater than or equal to the other
    #[inline]
    fn is_ge(self, other: Self) -> bool {
        self.total_compare(other).is_ge()
    }

    /// Test whether self is greater than the other
    #[inline]
    fn is_gt(self, other: Self) -> bool {
        self.total_compare(other).is_gt()
    }

    /// Whether another instance of this type (`other`) is bitwise equal to `self`
    fn is_eq(self, other: Self) -> bool;

    /// Downcast the provided object to a type-specific instance.
    fn downcast<V: PTypeDowncast>(visitor: V) -> V::Output<Self>;

    /// Upcast a type-specific instance to a generic instance.
    fn upcast<V: PTypeUpcast>(input: V::Input<Self>) -> V;
}

mod private {
    use half::f16;

    /// A private trait to prevent external implementations of `NativePType`.
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A visitor trait for converting a `NativePType` to another parameterized type.
#[expect(missing_docs, reason = "method names are self-documenting")]
pub trait PTypeDowncast {
    type Output<T: NativePType>;

    fn into_u8(self) -> Self::Output<u8>;
    fn into_u16(self) -> Self::Output<u16>;
    fn into_u32(self) -> Self::Output<u32>;
    fn into_u64(self) -> Self::Output<u64>;
    fn into_i8(self) -> Self::Output<i8>;
    fn into_i16(self) -> Self::Output<i16>;
    fn into_i32(self) -> Self::Output<i32>;
    fn into_i64(self) -> Self::Output<i64>;
    fn into_f16(self) -> Self::Output<f16>;
    fn into_f32(self) -> Self::Output<f32>;
    fn into_f64(self) -> Self::Output<f64>;
}

/// Extension trait to provide generic downcasting for [`PTypeDowncast`].
pub trait PTypeDowncastExt: PTypeDowncast {
    /// Downcast the object to a specific primitive type.
    fn downcast<T: NativePType>(self) -> Self::Output<T>
    where
        Self: Sized,
    {
        T::downcast(self)
    }
}

impl<T: PTypeDowncast> PTypeDowncastExt for T {}

macro_rules! impl_ptype_downcast {
    ($T:ty) => {
        #[inline]
        fn downcast<V: PTypeDowncast>(visitor: V) -> V::Output<Self> {
            paste::paste! { visitor.[<into_ $T>]() }
        }

        #[inline]
        fn upcast<V: PTypeUpcast>(input: V::Input<Self>) -> V {
            paste::paste! { V::[<from_ $T>](input) }
        }
    };
}

/// A visitor trait for converting a generic `NativePType` into a non-parameterized type.
#[expect(missing_docs, reason = "method names are self-documenting")]
pub trait PTypeUpcast {
    type Input<T: NativePType>;

    fn from_u8(input: Self::Input<u8>) -> Self;
    fn from_u16(input: Self::Input<u16>) -> Self;
    fn from_u32(input: Self::Input<u32>) -> Self;
    fn from_u64(input: Self::Input<u64>) -> Self;
    fn from_i8(input: Self::Input<i8>) -> Self;
    fn from_i16(input: Self::Input<i16>) -> Self;
    fn from_i32(input: Self::Input<i32>) -> Self;
    fn from_i64(input: Self::Input<i64>) -> Self;
    fn from_f16(input: Self::Input<f16>) -> Self;
    fn from_f32(input: Self::Input<f32>) -> Self;
    fn from_f64(input: Self::Input<f64>) -> Self;
}

macro_rules! native_ptype {
    ($T:ty, $ptype:tt) => {
        impl crate::dtype::NativeDType for $T {
            fn dtype() -> DType {
                DType::Primitive(PType::$ptype, crate::dtype::Nullability::NonNullable)
            }
        }

        impl NativePType for $T {
            const PTYPE: PType = PType::$ptype;

            #[inline]
            fn is_nan(self) -> bool {
                false
            }

            #[inline]
            fn is_infinite(self) -> bool {
                false
            }

            #[inline]
            fn total_compare(self, other: Self) -> Ordering {
                self.cmp(&other)
            }

            #[inline]
            fn is_eq(self, other: Self) -> bool {
                self == other
            }

            impl_ptype_downcast!($T);
        }
    };
}

macro_rules! native_float_ptype {
    ($T:ty, $ptype:tt) => {
        impl crate::dtype::NativeDType for $T {
            fn dtype() -> DType {
                DType::Primitive(PType::$ptype, crate::dtype::Nullability::NonNullable)
            }
        }

        impl NativePType for $T {
            const PTYPE: PType = PType::$ptype;

            #[inline]
            fn is_nan(self) -> bool {
                <$T>::is_nan(self)
            }

            #[inline]
            fn is_infinite(self) -> bool {
                <$T>::is_infinite(self)
            }

            #[inline]
            fn total_compare(self, other: Self) -> Ordering {
                self.total_cmp(&other)
            }

            #[inline]
            fn is_eq(self, other: Self) -> bool {
                self.to_bits() == other.to_bits()
            }

            impl_ptype_downcast!($T);
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
    (
        $self:expr,integral: |
        $integral_enc:ident |
        $intbody:block,floating: |
        $floating_point_enc:ident |
        $floatbody:block
    ) => {{
        use $crate::dtype::PType;
        use $crate::dtype::half::f16;
        match $self {
            PType::I8 => {
                type $integral_enc = i8;
                $intbody
            }
            PType::I16 => {
                type $integral_enc = i16;
                $intbody
            }
            PType::I32 => {
                type $integral_enc = i32;
                $intbody
            }
            PType::I64 => {
                type $integral_enc = i64;
                $intbody
            }
            PType::U8 => {
                type $integral_enc = u8;
                $intbody
            }
            PType::U16 => {
                type $integral_enc = u16;
                $intbody
            }
            PType::U32 => {
                type $integral_enc = u32;
                $intbody
            }
            PType::U64 => {
                type $integral_enc = u64;
                $intbody
            }
            PType::F16 => {
                type $floating_point_enc = f16;
                $floatbody
            }
            PType::F32 => {
                type $floating_point_enc = f32;
                $floatbody
            }
            PType::F64 => {
                type $floating_point_enc = f64;
                $floatbody
            }
        }
    }};
    (
        $self:expr,unsigned: |
        $unsigned_enc:ident |
        $unsigned_body:block,signed: |
        $signed_enc:ident |
        $signed_body:block,floating: |
        $floating_point_enc:ident |
        $floating_point_body:block
    ) => {{
        use $crate::dtype::PType;
        use $crate::dtype::half::f16;
        match $self {
            PType::U8 => {
                type $unsigned_enc = u8;
                $unsigned_body
            }
            PType::U16 => {
                type $unsigned_enc = u16;
                $unsigned_body
            }
            PType::U32 => {
                type $unsigned_enc = u32;
                $unsigned_body
            }
            PType::U64 => {
                type $unsigned_enc = u64;
                $unsigned_body
            }
            PType::I8 => {
                type $signed_enc = i8;
                $signed_body
            }
            PType::I16 => {
                type $signed_enc = i16;
                $signed_body
            }
            PType::I32 => {
                type $signed_enc = i32;
                $signed_body
            }
            PType::I64 => {
                type $signed_enc = i64;
                $signed_body
            }
            PType::F16 => {
                type $floating_point_enc = f16;
                $floating_point_body
            }
            PType::F32 => {
                type $floating_point_enc = f32;
                $floating_point_body
            }
            PType::F64 => {
                type $floating_point_enc = f64;
                $floating_point_body
            }
        }
    }};
    ($self:expr, | $tname:ident | $body:block) => {{
        use $crate::dtype::PType;
        use $crate::dtype::half::f16;
        match $self {
            PType::I8 => {
                type $tname = i8;
                $body
            }
            PType::I16 => {
                type $tname = i16;
                $body
            }
            PType::I32 => {
                type $tname = i32;
                $body
            }
            PType::I64 => {
                type $tname = i64;
                $body
            }
            PType::U8 => {
                type $tname = u8;
                $body
            }
            PType::U16 => {
                type $tname = u16;
                $body
            }
            PType::U32 => {
                type $tname = u32;
                $body
            }
            PType::U64 => {
                type $tname = u64;
                $body
            }
            PType::F16 => {
                type $tname = f16;
                $body
            }
            PType::F32 => {
                type $tname = f32;
                $body
            }
            PType::F64 => {
                type $tname = f64;
                $body
            }
        }
    }};
}

/// Macro to match over each integer PType, binding the corresponding native type (from `NativePType`)
#[macro_export]
macro_rules! match_each_integer_ptype {
    ($self:expr, | $enc:ident | $body:block) => {{
        use $crate::dtype::PType;
        match $self {
            PType::I8 => {
                type $enc = i8;
                $body
            }
            PType::I16 => {
                type $enc = i16;
                $body
            }
            PType::I32 => {
                type $enc = i32;
                $body
            }
            PType::I64 => {
                type $enc = i64;
                $body
            }
            PType::U8 => {
                type $enc = u8;
                $body
            }
            PType::U16 => {
                type $enc = u16;
                $body
            }
            PType::U32 => {
                type $enc = u32;
                $body
            }
            PType::U64 => {
                type $enc = u64;
                $body
            }
            other => panic!("Unsupported ptype {other}"),
        }
    }};
}

/// Macro to match over each unsigned integer type, binding the corresponding native type (from `NativePType`)
#[macro_export]
macro_rules! match_each_unsigned_integer_ptype {
    ($self:expr, | $enc:ident | $body:block) => {{
        use $crate::dtype::PType;
        match $self {
            PType::U8 => {
                type $enc = u8;
                $body
            }
            PType::U16 => {
                type $enc = u16;
                $body
            }
            PType::U32 => {
                type $enc = u32;
                $body
            }
            PType::U64 => {
                type $enc = u64;
                $body
            }
            other => panic!("Unsupported ptype {other}"),
        }
    }};
}

/// Macro to match over each signed integer type, binding the corresponding native type (from `NativePType`)
#[macro_export]
macro_rules! match_each_signed_integer_ptype {
    ($self:expr, | $enc:ident | $body:block) => {{
        use $crate::dtype::PType;
        match $self {
            PType::I8 => {
                type $enc = i8;
                $body
            }
            PType::I16 => {
                type $enc = i16;
                $body
            }
            PType::I32 => {
                type $enc = i32;
                $body
            }
            PType::I64 => {
                type $enc = i64;
                $body
            }
            other => panic!("Unsupported ptype {other}"),
        }
    }};
}

/// Macro to match over each floating point type, binding the corresponding native type (from `NativePType`)
#[macro_export]
macro_rules! match_each_float_ptype {
    ($self:expr, | $enc:ident | $body:block) => {{
        use $crate::dtype::PType;
        use $crate::dtype::half::f16;
        match $self {
            PType::F16 => {
                type $enc = f16;
                $body
            }
            PType::F32 => {
                type $enc = f32;
                $body
            }
            PType::F64 => {
                type $enc = f64;
                $body
            }
            other => panic!("Unsupported ptype {other}"),
        }
    }};
}

/// Macro to match over each SIMD capable `PType`, binding the corresponding native type (from `NativePType`)
///
/// Note: The match will panic in case of `PType::F16`.
#[macro_export]
macro_rules! match_each_native_simd_ptype {
    ($self:expr, | $enc:ident | $body:block) => {{
        use $crate::dtype::PType;
        match $self {
            PType::I8 => {
                type $enc = i8;
                $body
            }
            PType::I16 => {
                type $enc = i16;
                $body
            }
            PType::I32 => {
                type $enc = i32;
                $body
            }
            PType::I64 => {
                type $enc = i64;
                $body
            }
            PType::U8 => {
                type $enc = u8;
                $body
            }
            PType::U16 => {
                type $enc = u16;
                $body
            }
            PType::U32 => {
                type $enc = u32;
                $body
            }
            PType::U64 => {
                type $enc = u64;
                $body
            }
            PType::F16 => panic!("f16 does not implement simd::SimdElement"),
            PType::F32 => {
                type $enc = f32;
                $body
            }
            PType::F64 => {
                type $enc = f64;
                $body
            }
        }
    }};
}

/// Macro to match the smallest offset type for a given value
#[macro_export]
macro_rules! match_smallest_offset_type {
    ($n_elements:expr, | $offset_type:ident | $body:block) => {{
        let n_elements = $n_elements;
        if n_elements <= u8::MAX as usize {
            type $offset_type = u8;
            $body
        } else if n_elements <= u16::MAX as usize {
            type $offset_type = u16;
            $body
        } else if n_elements <= u32::MAX as usize {
            type $offset_type = u32;
            $body
        } else {
            assert!(u64::try_from(n_elements).is_ok());
            type $offset_type = u64;
            $body
        }
    }};
}

impl PType {
    /// Returns `true` iff this PType is an unsigned integer type
    #[inline]
    pub const fn is_unsigned_int(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    /// Returns `true` iff this PType is a signed integer type
    #[inline]
    pub const fn is_signed_int(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    /// Returns `true` iff this PType is an integer type
    /// Equivalent to `self.is_unsigned_int() || self.is_signed_int()`
    #[inline]
    pub const fn is_int(self) -> bool {
        self.is_unsigned_int() || self.is_signed_int()
    }

    /// Returns `true` iff this PType is a floating point type
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64)
    }

    /// Returns the number of bytes in this PType
    #[inline]
    pub const fn byte_width(&self) -> usize {
        match_each_native_ptype!(self, |T| { size_of::<T>() })
    }

    /// Returns the number of bits in this PType
    #[inline]
    pub const fn bit_width(&self) -> usize {
        self.byte_width() * 8
    }

    /// Returns the maximum value of this PType if it is an integer type
    /// Returns `u64::MAX` if the value is too large to fit in a `u64`
    #[inline]
    pub fn max_value_as_u64(&self) -> u64 {
        match_each_native_ptype!(self, |T| {
            <T as UpperBounded>::max_value()
                .to_u64()
                .unwrap_or(u64::MAX)
        })
    }

    /// Returns the PType that corresponds to the signed version of this PType
    #[inline]
    pub const fn to_signed(self) -> Self {
        match self {
            Self::U8 => Self::I8,
            Self::U16 => Self::I16,
            Self::U32 => Self::I32,
            Self::U64 => Self::I64,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::F16 | Self::F32 | Self::F64 => {
                self
            }
        }
    }

    /// Returns the PType that corresponds to the unsigned version of this PType
    /// For floating point types, this will simply return `self`
    #[inline]
    pub const fn to_unsigned(self) -> Self {
        match self {
            Self::I8 => Self::U8,
            Self::I16 => Self::U16,
            Self::I32 => Self::U32,
            Self::I64 => Self::U64,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::F16 | Self::F32 | Self::F64 => {
                self
            }
        }
    }

    /// Returns the minimum unsigned integer [`PType`] that can represent the given value.
    #[inline]
    pub const fn min_unsigned_ptype_for_value(value: u64) -> Self {
        if value <= u8::MAX as u64 {
            Self::U8
        } else if value <= u16::MAX as u64 {
            Self::U16
        } else if value <= u32::MAX as u64 {
            Self::U32
        } else {
            Self::U64
        }
    }

    /// Returns the minimum signed integer [`PType`] that can represent the given value.
    #[inline]
    pub const fn min_signed_ptype_for_value(value: i64) -> Self {
        if value >= i8::MIN as i64 && value <= i8::MAX as i64 {
            Self::I8
        } else if value >= i16::MIN as i64 && value <= i16::MAX as i64 {
            Self::I16
        } else if value >= i32::MIN as i64 && value <= i32::MAX as i64 {
            Self::I32
        } else {
            Self::I64
        }
    }

    /// Returns the wider of two unsigned integer [`PType`]s based on byte width.
    #[inline]
    pub const fn max_unsigned_ptype(self, other: Self) -> Self {
        debug_assert!(self.is_unsigned_int() && other.is_unsigned_int());
        if self.byte_width() >= other.byte_width() {
            self
        } else {
            other
        }
    }

    /// Returns the wider of two signed integer [`PType`]s based on byte width.
    #[inline]
    pub const fn max_signed_ptype(self, other: Self) -> Self {
        debug_assert!(self.is_signed_int() && other.is_signed_int());
        if self.byte_width() >= other.byte_width() {
            self
        } else {
            other
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

    #[inline]
    fn try_from(value: &DType) -> VortexResult<Self> {
        if let DType::Primitive(p, _) = value {
            Ok(*p)
        } else {
            Err(vortex_err!("Cannot convert DType {} into PType", value))
        }
    }
}

impl From<PType> for &DType {
    fn from(item: PType) -> Self {
        // We expand this match statement so that we can return a static reference.
        match item {
            PType::I8 => &DType::Primitive(PType::I8, NonNullable),
            PType::I16 => &DType::Primitive(PType::I16, NonNullable),
            PType::I32 => &DType::Primitive(PType::I32, NonNullable),
            PType::I64 => &DType::Primitive(PType::I64, NonNullable),
            PType::U8 => &DType::Primitive(PType::U8, NonNullable),
            PType::U16 => &DType::Primitive(PType::U16, NonNullable),
            PType::U32 => &DType::Primitive(PType::U32, NonNullable),
            PType::U64 => &DType::Primitive(PType::U64, NonNullable),
            PType::F16 => &DType::Primitive(PType::F16, NonNullable),
            PType::F32 => &DType::Primitive(PType::F32, NonNullable),
            PType::F64 => &DType::Primitive(PType::F64, NonNullable),
        }
    }
}

impl From<PType> for DType {
    fn from(item: PType) -> Self {
        DType::Primitive(item, NonNullable)
    }
}

/// A trait for types that can be converted to a little-endian byte slice
pub trait ToBytes: Sized {
    /// Returns a slice of this type's bytes in little-endian order
    fn to_le_bytes(&self) -> &[u8];
}

/// A trait for types that can be converted from a little-endian byte slice
pub trait TryFromBytes: Sized {
    /// Attempts to convert a slice of bytes in little-endian order to this type
    fn try_from_le_bytes(bytes: &[u8]) -> VortexResult<Self>;
}

macro_rules! try_from_bytes {
    ($T:ty) => {
        impl ToBytes for $T {
            #[inline]
            fn to_le_bytes(&self) -> &[u8] {
                // NOTE(ngates): this assumes the platform is little-endian. Currently enforced
                //  with a flag cfg(target_endian = "little")
                let raw_ptr = (self as *const $T).cast::<u8>();
                unsafe { std::slice::from_raw_parts(raw_ptr, std::mem::size_of::<$T>()) }
            }
        }

        impl TryFromBytes for $T {
            fn try_from_le_bytes(bytes: &[u8]) -> VortexResult<Self> {
                Ok(<$T>::from_le_bytes(bytes.try_into().map_err(|_| {
                    vortex_err!("Failed to convert bytes into {}", stringify!($T))
                })?))
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

/// A trait that allows conversion from a PType to its physical representation (i.e., unsigned)
pub trait PhysicalPType: NativePType {
    /// The physical type that corresponds to this native type.
    type Physical: NativePType + Unsigned;
}

macro_rules! physical_ptype {
    ($T:ty, $U:ty) => {
        impl PhysicalPType for $T {
            type Physical = $U;
        }
    };
}

physical_ptype!(i8, u8);
physical_ptype!(i16, u16);
physical_ptype!(i32, u32);
physical_ptype!(i64, u64);
physical_ptype!(u8, u8);
physical_ptype!(u16, u16);
physical_ptype!(u32, u32);
physical_ptype!(u64, u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_from_bytes() {
        assert_eq!(u8::try_from_le_bytes(&[0x01]).unwrap(), 0x01);
        assert_eq!(u16::try_from_le_bytes(&[0x01, 0x02]).unwrap(), 0x0201);
        assert_eq!(
            u32::try_from_le_bytes(&[0x01, 0x02, 0x03, 0x04]).unwrap(),
            0x04030201
        );
        assert_eq!(
            u64::try_from_le_bytes(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]).unwrap(),
            0x0807060504030201
        );
    }

    #[test]
    fn to_bytes_rt() {
        assert_eq!(&0x01u8.to_le_bytes(), &[0x01]);
        assert_eq!(&0x0201u16.to_le_bytes(), &[0x01, 0x02]);

        assert_eq!(u8::try_from_le_bytes(&42_u8.to_le_bytes()).unwrap(), 42);
        assert_eq!(u16::try_from_le_bytes(&42_u16.to_le_bytes()).unwrap(), 42);
        assert_eq!(u32::try_from_le_bytes(&42_u32.to_le_bytes()).unwrap(), 42);
        assert_eq!(u64::try_from_le_bytes(&42_u64.to_le_bytes()).unwrap(), 42);
        assert_eq!(i8::try_from_le_bytes(&42_i8.to_le_bytes()).unwrap(), 42);
        assert_eq!(i16::try_from_le_bytes(&42_i16.to_le_bytes()).unwrap(), 42);
        assert_eq!(i32::try_from_le_bytes(&42_i32.to_le_bytes()).unwrap(), 42);
        assert_eq!(i64::try_from_le_bytes(&42_i64.to_le_bytes()).unwrap(), 42);
        assert_eq!(
            f16::try_from_le_bytes(&f16::from_f32(42.0).to_le_bytes()).unwrap(),
            f16::from_f32(42.0)
        );
        assert_eq!(
            f32::try_from_le_bytes(&42.0_f32.to_le_bytes()).unwrap(),
            42.0
        );
        assert_eq!(
            f64::try_from_le_bytes(&42.0_f64.to_le_bytes()).unwrap(),
            42.0
        );
    }

    #[test]
    fn max_value_u64() {
        assert_eq!(PType::U8.max_value_as_u64(), u8::MAX as u64);
        assert_eq!(PType::U16.max_value_as_u64(), u16::MAX as u64);
        assert_eq!(PType::U32.max_value_as_u64(), u32::MAX as u64);
        assert_eq!(PType::U64.max_value_as_u64(), u64::MAX);
        assert_eq!(PType::I8.max_value_as_u64(), i8::MAX as u64);
        assert_eq!(PType::I16.max_value_as_u64(), i16::MAX as u64);
        assert_eq!(PType::I32.max_value_as_u64(), i32::MAX as u64);
        assert_eq!(PType::I64.max_value_as_u64(), i64::MAX as u64);
        assert_eq!(PType::F16.max_value_as_u64(), 65504); // f16 is a weird type...
        assert_eq!(PType::F32.max_value_as_u64(), u64::MAX);
        assert_eq!(PType::F64.max_value_as_u64(), u64::MAX);
    }

    #[test]
    fn widths() {
        assert_eq!(PType::U8.byte_width(), 1);
        assert_eq!(PType::U16.byte_width(), 2);
        assert_eq!(PType::U32.byte_width(), 4);
        assert_eq!(PType::U64.byte_width(), 8);
        assert_eq!(PType::I8.byte_width(), 1);
        assert_eq!(PType::I16.byte_width(), 2);
        assert_eq!(PType::I32.byte_width(), 4);
        assert_eq!(PType::I64.byte_width(), 8);
        assert_eq!(PType::F16.byte_width(), 2);
        assert_eq!(PType::F32.byte_width(), 4);
        assert_eq!(PType::F64.byte_width(), 8);

        assert_eq!(PType::U8.bit_width(), 8);
        assert_eq!(PType::U16.bit_width(), 16);
        assert_eq!(PType::U32.bit_width(), 32);
        assert_eq!(PType::U64.bit_width(), 64);
        assert_eq!(PType::I8.bit_width(), 8);
        assert_eq!(PType::I16.bit_width(), 16);
        assert_eq!(PType::I32.bit_width(), 32);
        assert_eq!(PType::I64.bit_width(), 64);
        assert_eq!(PType::F16.bit_width(), 16);
        assert_eq!(PType::F32.bit_width(), 32);
        assert_eq!(PType::F64.bit_width(), 64);
    }

    #[test]
    fn native_ptype_nan_handling() {
        let a = f32::NAN;
        let b = f32::NAN;
        assert_ne!(a, b);
        assert!(<f32 as NativePType>::is_nan(a));
        assert!(<f32 as NativePType>::is_nan(b));
        assert!(<f32 as NativePType>::is_eq(a, b));
        assert!(<f32 as NativePType>::total_compare(a, b) == Ordering::Equal);
    }

    #[test]
    fn to_signed() {
        assert_eq!(PType::U8.to_signed(), PType::I8);
        assert_eq!(PType::U16.to_signed(), PType::I16);
        assert_eq!(PType::U32.to_signed(), PType::I32);
        assert_eq!(PType::U64.to_signed(), PType::I64);
        assert_eq!(PType::I8.to_signed(), PType::I8);
        assert_eq!(PType::I16.to_signed(), PType::I16);
        assert_eq!(PType::I32.to_signed(), PType::I32);
        assert_eq!(PType::I64.to_signed(), PType::I64);
        assert_eq!(PType::F16.to_signed(), PType::F16);
        assert_eq!(PType::F32.to_signed(), PType::F32);
        assert_eq!(PType::F64.to_signed(), PType::F64);
    }

    #[test]
    fn to_unsigned() {
        assert_eq!(PType::U8.to_unsigned(), PType::U8);
        assert_eq!(PType::U16.to_unsigned(), PType::U16);
        assert_eq!(PType::U32.to_unsigned(), PType::U32);
        assert_eq!(PType::U64.to_unsigned(), PType::U64);
        assert_eq!(PType::I8.to_unsigned(), PType::U8);
        assert_eq!(PType::I16.to_unsigned(), PType::U16);
        assert_eq!(PType::I32.to_unsigned(), PType::U32);
        assert_eq!(PType::I64.to_unsigned(), PType::U64);
        assert_eq!(PType::F16.to_unsigned(), PType::F16);
        assert_eq!(PType::F32.to_unsigned(), PType::F32);
        assert_eq!(PType::F64.to_unsigned(), PType::F64);
    }

    #[test]
    fn to_dtype() {
        assert_eq!(
            DType::from(PType::U8),
            DType::Primitive(PType::U8, NonNullable)
        );
        assert_eq!(
            DType::from(PType::U16),
            DType::Primitive(PType::U16, NonNullable)
        );
        assert_eq!(
            DType::from(PType::U32),
            DType::Primitive(PType::U32, NonNullable)
        );
        assert_eq!(
            DType::from(PType::U64),
            DType::Primitive(PType::U64, NonNullable)
        );
        assert_eq!(
            DType::from(PType::I8),
            DType::Primitive(PType::I8, NonNullable)
        );
        assert_eq!(
            DType::from(PType::I16),
            DType::Primitive(PType::I16, NonNullable)
        );
        assert_eq!(
            DType::from(PType::I32),
            DType::Primitive(PType::I32, NonNullable)
        );
        assert_eq!(
            DType::from(PType::I64),
            DType::Primitive(PType::I64, NonNullable)
        );
        assert_eq!(
            DType::from(PType::F16),
            DType::Primitive(PType::F16, NonNullable)
        );
        assert_eq!(
            DType::from(PType::F32),
            DType::Primitive(PType::F32, NonNullable)
        );
        assert_eq!(
            DType::from(PType::F64),
            DType::Primitive(PType::F64, NonNullable)
        );
    }
}
