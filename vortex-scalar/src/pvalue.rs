use core::fmt::Display;
use std::cmp::Ordering;
use std::mem;

use num_traits::NumCast;
use paste::paste;
use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PType};
use vortex_error::{vortex_err, VortexError};

#[derive(Debug, Clone, Copy)]
pub enum PValue {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F16(f16),
    F32(f32),
    F64(f64),
}

impl PartialEq for PValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::U8(s), Self::U8(o)) => s.is_eq(*o),
            (Self::U16(s), Self::U16(o)) => s.is_eq(*o),
            (Self::U32(s), Self::U32(o)) => s.is_eq(*o),
            (Self::U64(s), Self::U64(o)) => s.is_eq(*o),
            (Self::I8(s), Self::I8(o)) => s.is_eq(*o),
            (Self::I16(s), Self::I16(o)) => s.is_eq(*o),
            (Self::I32(s), Self::I32(o)) => s.is_eq(*o),
            (Self::I64(s), Self::I64(o)) => s.is_eq(*o),
            (Self::F16(s), Self::F16(o)) => s.is_eq(*o),
            (Self::F32(s), Self::F32(o)) => s.is_eq(*o),
            (Self::F64(s), Self::F64(o)) => s.is_eq(*o),
            (..) => false,
        }
    }
}

impl PartialOrd for PValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::U8(s), Self::U8(o)) => Some(s.total_compare(*o)),
            (Self::U16(s), Self::U16(o)) => Some(s.total_compare(*o)),
            (Self::U32(s), Self::U32(o)) => Some(s.total_compare(*o)),
            (Self::U64(s), Self::U64(o)) => Some(s.total_compare(*o)),
            (Self::I8(s), Self::I8(o)) => Some(s.total_compare(*o)),
            (Self::I16(s), Self::I16(o)) => Some(s.total_compare(*o)),
            (Self::I32(s), Self::I32(o)) => Some(s.total_compare(*o)),
            (Self::I64(s), Self::I64(o)) => Some(s.total_compare(*o)),
            (Self::F16(s), Self::F16(o)) => Some(s.total_compare(*o)),
            (Self::F32(s), Self::F32(o)) => Some(s.total_compare(*o)),
            (Self::F64(s), Self::F64(o)) => Some(s.total_compare(*o)),
            (..) => None,
        }
    }
}

macro_rules! as_primitive {
    ($T:ty, $PT:tt) => {
        paste! {
            #[doc = "Access PValue as `" $T "`, returning `None` if conversion is unsuccessful"]
            pub fn [<as_ $T>](self) -> Option<$T> {
                match self {
                    PValue::U8(v) => <$T as NumCast>::from(v),
                    PValue::U16(v) => <$T as NumCast>::from(v),
                    PValue::U32(v) => <$T as NumCast>::from(v),
                    PValue::U64(v) => <$T as NumCast>::from(v),
                    PValue::I8(v) => <$T as NumCast>::from(v),
                    PValue::I16(v) => <$T as NumCast>::from(v),
                    PValue::I32(v) => <$T as NumCast>::from(v),
                    PValue::I64(v) => <$T as NumCast>::from(v),
                    PValue::F16(v) => <$T as NumCast>::from(v),
                    PValue::F32(v) => <$T as NumCast>::from(v),
                    PValue::F64(v) => <$T as NumCast>::from(v),
                }
            }
        }
    };
}

impl PValue {
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

    pub fn is_instance_of(&self, ptype: &PType) -> bool {
        &self.ptype() == ptype
    }

    #[inline]
    pub fn as_primitive<T: NativePType + TryFrom<PValue, Error = VortexError>>(
        &self,
    ) -> Result<T, VortexError> {
        T::try_from(*self)
    }

    #[allow(clippy::transmute_int_to_float, clippy::transmute_float_to_int)]
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
            PValue::U8(v) => unsafe { mem::transmute::<u8, i8>(*v) }.into(),
            PValue::U16(v) => match ptype {
                PType::I16 => unsafe { mem::transmute::<u16, i16>(*v) }.into(),
                PType::F16 => unsafe { mem::transmute::<u16, f16>(*v) }.into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::U32(v) => match ptype {
                PType::I32 => unsafe { mem::transmute::<u32, i32>(*v) }.into(),
                PType::F32 => unsafe { mem::transmute::<u32, f32>(*v) }.into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::U64(v) => match ptype {
                PType::I64 => unsafe { mem::transmute::<u64, i64>(*v) }.into(),
                PType::F64 => unsafe { mem::transmute::<u64, f64>(*v) }.into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::I8(v) => unsafe { mem::transmute::<i8, u8>(*v) }.into(),
            PValue::I16(v) => match ptype {
                PType::U16 => unsafe { mem::transmute::<i16, u16>(*v) }.into(),
                PType::F16 => unsafe { mem::transmute::<i16, f16>(*v) }.into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::I32(v) => match ptype {
                PType::U32 => unsafe { mem::transmute::<i32, u32>(*v) }.into(),
                PType::F32 => unsafe { mem::transmute::<i32, f32>(*v) }.into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::I64(v) => match ptype {
                PType::U64 => unsafe { mem::transmute::<i64, u64>(*v) }.into(),
                PType::F64 => unsafe { mem::transmute::<i64, f64>(*v) }.into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::F16(v) => match ptype {
                PType::U16 => unsafe { mem::transmute::<f16, u16>(*v) }.into(),
                PType::I16 => unsafe { mem::transmute::<f16, i16>(*v) }.into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::F32(v) => match ptype {
                PType::U32 => unsafe { mem::transmute::<f32, u32>(*v) }.into(),
                PType::I32 => unsafe { mem::transmute::<f32, i32>(*v) }.into(),
                _ => unreachable!("Only same width type are allowed to be reinterpreted"),
            },
            PValue::F64(v) => match ptype {
                PType::U64 => unsafe { mem::transmute::<f64, u64>(*v) }.into(),
                PType::I64 => unsafe { mem::transmute::<f64, i64>(*v) }.into(),
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
                    PValue::U8(v) => <$T as NumCast>::from(v),
                    PValue::U16(v) => <$T as NumCast>::from(v),
                    PValue::U32(v) => <$T as NumCast>::from(v),
                    PValue::U64(v) => <$T as NumCast>::from(v),
                    PValue::I8(v) => <$T as NumCast>::from(v),
                    PValue::I16(v) => <$T as NumCast>::from(v),
                    PValue::I32(v) => <$T as NumCast>::from(v),
                    PValue::I64(v) => <$T as NumCast>::from(v),
                    _ => None,
                }
                .ok_or_else(|| {
                    vortex_err!("Cannot read primitive value {:?} as {}", value, PType::$PT)
                })
            }
        }
    };
}

int_pvalue!(u8, U8);
int_pvalue!(u16, U16);
int_pvalue!(u32, U32);
int_pvalue!(u64, U64);
int_pvalue!(usize, U64);
int_pvalue!(i8, I8);
int_pvalue!(i16, I16);
int_pvalue!(i32, I32);
int_pvalue!(i64, I64);

macro_rules! float_pvalue {
    ($T:ty, $PT:tt) => {
        impl TryFrom<PValue> for $T {
            type Error = VortexError;

            fn try_from(value: PValue) -> Result<Self, Self::Error> {
                match value {
                    PValue::F16(f) => <$T as NumCast>::from(f),
                    PValue::F32(f) => <$T as NumCast>::from(f),
                    PValue::F64(f) => <$T as NumCast>::from(f),
                    _ => None,
                }
                .ok_or_else(|| {
                    vortex_err!("Cannot read primitive value {:?} as {}", value, PType::$PT)
                })
            }
        }
    };
}

float_pvalue!(f32, F32);
float_pvalue!(f64, F64);

impl TryFrom<PValue> for f16 {
    type Error = VortexError;

    fn try_from(value: PValue) -> Result<Self, Self::Error> {
        // We serialize f16 as u16.
        match value {
            PValue::U16(u) => Some(Self::from_bits(u)),
            PValue::F16(u) => Some(u),
            PValue::F32(f) => <Self as NumCast>::from(f),
            PValue::F64(f) => <Self as NumCast>::from(f),
            _ => None,
        }
        .ok_or_else(|| vortex_err!("Cannot read primitive value {:?} as {}", value, PType::F16))
    }
}

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
    fn from(value: usize) -> PValue {
        PValue::U64(value as u64)
    }
}

impl Display for PValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U8(v) => write!(f, "{}_u8", v),
            Self::U16(v) => write!(f, "{}_u16", v),
            Self::U32(v) => write!(f, "{}_u32", v),
            Self::U64(v) => write!(f, "{}_u64", v),
            Self::I8(v) => write!(f, "{}_i8", v),
            Self::I16(v) => write!(f, "{}_i16", v),
            Self::I32(v) => write!(f, "{}_i32", v),
            Self::I64(v) => write!(f, "{}_i64", v),
            Self::F16(v) => write!(f, "{}_f16", v),
            Self::F32(v) => write!(f, "{}_f32", v),
            Self::F64(v) => write!(f, "{}_f64", v),
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::half::f16;
    use vortex_dtype::PType;

    use crate::PValue;

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
}
