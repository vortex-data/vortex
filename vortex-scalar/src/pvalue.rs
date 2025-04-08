use core::fmt::Display;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::mem;

use num_traits::NumCast;
use paste::paste;
use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PType, ToBytes};
use vortex_error::{VortexError, VortexExpect, vortex_err};

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
        self.to_le_bytes().hash(state);
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

impl TryFrom<PValue> for f64 {
    type Error = VortexError;

    fn try_from(value: PValue) -> Result<Self, Self::Error> {
        // We serialize f64 as u64, but this can also sometimes be narrowed down to u8 if e.g. == 0
        match value {
            PValue::U8(u) => Some(Self::from_bits(u as u64)),
            PValue::U16(u) => Some(Self::from_bits(u as u64)),
            PValue::U32(u) => Some(Self::from_bits(u as u64)),
            PValue::U64(u) => Some(Self::from_bits(u)),
            PValue::F16(f) => <Self as NumCast>::from(f),
            PValue::F32(f) => <Self as NumCast>::from(f),
            PValue::F64(f) => <Self as NumCast>::from(f),
            _ => None,
        }
        .ok_or_else(|| vortex_err!("Cannot read primitive value {:?} as {}", value, PType::F64))
    }
}

impl TryFrom<PValue> for f32 {
    type Error = VortexError;

    #[allow(clippy::cast_possible_truncation)]
    fn try_from(value: PValue) -> Result<Self, Self::Error> {
        // We serialize f32 as u32, but this can also sometimes be narrowed down to u8 if e.g. == 0
        match value {
            PValue::U8(u) => Some(Self::from_bits(u as u32)),
            PValue::U16(u) => Some(Self::from_bits(u as u32)),
            PValue::U32(u) => Some(Self::from_bits(u)),
            // We assume that the value was created from a valid f16 and only changed in serialization
            PValue::U64(u) => <Self as NumCast>::from(Self::from_bits(u as u32)),
            PValue::F16(f) => <Self as NumCast>::from(f),
            PValue::F32(f) => <Self as NumCast>::from(f),
            PValue::F64(f) => <Self as NumCast>::from(f),
            _ => None,
        }
        .ok_or_else(|| vortex_err!("Cannot read primitive value {:?} as {}", value, PType::F32))
    }
}

impl TryFrom<PValue> for f16 {
    type Error = VortexError;

    #[allow(clippy::cast_possible_truncation)]
    fn try_from(value: PValue) -> Result<Self, Self::Error> {
        // We serialize f16 as u16, but this can also sometimes be narrowed down to u8 if e.g. == 0
        match value {
            PValue::U8(u) => Some(Self::from_bits(u as u16)),
            PValue::U16(u) => Some(Self::from_bits(u)),
            // We assume that the value was created from a valid f16 and only changed in serialization
            PValue::U32(u) => Some(Self::from_bits(u as u16)),
            PValue::U64(u) => Some(Self::from_bits(u as u16)),
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
            Self::U8(v) => write!(f, "{}u8", v),
            Self::U16(v) => write!(f, "{}u16", v),
            Self::U32(v) => write!(f, "{}u32", v),
            Self::U64(v) => write!(f, "{}u64", v),
            Self::I8(v) => write!(f, "{}i8", v),
            Self::I16(v) => write!(f, "{}i16", v),
            Self::I32(v) => write!(f, "{}i32", v),
            Self::I64(v) => write!(f, "{}i64", v),
            Self::F16(v) => write!(f, "{}f16", v),
            Self::F32(v) => write!(f, "{}f32", v),
            Self::F64(v) => write!(f, "{}f64", v),
        }
    }
}

#[cfg(test)]
mod test {
    use std::cmp::Ordering;

    use vortex_dtype::PType;
    use vortex_dtype::half::f16;

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
}
