use std::any::type_name;

use num_traits::{FromPrimitive, NumCast};
use vortex_dtype::half::f16;
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability, PType};
use vortex_error::{
    vortex_bail, vortex_err, vortex_panic, VortexError, VortexResult, VortexUnwrap,
};

use crate::pvalue::PValue;
use crate::value::ScalarValue;
use crate::{InnerScalarValue, Scalar};

#[derive(Debug, Clone)]
pub struct PrimitiveScalar<'a> {
    dtype: &'a DType,
    ptype: PType,
    pvalue: Option<PValue>,
}

impl<'a> PrimitiveScalar<'a> {
    pub fn try_new(dtype: &'a DType, value: &ScalarValue) -> VortexResult<Self> {
        if !matches!(dtype, DType::Primitive(..)) {
            vortex_bail!("Expected primitive scalar, found {}", dtype)
        }

        let ptype = PType::try_from(dtype)?;

        // Read the serialized value into the correct PValue.
        // The serialized form may come back over the wire as e.g. any integer type.
        let pvalue = match_each_native_ptype!(ptype, |$T| {
            if let Some(pvalue) = value.as_pvalue()? {
                Some(PValue::from(<$T>::try_from(pvalue)?))
            } else {
                None
            }
        });

        Ok(Self {
            dtype,
            ptype,
            pvalue,
        })
    }

    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    #[inline]
    pub fn ptype(&self) -> PType {
        self.ptype
    }

    pub fn typed_value<T: NativePType + TryFrom<PValue, Error = VortexError>>(&self) -> Option<T> {
        assert_eq!(
            self.ptype,
            T::PTYPE,
            "Attempting to read {} scalar as {}",
            self.ptype,
            T::PTYPE
        );

        self.pvalue.map(|pv| pv.as_primitive::<T>().vortex_unwrap())
    }

    pub fn cast(&self, dtype: &DType) -> VortexResult<Scalar> {
        let ptype = PType::try_from(dtype)?;
        match_each_native_ptype!(ptype, |$Q| {
            match_each_native_ptype!(self.ptype(), |$T| {
                Ok(Scalar::primitive::<$Q>(
                    <$Q as NumCast>::from(self.typed_value::<$T>().expect("Invalid value"))
                        .ok_or_else(|| vortex_err!("Can't cast {} scalar to {}", self.ptype, dtype))?,
                    dtype.nullability(),
                ))
            })
        })
    }

    /// Attempt to extract the primitive value as the given type.
    /// Fails on a bad cast.
    pub fn as_<T: FromPrimitiveOrF16>(&self) -> VortexResult<Option<T>> {
        match self.pvalue {
            None => Ok(None),
            Some(pv) => Ok(Some(match pv {
                PValue::U8(v) => T::from_u8(v)
                    .ok_or_else(|| vortex_err!("Failed to cast u8 to {}", type_name::<T>())),
                PValue::U16(v) => T::from_u16(v)
                    .ok_or_else(|| vortex_err!("Failed to cast u16 to {}", type_name::<T>())),
                PValue::U32(v) => T::from_u32(v)
                    .ok_or_else(|| vortex_err!("Failed to cast u32 to {}", type_name::<T>())),
                PValue::U64(v) => T::from_u64(v)
                    .ok_or_else(|| vortex_err!("Failed to cast u64 to {}", type_name::<T>())),
                PValue::I8(v) => T::from_i8(v)
                    .ok_or_else(|| vortex_err!("Failed to cast i8 to {}", type_name::<T>())),
                PValue::I16(v) => T::from_i16(v)
                    .ok_or_else(|| vortex_err!("Failed to cast i16 to {}", type_name::<T>())),
                PValue::I32(v) => T::from_i32(v)
                    .ok_or_else(|| vortex_err!("Failed to cast i32 to {}", type_name::<T>())),
                PValue::I64(v) => T::from_i64(v)
                    .ok_or_else(|| vortex_err!("Failed to cast i64 to {}", type_name::<T>())),
                PValue::F16(v) => T::from_f16(v)
                    .ok_or_else(|| vortex_err!("Failed to cast f16 to {}", type_name::<T>())),
                PValue::F32(v) => T::from_f32(v)
                    .ok_or_else(|| vortex_err!("Failed to cast f32 to {}", type_name::<T>())),
                PValue::F64(v) => T::from_f64(v)
                    .ok_or_else(|| vortex_err!("Failed to cast f64 to {}", type_name::<T>())),
            }?)),
        }
    }
}

pub trait FromPrimitiveOrF16: FromPrimitive {
    fn from_f16(v: f16) -> Option<Self>;
}

macro_rules! from_primitive_or_f16_for_non_floating_point {
    ($T:ty) => {
        impl FromPrimitiveOrF16 for $T {
            fn from_f16(_: f16) -> Option<Self> {
                None
            }
        }
    };
}

from_primitive_or_f16_for_non_floating_point!(u8);
from_primitive_or_f16_for_non_floating_point!(u16);
from_primitive_or_f16_for_non_floating_point!(u32);
from_primitive_or_f16_for_non_floating_point!(u64);
from_primitive_or_f16_for_non_floating_point!(i8);
from_primitive_or_f16_for_non_floating_point!(i16);
from_primitive_or_f16_for_non_floating_point!(i32);
from_primitive_or_f16_for_non_floating_point!(i64);

impl FromPrimitiveOrF16 for f16 {
    fn from_f16(v: f16) -> Option<Self> {
        Some(v)
    }
}

impl FromPrimitiveOrF16 for f32 {
    fn from_f16(v: f16) -> Option<Self> {
        Some(v.to_f32())
    }
}

impl FromPrimitiveOrF16 for f64 {
    fn from_f16(v: f16) -> Option<Self> {
        Some(v.to_f64())
    }
}

impl<'a> TryFrom<&'a Scalar> for PrimitiveScalar<'a> {
    type Error = VortexError;

    fn try_from(value: &'a Scalar) -> Result<Self, Self::Error> {
        Self::try_new(value.dtype(), value.value())
    }
}

impl Scalar {
    pub fn primitive<T: NativePType + Into<PValue>>(value: T, nullability: Nullability) -> Self {
        Self {
            dtype: DType::Primitive(T::PTYPE, nullability),
            value: ScalarValue(InnerScalarValue::Primitive(value.into())),
        }
    }

    pub fn reinterpret_cast(&self, ptype: PType) -> Self {
        let primitive = PrimitiveScalar::try_from(self).unwrap_or_else(|e| {
            vortex_panic!(e, "Failed to reinterpret cast {} to {}", self.dtype, ptype)
        });
        if primitive.ptype() == ptype {
            return self.clone();
        }

        assert_eq!(
            primitive.ptype().byte_width(),
            ptype.byte_width(),
            "can't reinterpret cast between integers of two different widths"
        );

        Scalar::new(
            DType::Primitive(ptype, self.dtype.nullability()),
            primitive
                .pvalue
                .map(|p| p.reinterpret_cast(ptype))
                .map(|x| ScalarValue(InnerScalarValue::Primitive(x)))
                .unwrap_or_else(|| ScalarValue(InnerScalarValue::Null)),
        )
    }

    pub fn zero<T: NativePType + Into<PValue>>(nullability: Nullability) -> Self {
        Self {
            dtype: DType::Primitive(T::PTYPE, nullability),
            value: ScalarValue(InnerScalarValue::Primitive(T::zero().into())),
        }
    }
}

macro_rules! primitive_scalar {
    ($T:ty) => {
        impl TryFrom<&Scalar> for $T {
            type Error = VortexError;

            fn try_from(value: &Scalar) -> Result<Self, Self::Error> {
                <Option<$T>>::try_from(value)?
                    .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
            }
        }

        impl TryFrom<Scalar> for $T {
            type Error = VortexError;

            fn try_from(value: Scalar) -> Result<Self, Self::Error> {
                <$T>::try_from(&value)
            }
        }

        impl TryFrom<&Scalar> for Option<$T> {
            type Error = VortexError;

            fn try_from(value: &Scalar) -> Result<Self, Self::Error> {
                Ok(PrimitiveScalar::try_from(value)?.typed_value::<$T>())
            }
        }

        impl TryFrom<Scalar> for Option<$T> {
            type Error = VortexError;

            fn try_from(value: Scalar) -> Result<Self, Self::Error> {
                <Option<$T>>::try_from(&value)
            }
        }

        impl From<$T> for Scalar {
            fn from(value: $T) -> Self {
                Scalar {
                    dtype: DType::Primitive(<$T>::PTYPE, Nullability::NonNullable),
                    value: ScalarValue(InnerScalarValue::Primitive(value.into())),
                }
            }
        }
    };
}

primitive_scalar!(u8);
primitive_scalar!(u16);
primitive_scalar!(u32);
primitive_scalar!(u64);
primitive_scalar!(i8);
primitive_scalar!(i16);
primitive_scalar!(i32);
primitive_scalar!(i64);
primitive_scalar!(f16);
primitive_scalar!(f32);
primitive_scalar!(f64);

/// Read a scalar as usize. For usize only, we implicitly cast for better ergonomics.
impl TryFrom<&Scalar> for usize {
    type Error = VortexError;

    fn try_from(value: &Scalar) -> Result<Self, Self::Error> {
        PrimitiveScalar::try_from(value)?
            .as_::<u64>()?
            .map(|v| v as Self)
            .ok_or_else(|| vortex_err!("cannot convert Null to usize"))
    }
}

/// Read a scalar as usize. For usize only, we implicitly cast for better ergonomics.
impl From<usize> for Scalar {
    fn from(value: usize) -> Self {
        Scalar::primitive(value as u64, Nullability::NonNullable)
    }
}
