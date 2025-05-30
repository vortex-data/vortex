use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::Hash;

use vortex_dtype::{DType, DecimalDType, Nullability};
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_bail};

use crate::scalar_value::InnerScalarValue;
use crate::{BigCast, Scalar, ScalarValue, ToPrimitive, i256};

#[macro_export]
macro_rules! match_each_decimal_value {
    ($self:expr, | $value:ident | $body:block) => {{
        match $self {
            DecimalValue::I8(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I16(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I32(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I64(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I128(v) => {
                let $value = v;
                $body
            }
            DecimalValue::I256(v) => {
                let $value = v;
                $body
            }
        }
    }};
}

/// Macro to match over each decimal value type, binding the corresponding native type (from `DecimalValueType`)
#[macro_export]
macro_rules! match_each_decimal_value_type {
    ($self:expr, | $enc:ident | $body:block) => {{
        use $crate::{DecimalValueType, i256};
        match $self {
            DecimalValueType::I8 => {
                type $enc = i8;
                $body
            }
            DecimalValueType::I16 => {
                type $enc = i16;
                $body
            }
            DecimalValueType::I32 => {
                type $enc = i32;
                $body
            }
            DecimalValueType::I64 => {
                type $enc = i64;
                $body
            }
            DecimalValueType::I128 => {
                type $enc = i128;
                $body
            }
            DecimalValueType::I256 => {
                type $enc = i256;
                $body
            }
            ty => unreachable!("unknown decimal value type {:?}", ty),
        }
    }};
}

/// Type of the decimal values.
#[derive(Clone, Copy, Debug, prost::Enumeration, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum DecimalValueType {
    I8 = 0,
    I16 = 1,
    I32 = 2,
    I64 = 3,
    I128 = 4,
    I256 = 5,
}

#[derive(Debug, Clone, Copy)]
pub enum DecimalValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    I256(i256),
}

// Comparisons between DecimalValue types should upcast to i256 and operate in the upcast space.
// Decimal values can take on any signed scalar type, but so long as their values are the same
// they are considered the same.
// DecimalScalar handles ensuring that both values being compared have the same precision/scale.
impl PartialEq for DecimalValue {
    fn eq(&self, other: &Self) -> bool {
        let self_upcast = match_each_decimal_value!(self, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        });
        let other_upcast = match_each_decimal_value!(other, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        });

        self_upcast == other_upcast
    }
}

impl Eq for DecimalValue {}

impl PartialOrd for DecimalValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let self_upcast = match_each_decimal_value!(self, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        });
        let other_upcast = match_each_decimal_value!(other, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        });

        self_upcast.partial_cmp(&other_upcast)
    }
}

// Hashing works in the upcast space similar to the other comparison and equality operators.
impl Hash for DecimalValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let self_upcast = match_each_decimal_value!(self, |v| {
            v.to_i256()
                .vortex_expect("upcast to i256 must always succeed")
        });
        self_upcast.hash(state);
    }
}

/// Type of decimal scalar values.
pub trait NativeDecimalType:
    Copy
    + Eq
    + Ord
    + Default
    + Send
    + Sync
    + BigCast
    // + AsPrimitive<i8>
    // + AsPrimitive<i16>
    // + AsPrimitive<i32>
    // + AsPrimitive<i64>
    // + AsPrimitive<i128>
    // + AsPrimitive<i256>
    + 'static
{
    const VALUES_TYPE: DecimalValueType;

    fn try_from(decimal_type: DecimalValue) -> Option<Self>;
}

impl NativeDecimalType for i8 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I8;

    fn try_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I8(v) => Some(v),
            _ => None,
        }
    }
}

impl NativeDecimalType for i16 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I16;

    fn try_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I16(v) => Some(v),
            _ => None,
        }
    }
}

impl NativeDecimalType for i32 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I32;

    fn try_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I32(v) => Some(v),
            _ => None,
        }
    }
}

impl NativeDecimalType for i64 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I64;

    fn try_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I64(v) => Some(v),
            _ => None,
        }
    }
}

impl NativeDecimalType for i128 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I128;

    fn try_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I128(v) => Some(v),
            _ => None,
        }
    }
}

impl NativeDecimalType for i256 {
    const VALUES_TYPE: DecimalValueType = DecimalValueType::I256;

    fn try_from(decimal_type: DecimalValue) -> Option<Self> {
        match decimal_type {
            DecimalValue::I256(v) => Some(v),
            _ => None,
        }
    }
}

impl Display for DecimalValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DecimalValue::I8(v8) => write!(f, "decimal8({v8})"),
            DecimalValue::I16(v16) => write!(f, "decimal16({v16})"),
            DecimalValue::I32(v32) => write!(f, "decimal32({v32})"),
            DecimalValue::I64(v32) => write!(f, "decimal64({v32})"),
            DecimalValue::I128(v128) => write!(f, "decimal128({v128})"),
            DecimalValue::I256(v256) => write!(f, "decimal256({v256})"),
        }
    }
}

impl Scalar {
    pub fn decimal(
        value: DecimalValue,
        decimal_type: DecimalDType,
        nullability: Nullability,
    ) -> Self {
        Self::new(
            DType::Decimal(decimal_type, nullability),
            ScalarValue(InnerScalarValue::Decimal(value)),
        )
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct DecimalScalar<'a> {
    dtype: &'a DType,
    decimal_type: DecimalDType,
    value: Option<DecimalValue>,
}

impl<'a> DecimalScalar<'a> {
    pub fn try_new(dtype: &'a DType, value: &ScalarValue) -> VortexResult<Self> {
        let decimal_type = DecimalDType::try_from(dtype)?;
        let value = value.as_decimal()?;

        Ok(Self {
            dtype,
            decimal_type,
            value,
        })
    }

    #[inline]
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    pub fn decimal_value(&self) -> &Option<DecimalValue> {
        &self.value
    }
}

impl<'a> TryFrom<&'a Scalar> for DecimalScalar<'a> {
    type Error = VortexError;

    fn try_from(scalar: &'a Scalar) -> Result<Self, Self::Error> {
        DecimalScalar::try_new(&scalar.dtype, &scalar.value)
    }
}

impl Display for DecimalScalar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.value.as_ref() {
            Some(&dv) => {
                // Introduce some of the scale factors instead.
                match dv {
                    DecimalValue::I8(v) => write!(
                        f,
                        "decimal8({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I16(v) => write!(
                        f,
                        "decimal16({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I32(v) => write!(
                        f,
                        "decimal32({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I64(v) => write!(
                        f,
                        "decimal64({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I128(v) => write!(
                        f,
                        "decimal128({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                    DecimalValue::I256(v) => write!(
                        f,
                        "decimal256({}, precision={}, scale={})",
                        v,
                        self.decimal_type.precision(),
                        self.decimal_type.scale()
                    ),
                }
            }
            None => {
                write!(f, "null")
            }
        }
    }
}

impl PartialEq for DecimalScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype.eq_ignore_nullability(other.dtype) && self.value == other.value
    }
}

impl Eq for DecimalScalar<'_> {}

/// Ord is not implemented since it's undefined for different PTypes
impl PartialOrd for DecimalScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !self.dtype.eq_ignore_nullability(other.dtype) {
            return None;
        }
        self.value.partial_cmp(&other.value)
    }
}

macro_rules! decimal_scalar_unpack {
    ($ty:ident, $arm:ident) => {
        impl TryFrom<DecimalScalar<'_>> for Option<$ty> {
            type Error = VortexError;

            fn try_from(value: DecimalScalar) -> Result<Self, Self::Error> {
                Ok(match value.value {
                    None => None,
                    Some(DecimalValue::$arm(v)) => Some(v),
                    v => vortex_bail!("Cannot extract decimal {:?} as {}", v, stringify!($ty)),
                })
            }
        }

        impl TryFrom<DecimalScalar<'_>> for $ty {
            type Error = VortexError;

            fn try_from(value: DecimalScalar) -> Result<Self, Self::Error> {
                match value.value {
                    None => vortex_bail!("Cannot extract value from null decimal"),
                    Some(DecimalValue::$arm(v)) => Ok(v),
                    v => vortex_bail!("Cannot extract decimal {:?} as {}", v, stringify!($ty)),
                }
            }
        }
    };
}

decimal_scalar_unpack!(i8, I8);
decimal_scalar_unpack!(i16, I16);
decimal_scalar_unpack!(i32, I32);
decimal_scalar_unpack!(i64, I64);
decimal_scalar_unpack!(i128, I128);
decimal_scalar_unpack!(i256, I256);

macro_rules! decimal_scalar_pack {
    ($from:ident, $to:ident, $arm:ident) => {
        impl From<$from> for DecimalValue {
            fn from(value: $from) -> Self {
                DecimalValue::$arm(value as $to)
            }
        }
    };
}

decimal_scalar_pack!(i8, i8, I8);
decimal_scalar_pack!(u8, i16, I16);
decimal_scalar_pack!(i16, i16, I16);
decimal_scalar_pack!(u16, i32, I32);
decimal_scalar_pack!(i32, i32, I32);
decimal_scalar_pack!(u32, i64, I64);
decimal_scalar_pack!(i64, i64, I64);
decimal_scalar_pack!(u64, i128, I128);

decimal_scalar_pack!(i128, i128, I128);
decimal_scalar_pack!(i256, i256, I256);

#[cfg(test)]
#[allow(clippy::disallowed_types)]
mod tests {
    use std::collections::HashSet;

    use rstest::rstest;

    use crate::{DecimalValue, i256};

    #[rstest]
    #[case(DecimalValue::I8(100), DecimalValue::I8(100))]
    #[case(DecimalValue::I16(0), DecimalValue::I256(i256::ZERO))]
    #[case(DecimalValue::I8(100), DecimalValue::I128(100))]
    fn test_decimal_value_eq(#[case] left: DecimalValue, #[case] right: DecimalValue) {
        assert_eq!(left, right);
    }

    #[rstest]
    #[case(DecimalValue::I128(10), DecimalValue::I8(11))]
    #[case(DecimalValue::I256(i256::ZERO), DecimalValue::I16(10))]
    #[case(DecimalValue::I128(-1_000), DecimalValue::I8(1))]
    fn test_decimal_value_cmp(#[case] lower: DecimalValue, #[case] upper: DecimalValue) {
        assert!(lower < upper, "expected {lower} < {upper}");
    }

    #[test]
    fn test_hash() {
        let mut set = HashSet::new();
        set.insert(DecimalValue::I8(100));
        set.insert(DecimalValue::I16(100));
        set.insert(DecimalValue::I32(100));
        set.insert(DecimalValue::I64(100));
        set.insert(DecimalValue::I128(100));
        set.insert(DecimalValue::I256(i256::from_i128(100)));
        assert_eq!(set.len(), 1);
    }
}
