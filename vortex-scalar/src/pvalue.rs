#![allow(clippy::unwrap_used)]

use core::fmt::Display;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::mem;

use num_traits::NumCast;
use paste::paste;
use vortex_avro::{AvroValue, FromAvro, ToAvro};
use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PType};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect};
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

fn record(
    name: &str,
    fields: Vec<vortex_avro::avro_private::RecordField>,
) -> vortex_avro::avro_private::Schema {
    vortex_avro::avro_private::Schema::Record(vortex_avro::avro_private::RecordSchema {
        name: vortex_avro::avro_private::Name::new(name).unwrap(),
        aliases: None,
        doc: None,
        fields,
        lookup: BTreeMap::new(),
        attributes: BTreeMap::new(),
    })
}

fn field(
    name: &str,
    field_type: vortex_avro::avro_private::Schema,
    pos: usize,
) -> vortex_avro::avro_private::RecordField {
    vortex_avro::avro_private::RecordField {
        name: name.to_string(),
        doc: None,
        aliases: None,
        position: pos,
        schema: field_type,
        custom_attributes: BTreeMap::new(),
        default: None,
        order: vortex_avro::avro_private::RecordFieldOrder::Ignore,
    }
}

fn pvalue_avro_schema() -> vortex_avro::avro_private::Schema {
    vortex_avro::avro_private::Schema::Union(
        vortex_avro::avro_private::UnionSchema::new(vec![
            // tag 0: u8
            record(
                "PValue__u8",
                vec![field("value", vortex_avro::avro_private::Schema::Int, 0)],
            ),
            // tag 1: u16
            record(
                "PValue__u16",
                vec![field("value", vortex_avro::avro_private::Schema::Int, 0)],
            ),
            // tag 2: u32
            record(
                "PValue__u32",
                vec![field("value", vortex_avro::avro_private::Schema::Int, 0)],
            ),
            // tag 3: u64
            record(
                "PValue__u64",
                vec![field("value", vortex_avro::avro_private::Schema::Long, 0)],
            ),
            // tag 4: i8
            record(
                "PValue__i8",
                vec![field("value", vortex_avro::avro_private::Schema::Int, 0)],
            ),
            // tag 5: i16
            record(
                "PValue__i16",
                vec![field("value", vortex_avro::avro_private::Schema::Int, 0)],
            ),
            // tag 6: i32
            record(
                "PValue__i32",
                vec![field("value", vortex_avro::avro_private::Schema::Int, 0)],
            ),
            // tag 7: i64
            record(
                "PValue__i64",
                vec![field("value", vortex_avro::avro_private::Schema::Long, 0)],
            ),
            // tag 8: f32
            record(
                "PValue__f32",
                vec![field("value", vortex_avro::avro_private::Schema::Float, 0)],
            ),
            // tag 9: f64
            record(
                "PValue__f64",
                vec![field("value", vortex_avro::avro_private::Schema::Double, 0)],
            ),
        ])
        .unwrap(),
    )
}

// Impl of FromAvro for PValue
// In Avro, we will always serialize a PValue as an Int64, as that is the maximum width of PValue.
// We can then cast the bits accordingly on read.
impl TryFrom<AvroValue> for PValue {
    type Error = VortexError;

    fn try_from(value: AvroValue) -> Result<Self, Self::Error> {
        let AvroValue::Union(tag, value) = value else {
            vortex_bail!("Expected AvroValue::Long, got {:?}", value);
        };

        match (tag, *value) {
            (0, AvroValue::Record(fields)) => {
                let Some(field) = fields.first() else {
                    vortex_bail!("Expected PValue::U8 Avro value to have value field");
                };

                let AvroValue::Int(v) = field.1 else {
                    vortex_bail!("Expected PValue::U8 Avro value to have value field as Int");
                };

                Ok(PValue::U8(v as u8))
            }
            (1, AvroValue::Record(fields)) => {
                let Some(field) = fields.first() else {
                    vortex_bail!("Expected PValue::U16 Avro value to have value field");
                };

                let AvroValue::Int(v) = field.1 else {
                    vortex_bail!("Expected PValue::U16 Avro value to have value field as Int");
                };

                Ok(PValue::U16(v as u16))
            }
            (2, AvroValue::Record(fields)) => {
                let Some(field) = fields.first() else {
                    vortex_bail!("Expected PValue::U32 Avro value to have value field");
                };

                let AvroValue::Int(v) = field.1 else {
                    vortex_bail!("Expected PValue::U32 Avro value to have value field as Int");
                };

                Ok(PValue::U32(v as u32))
            }
            (3, AvroValue::Record(fields)) => {
                let Some(field) = fields.first() else {
                    vortex_bail!("Expected PValue::U64 Avro value to have value field");
                };

                let AvroValue::Long(v) = field.1 else {
                    vortex_bail!("Expected PValue::U64 Avro value to have value field as Long");
                };

                Ok(PValue::U64(v as u64))
            }
            (4, AvroValue::Record(fields)) => {
                let Some(field) = fields.first() else {
                    vortex_bail!("Expected PValue::I8 Avro value to have value field");
                };

                let AvroValue::Int(v) = field.1 else {
                    vortex_bail!("Expected PValue::I8 Avro value to have value field as Int");
                };

                Ok(PValue::I8(v as i8))
            }
            (5, AvroValue::Record(fields)) => {
                let Some(field) = fields.first() else {
                    vortex_bail!("Expected PValue::I16 Avro value to have value field");
                };

                let AvroValue::Int(v) = field.1 else {
                    vortex_bail!("Expected PValue::I16 Avro value to have value field as Int");
                };

                Ok(PValue::I16(v as i16))
            }
            (6, AvroValue::Record(fields)) => {
                let Some(field) = fields.first() else {
                    vortex_bail!("Expected PValue::I32 Avro value to have value field");
                };

                let AvroValue::Int(v) = field.1 else {
                    vortex_bail!("Expected PValue::I32 Avro value to have value field as Int");
                };

                Ok(PValue::I32(v))
            }
            (7, AvroValue::Record(fields)) => {
                let Some(field) = fields.first() else {
                    vortex_bail!("Expected PValue::I64 Avro value to have value field");
                };

                let AvroValue::Long(v) = field.1 else {
                    vortex_bail!("Expected PValue::I64 Avro value to have value field as Long");
                };

                Ok(PValue::I64(v))
            }
            (8, AvroValue::Record(fields)) => {
                let Some(field) = fields.first() else {
                    vortex_bail!("Expected PValue::F32 Avro value to have value field");
                };

                let AvroValue::Float(v) = field.1 else {
                    vortex_bail!("Expected PValue::F32 Avro value to have value field as Float");
                };

                Ok(PValue::F32(v))
            }
            (9, AvroValue::Record(fields)) => {
                let Some(field) = fields.first() else {
                    vortex_bail!("Expected PValue::F64 Avro value to have value field");
                };

                let AvroValue::Double(v) = field.1 else {
                    vortex_bail!("Expected PValue::F64 Avro value to have value field as Double");
                };

                Ok(PValue::F64(v))
            }
            (tag, value) => vortex_bail!(
                "Avro PValue: invalid (tag, value) pair: ({tag}, {:?})",
                value
            ),
        }
    }
}

impl FromAvro for PValue {
    fn read_schema() -> vortex_avro::avro_private::Schema {
        pvalue_avro_schema()
    }
}

// Impl of ToAvro for PValue
impl From<PValue> for AvroValue {
    fn from(pvalue: PValue) -> Self {
        match pvalue {
            PValue::U8(v) => AvroValue::Union(
                0,
                Box::new(AvroValue::Record(vec![(
                    "value".to_string(),
                    AvroValue::Int(v as i32),
                )])),
            ),
            PValue::U16(v) => AvroValue::Union(
                1,
                Box::new(AvroValue::Record(vec![(
                    "value".to_string(),
                    AvroValue::Int(v as i32),
                )])),
            ),
            PValue::U32(v) => AvroValue::Union(
                2,
                Box::new(AvroValue::Record(vec![(
                    "value".to_string(),
                    AvroValue::Int(v as i32),
                )])),
            ),
            PValue::U64(v) => AvroValue::Union(
                3,
                Box::new(AvroValue::Record(vec![(
                    "value".to_string(),
                    AvroValue::Long(v as i64),
                )])),
            ),
            PValue::I8(v) => AvroValue::Union(
                4,
                Box::new(AvroValue::Record(vec![(
                    "value".to_string(),
                    AvroValue::Int(v as i32),
                )])),
            ),
            PValue::I16(v) => AvroValue::Union(
                5,
                Box::new(AvroValue::Record(vec![(
                    "value".to_string(),
                    AvroValue::Int(v as i32),
                )])),
            ),
            PValue::I32(v) => AvroValue::Union(
                6,
                Box::new(AvroValue::Record(vec![(
                    "value".to_string(),
                    AvroValue::Int(v),
                )])),
            ),
            PValue::I64(v) => AvroValue::Union(
                7,
                Box::new(AvroValue::Record(vec![(
                    "value".to_string(),
                    AvroValue::Long(v),
                )])),
            ),
            PValue::F16(_) => todo!("f16 not supported for PValue Avro serialization"),
            PValue::F32(v) => AvroValue::Union(
                8,
                Box::new(AvroValue::Record(vec![(
                    "value".to_string(),
                    AvroValue::Float(v),
                )])),
            ),
            PValue::F64(v) => AvroValue::Union(
                9,
                Box::new(AvroValue::Record(vec![(
                    "value".to_string(),
                    AvroValue::Double(v),
                )])),
            ),
        }
    }
}

impl ToAvro for PValue {
    fn write_schema() -> vortex_avro::avro_private::Schema {
        pvalue_avro_schema()
    }
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
    use std::cmp::Ordering;

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
