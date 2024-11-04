use std::fmt::{Display, Formatter};

use itertools::Itertools;
use vortex_datetime_dtype::{is_temporal_ext_type, TemporalMetadata};
use vortex_dtype::DType;

use crate::binary::BinaryScalar;
use crate::extension::ExtScalar;
use crate::struct_::StructScalar;
use crate::utf8::Utf8Scalar;
use crate::{PValue, Scalar, ScalarValue};

impl Display for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.dtype() {
            DType::Null | DType::Bool(_) | DType::Primitive(..) => Display::fmt(&self.value, f),
            DType::Utf8(_) => {
                match Utf8Scalar::try_from(self)
                    .map_err(|_| std::fmt::Error)?
                    .value()
                {
                    None => write!(f, "null"),
                    Some(bs) => write!(f, "{}", bs.as_str()),
                }
            }
            DType::Binary(_) => {
                match BinaryScalar::try_from(self)
                    .map_err(|_| std::fmt::Error)?
                    .value()
                {
                    None => write!(f, "null"),
                    Some(buf) => {
                        write!(
                            f,
                            "{}",
                            buf.as_slice().iter().map(|b| format!("{b:x}")).format(",")
                        )
                    }
                }
            }
            DType::Struct(dtype, _) => {
                let v = StructScalar::try_from(self).map_err(|_| std::fmt::Error)?;

                if v.is_null() {
                    write!(f, "null")
                } else {
                    write!(f, "{{")?;
                    let formatted_fields = dtype
                        .names()
                        .iter()
                        .enumerate()
                        .map(|(idx, name)| match v.field_by_idx(idx) {
                            None => format!("{name}:null"),
                            Some(val) => format!("{name}:{val}"),
                        })
                        .format(",");
                    write!(f, "{}", formatted_fields)?;
                    write!(f, "}}")
                }
            }
            DType::List(..) => todo!(),
            // Specialized handling for date/time/timestamp builtin extension types.
            DType::Extension(dtype) if is_temporal_ext_type(dtype.id()) => {
                let metadata =
                    TemporalMetadata::try_from(dtype.as_ref()).map_err(|_| std::fmt::Error)?;
                match ExtScalar::try_from(self)
                    .map_err(|_| std::fmt::Error)?
                    .value()
                {
                    ScalarValue::Null => write!(f, "null"),
                    ScalarValue::Primitive(PValue::I32(v)) => {
                        write!(
                            f,
                            "{}",
                            metadata.to_jiff(*v as i64).map_err(|_| std::fmt::Error)?
                        )
                    }
                    ScalarValue::Primitive(PValue::I64(v)) => {
                        write!(f, "{}", metadata.to_jiff(*v).map_err(|_| std::fmt::Error)?)
                    }
                    _ => Err(std::fmt::Error),
                }
            }
            // Generic handling of unknown extension types.
            // TODO(aduffy): Allow extension authors plugin their own Scalar display.
            DType::Extension(..) => {
                let scalar_value = ExtScalar::try_from(self)
                    .map_err(|_| std::fmt::Error)?
                    .value();
                if scalar_value.is_null() {
                    write!(f, "null")
                } else {
                    write!(f, "{}", scalar_value)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::Buffer;
    use vortex_datetime_dtype::{TemporalMetadata, TimeUnit, DATE_ID, TIMESTAMP_ID, TIME_ID};
    use vortex_dtype::Nullability::{NonNullable, Nullable};
    use vortex_dtype::{DType, ExtDType, ExtMetadata, PType, StructDType};

    use crate::{PValue, Scalar, ScalarValue};

    const MINUTES: i32 = 60;
    const HOURS: i32 = 60 * MINUTES;
    const DAYS: i32 = 24 * HOURS;

    #[test]
    fn display_bool() {
        assert_eq!(format!("{}", Scalar::from(false)), "false");
        assert_eq!(format!("{}", Scalar::from(true)), "true");
        assert_eq!(format!("{}", Scalar::null(DType::Bool(Nullable))), "null");
    }

    #[test]
    fn display_primitive() {
        assert_eq!(format!("{}", Scalar::from(0_u8)), "0_u8");
        assert_eq!(format!("{}", Scalar::from(255_u8)), "255_u8");

        assert_eq!(format!("{}", Scalar::from(0_u16)), "0_u16");
        assert_eq!(format!("{}", Scalar::from(!0_u16)), "65535_u16");

        assert_eq!(format!("{}", Scalar::from(0_u32)), "0_u32");
        assert_eq!(format!("{}", Scalar::from(!0_u32)), "4294967295_u32");

        assert_eq!(format!("{}", Scalar::from(0_u64)), "0_u64");
        assert_eq!(
            format!("{}", Scalar::from(!0_u64)),
            "18446744073709551615_u64"
        );

        assert_eq!(
            format!("{}", Scalar::null(DType::Primitive(PType::U8, Nullable))),
            "null"
        );
    }

    #[test]
    fn display_utf8() {
        assert_eq!(format!("{}", Scalar::from("Hello World!")), "Hello World!");
        assert_eq!(format!("{}", Scalar::null(DType::Utf8(Nullable))), "null");
    }

    #[test]
    fn display_binary() {
        assert_eq!(
            format!(
                "{}",
                Scalar::binary(Buffer::from("Hello World!".as_bytes()), NonNullable)
            ),
            "48,65,6c,6c,6f,20,57,6f,72,6c,64,21"
        );
        assert_eq!(format!("{}", Scalar::null(DType::Binary(Nullable))), "null");
    }

    #[test]
    fn display_empty_struct() {
        fn dtype() -> DType {
            DType::Struct(StructDType::new(Arc::new([]), vec![]), Nullable)
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(format!("{}", Scalar::r#struct(dtype(), vec![])), "{}");
    }

    #[test]
    fn display_one_field_struct() {
        fn dtype() -> DType {
            DType::Struct(
                StructDType::new(
                    Arc::new([Arc::from("foo")]),
                    vec![DType::Primitive(PType::U32, Nullable)],
                ),
                Nullable,
            )
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!("{}", Scalar::r#struct(dtype(), vec![ScalarValue::Null])),
            "{foo:null}"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::r#struct(dtype(), vec![ScalarValue::Primitive(PValue::U32(32))])
            ),
            "{foo:32_u32}"
        );
    }

    #[test]
    fn display_two_field_struct() {
        fn dtype() -> DType {
            DType::Struct(
                StructDType::new(
                    Arc::new([Arc::from("foo"), Arc::from("bar")]),
                    vec![
                        DType::Bool(Nullable),
                        DType::Primitive(PType::U32, Nullable),
                    ],
                ),
                Nullable,
            )
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!("{}", Scalar::r#struct(dtype(), vec![])),
            "{foo:null,bar:null}"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::r#struct(dtype(), vec![ScalarValue::Bool(true)])
            ),
            "{foo:true,bar:null}"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::r#struct(
                    dtype(),
                    vec![
                        ScalarValue::Bool(true),
                        ScalarValue::Primitive(PValue::U32(32))
                    ]
                )
            ),
            "{foo:true,bar:32_u32}"
        );
    }

    #[test]
    fn display_time() {
        fn dtype() -> DType {
            DType::Extension(Arc::new(ExtDType::new(
                TIME_ID.clone(),
                Arc::new(DType::Primitive(PType::I32, Nullable)),
                Some(ExtMetadata::from(TemporalMetadata::Time(TimeUnit::S))),
            )))
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::new(
                    dtype(),
                    ScalarValue::Primitive(PValue::I32(3 * MINUTES + 25))
                )
            ),
            "00:03:25"
        );
    }

    #[test]
    fn display_date() {
        fn dtype() -> DType {
            DType::Extension(Arc::new(ExtDType::new(
                DATE_ID.clone(),
                Arc::new(DType::Primitive(PType::I32, Nullable)),
                Some(ExtMetadata::from(TemporalMetadata::Date(TimeUnit::D))),
            )))
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::new(dtype(), ScalarValue::Primitive(PValue::I32(25)))
            ),
            "1970-01-26"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::new(dtype(), ScalarValue::Primitive(PValue::I32(365)))
            ),
            "1971-01-01"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::new(dtype(), ScalarValue::Primitive(PValue::I32(365 * 4)))
            ),
            "1973-12-31"
        );
    }

    #[test]
    fn display_local_timestamp() {
        fn dtype() -> DType {
            DType::Extension(Arc::new(ExtDType::new(
                TIMESTAMP_ID.clone(),
                Arc::new(DType::Primitive(PType::I32, Nullable)),
                Some(ExtMetadata::from(TemporalMetadata::Timestamp(
                    TimeUnit::S,
                    None,
                ))),
            )))
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::new(
                    dtype(),
                    ScalarValue::Primitive(PValue::I32(3 * DAYS + 2 * HOURS + 5 * MINUTES + 10))
                )
            ),
            "1970-01-04T02:05:10Z"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn display_zoned_timestamp() {
        fn dtype() -> DType {
            DType::Extension(Arc::new(ExtDType::new(
                TIMESTAMP_ID.clone(),
                Arc::new(DType::Primitive(PType::I64, Nullable)),
                Some(ExtMetadata::from(TemporalMetadata::Timestamp(
                    TimeUnit::S,
                    Some(String::from("Pacific/Guam")),
                ))),
            )))
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::new(dtype(), ScalarValue::Primitive(PValue::I32(0)))
            ),
            "1970-01-01T10:00:00+10:00[Pacific/Guam]"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::new(
                    dtype(),
                    ScalarValue::Primitive(PValue::I32(3 * DAYS + 2 * HOURS + 5 * MINUTES + 10))
                )
            ),
            "1970-01-04T12:05:10+10:00[Pacific/Guam]"
        );
    }
}
