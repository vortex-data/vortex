use std::fmt::{Display, Formatter};

use vortex_dtype::DType;

use crate::Scalar;

impl Display for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.dtype() {
            DType::Null => write!(f, "null"),
            DType::Bool(_) => write!(f, "{}", self.as_bool()),
            DType::Primitive(..) => write!(f, "{}", self.as_primitive()),
            DType::Decimal(..) => write!(f, "{}", self.as_decimal()),
            DType::Utf8(_) => write!(f, "{}", self.as_utf8()),
            DType::Binary(_) => write!(f, "{}", self.as_binary()),
            DType::Struct(..) => write!(f, "{}", self.as_struct()),
            DType::List(..) => write!(f, "{}", self.as_list()),
            DType::Extension(_) => write!(f, "{}", self.as_extension()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::ByteBuffer;
    use vortex_dtype::Nullability::{NonNullable, Nullable};
    use vortex_dtype::datetime::{DATE_ID, TIME_ID, TIMESTAMP_ID, TemporalMetadata, TimeUnit};
    use vortex_dtype::{DType, ExtDType, ExtMetadata, PType, StructDType};

    use crate::{InnerScalarValue, PValue, Scalar, ScalarValue};

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
        assert_eq!(format!("{}", Scalar::from(0u8)), "0u8");
        assert_eq!(format!("{}", Scalar::from(255u8)), "255u8");

        assert_eq!(format!("{}", Scalar::from(0u16)), "0u16");
        assert_eq!(format!("{}", Scalar::from(!0u16)), "65535u16");

        assert_eq!(format!("{}", Scalar::from(0u32)), "0u32");
        assert_eq!(format!("{}", Scalar::from(!0u32)), "4294967295u32");

        assert_eq!(format!("{}", Scalar::from(0u64)), "0u64");
        assert_eq!(
            format!("{}", Scalar::from(!0u64)),
            "18446744073709551615u64"
        );

        assert_eq!(
            format!("{}", Scalar::null(DType::Primitive(PType::U8, Nullable))),
            "null"
        );
    }

    #[test]
    fn display_utf8() {
        assert_eq!(
            format!("{}", Scalar::from("Hello World!")),
            "\"Hello World!\""
        );
        assert_eq!(format!("{}", Scalar::null(DType::Utf8(Nullable))), "null");
    }

    #[test]
    fn display_binary() {
        assert_eq!(
            format!(
                "{}",
                Scalar::binary(
                    ByteBuffer::from("Hello World!".as_bytes().to_vec()),
                    NonNullable
                )
            ),
            "\"48 65 6c 6c 6f 20 57 6f 72 6c 64 21\""
        );
        assert_eq!(format!("{}", Scalar::null(DType::Binary(Nullable))), "null");
    }

    #[test]
    fn display_empty_struct() {
        fn dtype() -> DType {
            DType::Struct(Arc::new(StructDType::new([].into(), vec![])), Nullable)
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(format!("{}", Scalar::struct_(dtype(), vec![])), "{}");
    }

    #[test]
    fn display_one_field_struct() {
        fn dtype() -> DType {
            DType::Struct(
                Arc::new(StructDType::new(
                    [Arc::from("foo")].into(),
                    vec![DType::Primitive(PType::U32, Nullable)],
                )),
                Nullable,
            )
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(dtype(), vec![Scalar::null_typed::<u32>()])
            ),
            "{foo: null}"
        );

        assert_eq!(
            format!("{}", Scalar::struct_(dtype(), vec![Scalar::from(32_u32)])),
            "{foo: 32u32}"
        );
    }

    #[test]
    fn display_two_field_struct() {
        // fn dtype() -> (DType, DType, DType) {
        let f1 = DType::Bool(Nullable);
        let f2 = DType::Primitive(PType::U32, Nullable);
        let dtype = DType::Struct(
            Arc::new(StructDType::new(
                [Arc::from("foo"), Arc::from("bar")].into(),
                vec![f1.clone(), f2.clone()],
            )),
            Nullable,
        );
        // }

        assert_eq!(format!("{}", Scalar::null(dtype.clone())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(
                    dtype.clone(),
                    vec![Scalar::null(f1), Scalar::null(f2.clone())]
                )
            ),
            "{foo: null, bar: null}"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(dtype.clone(), vec![Scalar::from(true), Scalar::null(f2)])
            ),
            "{foo: true, bar: null}"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(dtype, vec![Scalar::from(true), Scalar::from(32_u32)])
            ),
            "{foo: true, bar: 32u32}"
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
                    ScalarValue(InnerScalarValue::Primitive(PValue::I32(3 * MINUTES + 25)))
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
                Scalar::new(
                    dtype(),
                    ScalarValue(InnerScalarValue::Primitive(PValue::I32(25)))
                )
            ),
            "1970-01-26"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::new(
                    dtype(),
                    ScalarValue(InnerScalarValue::Primitive(PValue::I32(365)))
                )
            ),
            "1971-01-01"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::new(
                    dtype(),
                    ScalarValue(InnerScalarValue::Primitive(PValue::I32(365 * 4)))
                )
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
                    ScalarValue(InnerScalarValue::Primitive(PValue::I32(
                        3 * DAYS + 2 * HOURS + 5 * MINUTES + 10
                    )))
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
                Scalar::new(
                    dtype(),
                    ScalarValue(InnerScalarValue::Primitive(PValue::I32(0)))
                )
            ),
            "1970-01-01T10:00:00+10:00[Pacific/Guam]"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::new(
                    dtype(),
                    ScalarValue(InnerScalarValue::Primitive(PValue::I32(
                        3 * DAYS + 2 * HOURS + 5 * MINUTES + 10
                    )))
                )
            ),
            "1970-01-04T12:05:10+10:00[Pacific/Guam]"
        );
    }
}
