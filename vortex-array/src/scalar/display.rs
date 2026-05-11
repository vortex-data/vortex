// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`Display`] implementations for [`Scalar`].

use std::fmt::Display;
use std::fmt::Formatter;

use crate::dtype::DType;
use crate::scalar::Scalar;

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
            DType::Union(..) => todo!("TODO(connor)[Union]: unimplemented"),
            DType::List(..) | DType::FixedSizeList(..) => write!(f, "{}", self.as_list()),
            DType::Extension(_) => write!(f, "{}", self.as_extension()),
            DType::Variant(_) => write!(f, "{}", self.as_variant()),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;

    use crate::dtype::DType;
    use crate::dtype::FieldName;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::PValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;

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
            DType::Struct(StructFields::new(Default::default(), vec![]), Nullable)
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(format!("{}", Scalar::struct_(dtype(), vec![])), "{}");
    }

    #[test]
    fn display_one_field_struct() {
        fn dtype() -> DType {
            DType::Struct(
                StructFields::new(
                    [FieldName::from("foo")].into(),
                    vec![DType::Primitive(PType::U32, Nullable)],
                ),
                Nullable,
            )
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(dtype(), vec![Scalar::null_native::<u32>()])
            ),
            "{foo: null}"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(dtype(), vec![Scalar::from(Some(32_u32))])
            ),
            "{foo: 32u32}"
        );
    }

    #[test]
    fn display_two_field_struct() {
        // fn dtype() -> (DType, DType, DType) {
        let f1 = DType::Bool(Nullable);
        let f2 = DType::Primitive(PType::U32, Nullable);
        let dtype = DType::Struct(
            StructFields::new(
                [FieldName::from("foo"), FieldName::from("bar")].into(),
                vec![f1.clone(), f2.clone()],
            ),
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
                Scalar::struct_(dtype.clone(), vec![Some(true).into(), Scalar::null(f2)])
            ),
            "{foo: true, bar: null}"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(dtype, vec![Some(true).into(), Some(32_u32).into()])
            ),
            "{foo: true, bar: 32u32}"
        );
    }

    #[test]
    fn display_time() {
        fn dtype() -> DType {
            DType::Extension(Time::new(TimeUnit::Seconds, Nullable).erased())
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::new(
                    dtype(),
                    Some(ScalarValue::Primitive(PValue::I32(3 * MINUTES + 25)))
                )
            ),
            "00:03:25"
        );
    }

    #[test]
    fn display_date() {
        fn dtype() -> DType {
            DType::Extension(Date::new(TimeUnit::Days, Nullable).erased())
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::new(dtype(), Some(ScalarValue::Primitive(PValue::I32(25))))
            ),
            "1970-01-26"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::new(dtype(), Some(ScalarValue::Primitive(PValue::I32(365))))
            ),
            "1971-01-01"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::new(dtype(), Some(ScalarValue::Primitive(PValue::I32(365 * 4))))
            ),
            "1973-12-31"
        );
    }

    #[test]
    fn display_variant_values() {
        assert_eq!(
            format!("{}", Scalar::null(DType::Variant(Nullable))),
            "null"
        );
        assert_eq!(
            format!("{}", Scalar::variant(Scalar::null(DType::Null))),
            "variant(null)"
        );
        assert_eq!(
            format!("{}", Scalar::variant(Scalar::from(42_u32))),
            "variant(42u32)"
        );
    }

    #[test]
    fn display_local_timestamp() {
        fn dtype() -> DType {
            DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullable).erased())
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::new(
                    dtype(),
                    Some(ScalarValue::Primitive(PValue::I64(
                        (3 * DAYS + 2 * HOURS + 5 * MINUTES + 10) as i64
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
            DType::Extension(
                Timestamp::new_with_tz(TimeUnit::Seconds, Some("Pacific/Guam".into()), Nullable)
                    .erased(),
            )
        }

        assert_eq!(format!("{}", Scalar::null(dtype())), "null");

        assert_eq!(
            format!(
                "{}",
                Scalar::new(dtype(), Some(ScalarValue::Primitive(PValue::I64(0i64))))
            ),
            "1970-01-01T10:00:00+10:00[Pacific/Guam]"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::new(
                    dtype(),
                    Some(ScalarValue::Primitive(PValue::I64(
                        (3 * DAYS + 2 * HOURS + 5 * MINUTES + 10) as i64
                    )))
                )
            ),
            "1970-01-04T12:05:10+10:00[Pacific/Guam]"
        );
    }
}
