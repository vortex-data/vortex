// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

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
            DType::FixedSizeList(..) => write!(f, "{}", self.as_fixed_size_list()),
            DType::Extension(_) => write!(f, "{}", self.as_extension()),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;
    use vortex_dtype::DType;
    use vortex_dtype::FieldName;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::PType;
    use vortex_dtype::StructFields;
    use vortex_dtype::datetime::Date;
    use vortex_dtype::datetime::Time;
    use vortex_dtype::datetime::TimeUnit;
    use vortex_dtype::datetime::Timestamp;

    use crate::PValue;
    use crate::Scalar;
    use crate::ScalarValue;

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
        let fields = StructFields::new(Default::default(), vec![]);

        assert_eq!(
            format!("{}", Scalar::null(DType::Struct(fields.clone(), Nullable))),
            "null"
        );

        assert_eq!(
            format!("{}", Scalar::struct_(fields, Nullable, vec![])),
            "{}"
        );
    }

    #[test]
    fn display_one_field_struct() {
        let fields = StructFields::new(
            [FieldName::from("foo")].into(),
            vec![DType::Primitive(PType::U32, Nullable)],
        );

        assert_eq!(
            format!("{}", Scalar::null(DType::Struct(fields.clone(), Nullable))),
            "null"
        );

        assert_eq!(
            format!("{}", Scalar::struct_(fields.clone(), Nullable, vec![None])),
            "{foo: null}"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(fields, Nullable, vec![Some(ScalarValue::from(32_u32))])
            ),
            "{foo: 32u32}"
        );
    }

    #[test]
    fn display_two_field_struct() {
        let fields = StructFields::new(
            [FieldName::from("foo"), FieldName::from("bar")].into(),
            vec![
                DType::Bool(Nullable),
                DType::Primitive(PType::U32, Nullable),
            ],
        );

        assert_eq!(
            format!("{}", Scalar::null(DType::Struct(fields.clone(), Nullable))),
            "null"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(fields.clone(), Nullable, vec![None, None])
            ),
            "{foo: null, bar: null}"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(
                    fields.clone(),
                    Nullable,
                    vec![Some(ScalarValue::from(true)), None]
                )
            ),
            "{foo: true, bar: null}"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::struct_(
                    fields,
                    Nullable,
                    vec![
                        Some(ScalarValue::from(true)),
                        Some(ScalarValue::from(32_u32))
                    ]
                )
            ),
            "{foo: true, bar: 32u32}"
        );
    }

    #[test]
    fn display_time() {
        let ext_dtype = Time::new(TimeUnit::Seconds, Nullable);

        assert_eq!(
            format!(
                "{}",
                Scalar::null(DType::Extension(ext_dtype.clone().erased()))
            ),
            "null"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::extension(
                    ext_dtype,
                    ScalarValue::Primitive(PValue::I32(3 * MINUTES + 25))
                )
                .unwrap()
            ),
            "00:03:25"
        );
    }

    #[test]
    fn display_date() {
        let ext_dtype = Date::new(TimeUnit::Days, Nullable);

        assert_eq!(
            format!(
                "{}",
                Scalar::null(DType::Extension(ext_dtype.clone().erased()))
            ),
            "null"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::extension(ext_dtype.clone(), ScalarValue::Primitive(PValue::I32(25)))
                    .unwrap()
            ),
            "1970-01-26"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::extension(ext_dtype.clone(), ScalarValue::Primitive(PValue::I32(365)))
                    .unwrap()
            ),
            "1971-01-01"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::extension(ext_dtype, ScalarValue::Primitive(PValue::I32(365 * 4))).unwrap()
            ),
            "1973-12-31"
        );
    }

    #[test]
    fn display_local_timestamp() {
        let ext_dtype = Timestamp::new(TimeUnit::Seconds, Nullable);

        assert_eq!(
            format!(
                "{}",
                Scalar::null(DType::Extension(ext_dtype.clone().erased()))
            ),
            "null"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::extension(
                    ext_dtype,
                    ScalarValue::Primitive(PValue::I32(3 * DAYS + 2 * HOURS + 5 * MINUTES + 10))
                )
                .unwrap()
            ),
            "1970-01-04T02:05:10Z"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn display_zoned_timestamp() {
        let ext_dtype =
            Timestamp::new_with_tz(TimeUnit::Seconds, Some("Pacific/Guam".into()), Nullable);

        assert_eq!(
            format!(
                "{}",
                Scalar::null(DType::Extension(ext_dtype.clone().erased()))
            ),
            "null"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::extension(ext_dtype.clone(), ScalarValue::Primitive(PValue::I32(0)))
                    .unwrap()
            ),
            "1970-01-01T10:00:00+10:00[Pacific/Guam]"
        );

        assert_eq!(
            format!(
                "{}",
                Scalar::extension(
                    ext_dtype,
                    ScalarValue::Primitive(PValue::I32(3 * DAYS + 2 * HOURS + 5 * MINUTES + 10))
                )
                .unwrap()
            ),
            "1970-01-04T12:05:10+10:00[Pacific/Guam]"
        );
    }
}
