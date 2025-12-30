// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;

use crate::DType;
use crate::DecimalDType;
use crate::ExtDType;
use crate::ExtID;
use crate::FieldName;
use crate::FieldNames;
use crate::NativeDecimalType;
use crate::Nullability;
use crate::PType;
use crate::StructFields;
use crate::datetime::DATE_ID;
use crate::datetime::TIME_ID;
use crate::datetime::TIMESTAMP_ID;
use crate::datetime::TemporalMetadata;
use crate::datetime::TimeUnit;
use crate::i256;
mod decimal;

impl<'a> Arbitrary<'a> for DType {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        random_dtype(u, 2)
    }
}

impl<'a> Arbitrary<'a> for FieldName {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let i: Arc<str> = Arbitrary::arbitrary(u)?;
        Ok(Self::from(i))
    }
}

fn random_dtype(u: &mut Unstructured<'_>, depth: u8) -> Result<DType> {
    const BASE_TYPE_COUNT: i32 = 5;
    // TODO(joe): update to 3 once fsl works
    const CONTAINER_TYPE_COUNT: i32 = 3;
    let max_dtype_kind = if depth == 0 {
        BASE_TYPE_COUNT
    } else {
        CONTAINER_TYPE_COUNT + BASE_TYPE_COUNT
    };
    Ok(match u.int_in_range(1..=max_dtype_kind)? {
        // base types
        1 => DType::Bool(u.arbitrary()?),
        2 => DType::Primitive(u.arbitrary()?, u.arbitrary()?),
        3 => DType::Decimal(u.arbitrary()?, u.arbitrary()?),
        4 => DType::Utf8(u.arbitrary()?),
        5 => DType::Binary(u.arbitrary()?),

        // container types
        6 => DType::Struct(random_struct_dtype(u, depth - 1)?, u.arbitrary()?),
        7 => DType::List(Arc::new(random_dtype(u, depth - 1)?), u.arbitrary()?),
        8 => DType::Extension(Arc::new(random_ext_dtype(u, depth - 1)?)),

        // 8 => DType::FixedSizeList(
        //     Arc::new(random_dtype(u, depth - 1)?),
        //     // We limit the list size to 3 rather (following random struct fields).
        //     u.choose_index(3)?.try_into().vortex_expect("impossible"),
        //     u.arbitrary()?,
        // ),
        // Null,
        // Extension(ExtDType, Nullability),
        _ => unreachable!("Number out of range"),
    })
}

impl<'a> Arbitrary<'a> for Nullability {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(if u.arbitrary()? {
            Nullability::Nullable
        } else {
            Nullability::NonNullable
        })
    }
}

impl<'a> Arbitrary<'a> for PType {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        Ok(match u.int_in_range(0..=10)? {
            0 => PType::U8,
            1 => PType::U16,
            2 => PType::U32,
            3 => PType::U64,
            4 => PType::I8,
            5 => PType::I16,
            6 => PType::I32,
            7 => PType::I64,
            8 => PType::F16,
            9 => PType::F32,
            10 => PType::F64,
            _ => unreachable!("Number out of range"),
        })
    }
}

impl<'a> Arbitrary<'a> for DecimalDType {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        // Get a random integer for the scale
        let precision = u.int_in_range(1..=i256::MAX_PRECISION)?;
        let scale = u.int_in_range(-i256::MAX_SCALE..=(precision as i8))?;
        Ok(Self::new(precision, scale))
    }
}

impl<'a> Arbitrary<'a> for ExtDType {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        random_ext_dtype(u, 1)
    }
}

fn random_ext_dtype(u: &mut Unstructured<'_>, _depth: u8) -> Result<ExtDType> {
    let choice = u.int_in_range(0..=3)?;

    match choice {
        0 => {
            // DATE: i32 (Days) or i64 (Milliseconds)
            let (ptype, time_unit) = match u.int_in_range(0..=1)? {
                0 => (PType::I32, TimeUnit::Days),
                1 => (PType::I64, TimeUnit::Milliseconds),
                _ => unreachable!(),
            };

            Ok(ExtDType::new(
                DATE_ID.clone(),
                DType::Primitive(ptype, u.arbitrary()?).into(),
                Some(TemporalMetadata::Date(time_unit).into()),
            ))
        }
        1 => {
            // TIME: i32 for Seconds/Milliseconds, i64 for Microseconds/Nanoseconds
            let (ptype, time_unit) = match u.int_in_range(0..=3)? {
                0 => (PType::I32, TimeUnit::Seconds),
                1 => (PType::I32, TimeUnit::Milliseconds),
                2 => (PType::I64, TimeUnit::Microseconds),
                3 => (PType::I64, TimeUnit::Nanoseconds),
                _ => unreachable!(),
            };

            Ok(ExtDType::new(
                TIME_ID.clone(),
                DType::Primitive(ptype, u.arbitrary()?).into(),
                Some(TemporalMetadata::Time(time_unit).into()),
            ))
        }
        2 => {
            // TIMESTAMP: always i64 with time unit and optional timezone
            let time_unit = match u.int_in_range(0..=3)? {
                0 => TimeUnit::Seconds,
                1 => TimeUnit::Milliseconds,
                2 => TimeUnit::Microseconds,
                3 => TimeUnit::Nanoseconds,
                _ => unreachable!(),
            };

            let time_zone = u
                .arbitrary::<bool>()?
                .then(|| {
                    u.choose(&["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"])
                        .map(|s| s.to_string())
                })
                .transpose()?;

            Ok(ExtDType::new(
                TIMESTAMP_ID.clone(),
                DType::Primitive(PType::I64, u.arbitrary()?).into(),
                Some(TemporalMetadata::Timestamp(time_unit, time_zone).into()),
            ))
        }
        3 => {
            // Extension type to store even numbers
            Ok(ExtDType::new(
                ExtID::new("vortex.even".into()),
                DType::Primitive(PType::I64, u.arbitrary()?).into(),
                None,
            ))
        }
        _ => unreachable!(),
    }
}

impl<'a> Arbitrary<'a> for StructFields {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        random_struct_dtype(u, 1)
    }
}

fn random_struct_dtype(u: &mut Unstructured<'_>, depth: u8) -> Result<StructFields> {
    let field_count = u.choose_index(3)?;
    let names: FieldNames = (0..field_count)
        .map(|_| FieldName::arbitrary(u))
        .collect::<Result<FieldNames>>()?;
    let dtypes = (0..names.len())
        .map(|_| random_dtype(u, depth))
        .collect::<Result<Vec<_>>>()?;
    Ok(StructFields::new(names, dtypes))
}
