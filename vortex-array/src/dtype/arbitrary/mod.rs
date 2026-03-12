// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_error::VortexExpect;

use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::StructFields;
use crate::dtype::i256;

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
        8 => DType::FixedSizeList(
            Arc::new(random_dtype(u, depth - 1)?),
            // We limit the list size to 3 rather (following random struct fields).
            u.choose_index(3)?.try_into().vortex_expect("impossible"),
            u.arbitrary()?,
        ),
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
