use std::sync::Arc;

use arbitrary::{Arbitrary, Result, Unstructured};

use crate::{DType, FieldName, FieldNames, Nullability, PType, StructDType};

impl<'a> Arbitrary<'a> for DType {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        random_dtype(u, 2)
    }
}

fn random_dtype(u: &mut Unstructured<'_>, depth: u8) -> Result<DType> {
    let max_dtype_kind = if depth == 0 { 3 } else { 4 };
    Ok(match u.int_in_range(0..=max_dtype_kind)? {
        0 => DType::Bool(u.arbitrary()?),
        1 => DType::Primitive(u.arbitrary()?, u.arbitrary()?),
        2 => DType::Utf8(u.arbitrary()?),
        3 => DType::Binary(u.arbitrary()?),
        4 => DType::Struct(random_struct_dtype(u, depth - 1)?, u.arbitrary()?),
        // Null,
        // List(Arc<DType>, Nullability),
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

impl<'a> Arbitrary<'a> for StructDType {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        random_struct_dtype(u, 1)
    }
}

fn random_struct_dtype(u: &mut Unstructured<'_>, depth: u8) -> Result<StructDType> {
    let field_count = u.choose_index(3)?;
    let names: FieldNames = (0..field_count)
        .map(|_| FieldName::arbitrary(u))
        .collect::<Result<Arc<_>>>()?;
    let dtypes = (0..names.len())
        .map(|_| random_dtype(u, depth))
        .collect::<Result<Vec<_>>>()?;
    Ok(StructDType::new(names, dtypes))
}
