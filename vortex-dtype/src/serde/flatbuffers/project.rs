// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_flatbuffers::FlatBuffer;

use crate::field::Field;
use crate::{DType, StructFields, flatbuffers as fb};

/// Convert name references in projection list into index references.
///
/// This is mostly useful if you want to deduplicate multiple projections against serialized schema.
pub fn resolve_field<'a, 'b: 'a>(fb: fb::Struct_<'b>, field: &'a Field) -> VortexResult<usize> {
    match field {
        Field::Name(n) => {
            let names = fb
                .names()
                .ok_or_else(|| vortex_err!("Missing field names"))?;
            names
                .iter()
                .position(|name| name == &**n)
                .ok_or_else(|| vortex_err!("Unknown field name {n}"))
        }
        _ => vortex_bail!("Only field names are supported for now"),
    }
}

/// Deserialize single field out of a struct dtype and as a top level dtype
pub fn extract_field(
    fb_dtype: fb::DType<'_>,
    field: &Field,
    buffer: &FlatBuffer,
) -> VortexResult<DType> {
    let fb_struct = fb_dtype
        .type__as_struct_()
        .ok_or_else(|| vortex_err!("The top-level type should be a struct"))?;
    let idx = resolve_field(fb_struct, field)?;
    let (_, dtype) = read_field(fb_struct, idx, buffer)?;
    Ok(dtype)
}

/// Deserialize flatbuffer schema selecting only columns defined by projection
pub fn project_and_deserialize(
    fb_dtype: fb::DType<'_>,
    projection: &[Field],
    buffer: &FlatBuffer,
) -> VortexResult<DType> {
    let fb_struct = fb_dtype
        .type__as_struct_()
        .ok_or_else(|| vortex_err!("The top-level type should be a struct"))?;
    let nullability = fb_struct.nullable().into();

    let struct_dtype = projection
        .iter()
        .map(|f| resolve_field(fb_struct, f))
        .map(|idx| idx.and_then(|i| read_field(fb_struct, i, buffer)))
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(DType::Struct(
        StructFields::from_iter(struct_dtype),
        nullability,
    ))
}

fn read_field(
    fb_struct: fb::Struct_,
    idx: usize,
    buffer: &FlatBuffer,
) -> VortexResult<(Arc<str>, DType)> {
    let name = fb_struct
        .names()
        .ok_or_else(|| vortex_err!("Missing field names"))?
        .get(idx);
    let fb_dtype = fb_struct
        .dtypes()
        .ok_or_else(|| vortex_err!("Missing field dtypes"))?
        .get(idx);

    let dtype = DType::try_from_view(fb_dtype, buffer.clone())?;

    Ok((name.into(), dtype))
}

#[cfg(test)]
mod tests {
    use vortex_flatbuffers::WriteFlatBufferExt;

    use super::*;
    use crate::{DType, FieldName, Nullability, PType, StructFields};

    fn create_test_struct_dtype() -> DType {
        DType::Struct(
            StructFields::from_iter([
                ("id", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("name", DType::Utf8(Nullability::Nullable)),
                ("age", DType::Primitive(PType::I32, Nullability::Nullable)),
                ("email", DType::Utf8(Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        )
    }

    fn serialize_dtype(dtype: &DType) -> FlatBuffer {
        let bytes = dtype.write_flatbuffer_bytes();
        FlatBuffer::from(bytes)
    }

    #[test]
    fn test_resolve_field_by_name() {
        let dtype = create_test_struct_dtype();
        let buffer = serialize_dtype(&dtype);
        let fb_dtype = fb::root_as_dtype(buffer.as_ref()).unwrap();
        let fb_struct = fb_dtype.type__as_struct_().unwrap();

        // Test resolving existing field
        let field = Field::Name("name".into());
        let idx = resolve_field(fb_struct, &field).unwrap();
        assert_eq!(idx, 1);

        // Test resolving non-existent field
        let field = Field::Name("nonexistent".into());
        let result = resolve_field(fb_struct, &field);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unknown field name")
        );
    }

    #[test]
    fn test_resolve_field_element_type() {
        let dtype = create_test_struct_dtype();
        let buffer = serialize_dtype(&dtype);
        let fb_dtype = fb::root_as_dtype(buffer.as_ref()).unwrap();
        let fb_struct = fb_dtype.type__as_struct_().unwrap();

        // Currently only field names are supported, not ElementType
        let field = Field::ElementType;
        let result = resolve_field(fb_struct, &field);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Only field names are supported")
        );
    }

    #[test]
    fn test_extract_field() {
        let dtype = create_test_struct_dtype();
        let buffer = serialize_dtype(&dtype);
        let fb_dtype = fb::root_as_dtype(buffer.as_ref()).unwrap();

        // Extract "name" field
        let field = Field::Name("name".into());
        let extracted = extract_field(fb_dtype, &field, &buffer).unwrap();
        assert_eq!(extracted, DType::Utf8(Nullability::Nullable));

        // Extract "age" field
        let field = Field::Name("age".into());
        let extracted = extract_field(fb_dtype, &field, &buffer).unwrap();
        assert_eq!(
            extracted,
            DType::Primitive(PType::I32, Nullability::Nullable)
        );
    }

    #[test]
    fn test_extract_field_non_struct() {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let buffer = serialize_dtype(&dtype);
        let fb_dtype = fb::root_as_dtype(buffer.as_ref()).unwrap();

        let field = Field::Name("name".into());
        let result = extract_field(fb_dtype, &field, &buffer);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("should be a struct")
        );
    }

    #[test]
    fn test_project_and_deserialize() {
        let dtype = create_test_struct_dtype();
        let buffer = serialize_dtype(&dtype);
        let fb_dtype = fb::root_as_dtype(buffer.as_ref()).unwrap();

        // Project only "name" and "age" fields
        let projection = vec![Field::Name("name".into()), Field::Name("age".into())];
        let projected = project_and_deserialize(fb_dtype, &projection, &buffer).unwrap();

        // Check the result is a struct with only the projected fields
        if let DType::Struct(fields, nullability) = projected {
            assert_eq!(fields.nfields(), 2);
            assert_eq!(fields.field_name(0).unwrap(), &FieldName::from("name"));
            assert_eq!(fields.field_name(1).unwrap(), &FieldName::from("age"));
            assert_eq!(
                fields.field_by_index(0).unwrap(),
                DType::Utf8(Nullability::Nullable)
            );
            assert_eq!(
                fields.field_by_index(1).unwrap(),
                DType::Primitive(PType::I32, Nullability::Nullable)
            );
            assert_eq!(nullability, Nullability::NonNullable);
        } else {
            unreachable!("Expected Struct dtype");
        }
    }

    #[test]
    fn test_project_with_invalid_field() {
        let dtype = create_test_struct_dtype();
        let buffer = serialize_dtype(&dtype);
        let fb_dtype = fb::root_as_dtype(buffer.as_ref()).unwrap();

        // Project with a non-existent field
        let projection = vec![
            Field::Name("name".into()),
            Field::Name("nonexistent".into()),
        ];
        let result = project_and_deserialize(fb_dtype, &projection, &buffer);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unknown field name")
        );
    }

    #[test]
    fn test_empty_projection() {
        let dtype = create_test_struct_dtype();
        let buffer = serialize_dtype(&dtype);
        let fb_dtype = fb::root_as_dtype(buffer.as_ref()).unwrap();

        // Empty projection should return empty struct
        let projection = vec![];
        let projected = project_and_deserialize(fb_dtype, &projection, &buffer).unwrap();

        if let DType::Struct(fields, _) = projected {
            assert_eq!(fields.nfields(), 0);
        } else {
            unreachable!("Expected Struct dtype");
        }
    }
}
