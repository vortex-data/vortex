use std::sync::Arc;

use vortex_error::{vortex_err, VortexResult};
use vortex_flatbuffers::FlatBuffer;

use crate::field::Field;
use crate::{flatbuffers as fb, DType, StructDType};

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
        Field::Index(i) => Ok(*i),
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
        StructDType::from_iter(struct_dtype).into(),
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
