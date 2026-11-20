// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Folds a shredded union back into the residual binary value.
//!
//! Parquet Variant storage has no union, so a Variant shredded into one exports with the union
//! as Variant bytes. A member holds the whole value of a row, and the residual holds nothing of
//! it, so the bytes of the member value take the place of the residual of that row.
//!
//! The bytes are written with the metadata of the row, so that the field ids of an object are the
//! ids of its names in that metadata. A member object was shredded from an object of that row, so
//! the metadata has each of its names.

use parquet_variant::ObjectFieldBuilder;
use parquet_variant::VariantBuilder;
use parquet_variant::VariantBuilderExt;
use parquet_variant::VariantMetadata;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use crate::extract::bytes_at;

/// The residual binary value of a node that holds `union`, with `residual` the residual of the
/// rows that the union does not hold, and `metadata` the metadata of each row.
pub(crate) fn fold_union(
    union: ArrayRef,
    residual: Option<ArrayRef>,
    metadata: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = union.len();
    let union = union.execute::<UnionArray>(ctx)?.into_array();
    let metadata = metadata.clone().execute::<VarBinViewArray>(ctx)?;
    let residual = residual
        .map(|residual| {
            let valid = residual.validity()?.execute_mask(len, ctx)?;
            Ok::<_, vortex_error::VortexError>((residual.execute::<VarBinViewArray>(ctx)?, valid))
        })
        .transpose()?;

    let mut value = VarBinViewBuilder::with_capacity_in(
        DType::Binary(Nullability::Nullable),
        len,
        ctx.allocator().clone(),
    );
    for idx in 0..len {
        let member = union.execute_scalar(idx, ctx)?;
        match member.as_union().child() {
            Some(member) => value.append_value(encode(bytes_at(&metadata, idx), &member)?),
            None => match &residual {
                Some((residual, valid)) if valid.value(idx) => {
                    value.append_value(bytes_at(residual, idx));
                }
                _ => value.append_null(),
            },
        }
    }
    Ok(value.finish_into_varbinview().into_array())
}

/// The Variant bytes of `value`, with the field ids of the names in `metadata`.
fn encode(metadata: &[u8], value: &Scalar) -> VortexResult<Vec<u8>> {
    let metadata = VariantMetadata::try_new(metadata)?;
    let mut builder = VariantBuilder::new().with_field_names(metadata.iter());
    append(&mut builder, value)?;

    let (written, value) = builder.finish();
    vortex_ensure_eq!(
        VariantMetadata::try_new(&written)?.len(),
        metadata.len(),
        "a shredded union holds an object field whose name is not in the metadata of its row"
    );
    Ok(value)
}

/// Appends the typed value `value` to `builder`. A null is a value that is missing.
fn append<B: VariantBuilderExt>(builder: &mut B, value: &Scalar) -> VortexResult<()> {
    if value.is_null() {
        builder.append_null();
        return Ok(());
    }

    let missing = || vortex_err!("a valid {} scalar has no value", value.dtype());
    match value.dtype() {
        DType::Bool(_) => builder.append_value(value.as_bool().value().ok_or_else(missing)?),
        DType::Primitive(ptype, _) => {
            let primitive = value.as_primitive();
            match ptype {
                PType::I8 => builder.append_value(primitive.as_::<i8>().ok_or_else(missing)?),
                PType::I16 => builder.append_value(primitive.as_::<i16>().ok_or_else(missing)?),
                PType::I32 => builder.append_value(primitive.as_::<i32>().ok_or_else(missing)?),
                PType::I64 => builder.append_value(primitive.as_::<i64>().ok_or_else(missing)?),
                PType::F32 => builder.append_value(primitive.as_::<f32>().ok_or_else(missing)?),
                PType::F64 => builder.append_value(primitive.as_::<f64>().ok_or_else(missing)?),
                _ => vortex_bail!("a shredded union cannot hold a {ptype} value"),
            }
        }
        DType::Utf8(_) => {
            let string = value.as_utf8().value().ok_or_else(missing)?;
            builder.append_value(string.as_str());
        }
        DType::Binary(_) => {
            let binary = value.as_binary().value().ok_or_else(missing)?;
            builder.append_value(binary.as_slice());
        }
        DType::Struct(..) => {
            let fields = value.as_struct();
            let mut object = builder.try_new_object()?;
            for name in fields.names().iter() {
                let field = fields.field(name.as_ref()).ok_or_else(missing)?;
                append(
                    &mut ObjectFieldBuilder::new(name.as_ref(), &mut object),
                    &field,
                )?;
            }
            object.finish();
        }
        DType::List(..) => {
            let elements = value.as_list().elements().ok_or_else(missing)?;
            let mut list = builder.try_new_list()?;
            for element in &elements {
                append(&mut list, element)?;
            }
            list.finish();
        }
        DType::Union(..) => match value.as_union().child() {
            Some(member) => append(builder, &member)?,
            None => builder.append_null(),
        },
        _ => vortex_bail!("a shredded union cannot hold a {} value", value.dtype()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use parquet_variant::Variant as PqVariant;
    use parquet_variant::VariantBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use super::encode;

    /// An object is written with the field ids of the metadata of its row, so the bytes read
    /// back with that metadata.
    #[test]
    fn encodes_an_object_with_the_ids_of_its_row() -> VortexResult<()> {
        let mut row = VariantBuilder::new();
        row.new_object()
            .with_field("z", 1i64)
            .with_field("a", "x")
            .finish();
        let (metadata, _) = row.finish();

        let object = Scalar::struct_(
            DType::struct_(
                [
                    ("a", DType::Utf8(Nullability::Nullable)),
                    ("z", DType::Primitive(PType::I64, Nullability::Nullable)),
                ],
                Nullability::Nullable,
            ),
            [
                Scalar::utf8("x", Nullability::Nullable),
                Scalar::primitive(1i64, Nullability::Nullable),
            ],
        );
        let value = encode(&metadata, &object)?;

        let variant = PqVariant::try_new(&metadata, &value)?;
        let object = variant.as_object().expect("an object");
        assert_eq!(object.get("a"), Some(PqVariant::from("x")));
        assert_eq!(object.get("z"), Some(PqVariant::from(1i64)));
        Ok(())
    }
}
