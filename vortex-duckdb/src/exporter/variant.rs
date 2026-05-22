// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;

use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::validity::Validity;
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex_parquet_variant::ParquetVariantArray;
use vortex_parquet_variant::ParquetVariantArrayExt;

use crate::cpp;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;
use crate::duckdb::Vector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::all_invalid;

struct VariantExporter {
    validity: Mask,
    metadata: Vector,
    value: Vector,
    typed_value: Option<Vector>,
}

pub(crate) fn new_exporter(
    array: ParquetVariantArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let len = array.len();
    let validity = ParquetVariantArrayExt::validity(&array);
    if matches!(validity, Validity::AllInvalid) {
        return Ok(all_invalid::new_exporter());
    }
    let validity = validity.to_array(len).execute::<Mask>(ctx)?;

    let metadata = export_child(
        array.metadata_array().clone(),
        &LogicalType::blob(),
        len,
        cache,
        ctx,
    )?;
    let value = match array.value_array() {
        Some(value) => export_child(value.clone(), &LogicalType::blob(), len, cache, ctx)?,
        None => all_null_blob_vector(len),
    };
    let typed_value = array
        .typed_value_array()
        .map(|typed_value| {
            let logical_type = LogicalType::try_from(typed_value.dtype())?;
            export_child(typed_value.clone(), &logical_type, len, cache, ctx)
        })
        .transpose()?;

    Ok(Box::new(VariantExporter {
        validity,
        metadata,
        value,
        typed_value,
    }))
}

fn export_child(
    array: ArrayRef,
    logical_type: &LogicalType,
    len: usize,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vector> {
    let mut vector = Vector::with_capacity(logical_type, len);
    super::new_array_exporter_with_flatten(array, cache, ctx, true)?.export(
        0,
        len,
        &mut vector,
        ctx,
    )?;
    Ok(vector)
}

fn all_null_blob_vector(len: usize) -> Vector {
    let logical_type = LogicalType::blob();
    let mut vector = Vector::with_capacity(&logical_type, len);
    vector.reference_value(&Value::null(&logical_type));
    vector
}

impl ColumnExporter for VariantExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if len == 0 {
            return Ok(());
        }

        let range = offset as u64..(offset + len) as u64;
        let metadata = Vector::slice(&self.metadata, range.clone());
        let value = Vector::slice(&self.value, range.clone());
        let typed_value = self
            .typed_value
            .as_ref()
            .map(|typed_value| Vector::slice(typed_value, range));

        let mut err = ptr::null_mut();
        unsafe {
            cpp::duckdb_vx_variant_from_parquet(
                metadata.as_ptr(),
                value.as_ptr(),
                typed_value
                    .as_ref()
                    .map_or(ptr::null_mut(), |typed_value| typed_value.as_ptr()),
                typed_value.is_some(),
                vector.as_ptr(),
                len as _,
                &raw mut err,
            );
        }
        if !err.is_null() {
            return Err(crate::duckdb::ffi_error(err));
        }

        unsafe {
            vector.set_validity(&self.validity, offset, len);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::validity::Validity;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex_parquet_variant::ParquetVariant;

    use super::*;
    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::exporter::ConversionCache;

    #[test]
    fn all_invalid_variant() -> VortexResult<()> {
        let metadata = VarBinViewArray::from_iter(
            [Some(&b"unused"[..]); 3],
            DType::Binary(Nullability::NonNullable),
        )
        .into_array();
        let value = VarBinViewArray::from_iter(
            [Option::<&[u8]>::None; 3],
            DType::Binary(Nullability::Nullable),
        )
        .into_array();
        let array = ParquetVariant::try_new(Validity::AllInvalid, metadata, Some(value), None)?;

        let mut chunk = DataChunk::new([LogicalType::variant()]);
        let mut ctx = SESSION.create_execution_ctx();
        let cache = ConversionCache::default();
        new_exporter(array, &cache, &mut ctx)?.export(0, 3, chunk.get_vector_mut(0), &mut ctx)?;
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- CONSTANT VARIANT: 3 = [ NULL]
"#
        );
        Ok(())
    }
}
