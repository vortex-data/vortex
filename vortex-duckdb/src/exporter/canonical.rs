// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::TemporalArray;
use vortex::array::arrays::variant::VariantArrayExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex_parquet_variant::ParquetVariant;

use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::all_invalid;
use crate::exporter::bool;
use crate::exporter::decimal;
use crate::exporter::fixed_size_list;
use crate::exporter::list_view;
use crate::exporter::primitive;
use crate::exporter::struct_;
use crate::exporter::temporal;
use crate::exporter::varbinview;
use crate::exporter::variant;

pub(crate) fn new_exporter(
    array: ArrayRef,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let encoding_id = array.encoding_id();
    match array.execute::<Canonical>(ctx)? {
        Canonical::Null(_) => Ok(all_invalid::new_exporter()),
        Canonical::Bool(array) => bool::new_exporter(array, ctx),
        Canonical::Primitive(array) => primitive::new_exporter(array, ctx),
        Canonical::Decimal(array) => decimal::new_exporter(array, ctx),
        Canonical::VarBinView(array) => varbinview::new_exporter(array, ctx),
        Canonical::List(array) => list_view::new_exporter(array, cache, ctx),
        Canonical::FixedSizeList(array) => fixed_size_list::new_exporter(array, cache, ctx),
        Canonical::Struct(array) => struct_::new_exporter(array, cache, ctx),
        Canonical::Extension(ext) => {
            if let Ok(temporal_array) = TemporalArray::try_from(ext) {
                return temporal::new_exporter(temporal_array, ctx);
            }
            vortex_bail!("no non-temporal extension exporter")
        }
        Canonical::Variant(array) => {
            let core_storage = array.core_storage().clone();
            let Ok(parquet_variant) = core_storage.execute_until::<ParquetVariant>(ctx) else {
                vortex_bail!(
                    "Variant arrays can't be exported to DuckDB from {encoding_id}: core storage is not ParquetVariant"
                );
            };
            let Ok(parquet_variant) = parquet_variant.try_downcast::<ParquetVariant>() else {
                vortex_bail!(
                    "Variant arrays can't be exported to DuckDB from {encoding_id}: core storage is not ParquetVariant"
                );
            };
            variant::new_exporter(parquet_variant, cache, ctx)
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::Canonical;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::validity::Validity;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::error::vortex_err;
    use vortex_parquet_variant::ParquetVariant;

    use super::*;
    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    #[test]
    fn exports_canonical_variant_backed_by_parquet_variant() -> VortexResult<()> {
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
        let parquet_variant =
            ParquetVariant::try_new(Validity::AllInvalid, metadata, Some(value), None)?;

        let mut ctx = SESSION.create_execution_ctx();
        let Canonical::Variant(variant) = parquet_variant
            .into_array()
            .execute::<Canonical>(&mut ctx)?
        else {
            return Err(vortex_err!("expected canonical Variant"));
        };

        let mut chunk = DataChunk::new([LogicalType::variant()]);
        let cache = ConversionCache::default();
        new_exporter(variant.into_array(), &cache, &mut ctx)?.export(
            0,
            3,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
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
