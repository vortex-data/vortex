// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array as _;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrow::ArrowExport;
use vortex_array::arrow::ArrowExportVTable;
use vortex_array::arrow::ArrowImport;
use vortex_array::arrow::ArrowImportVTable;
use vortex_array::arrow::ArrowSession;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;

/// Arrow canonical extension name for Parquet Variant storage.
const PARQUET_VARIANT_ARROW_EXTENSION_NAME: &str = "arrow.parquet.variant";
static ARROW_PARQUET_VARIANT: CachedId = CachedId::new(PARQUET_VARIANT_ARROW_EXTENSION_NAME);

impl ArrowExportVTable for ParquetVariant {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_PARQUET_VARIANT
    }

    fn vortex_id(&self) -> Id {
        ParquetVariant.id()
    }

    // The current API doesn't see the array at this point.
    // which is what we actually need to know exactly what the arrow
    // storage type is.
    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        if !dtype.is_variant() {
            return Ok(None);
        }

        Ok(Some(
            Field::new(
                name,
                DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("metadata", DataType::BinaryView, false)),
                    Arc::new(Field::new("value", DataType::BinaryView, false)),
                ])),
                dtype.is_nullable(),
            )
            .with_metadata(
                [(
                    EXTENSION_TYPE_NAME_KEY.to_string(),
                    PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
                )]
                .into(),
            ),
        ))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        if target
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .is_some_and(|ext| ext != PARQUET_VARIANT_ARROW_EXTENSION_NAME)
            || !array.dtype().is_variant()
        {
            return Ok(ArrowExport::Unsupported(array));
        }

        let executed = array.execute_until::<ParquetVariant>(ctx)?;
        let parquet_array = executed
            .as_opt::<ParquetVariant>()
            .ok_or_else(|| vortex_err!("cannot export Variant without ParquetVariant storage"))?;
        let arrow_variant = parquet_array.to_arrow(ctx)?;
        Ok(ArrowExport::Exported(Arc::new(arrow_variant.into_inner())))
    }
}

impl ArrowImportVTable for ParquetVariant {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_PARQUET_VARIANT
    }

    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        if field
            .metadata()
            .get(EXTENSION_TYPE_NAME_KEY)
            .is_some_and(|ext| ext != PARQUET_VARIANT_ARROW_EXTENSION_NAME)
        {
            return Ok(None);
        }

        Ok(Some(DType::Variant(field.is_nullable().into())))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        field: &Field,
        dtype: &DType,
    ) -> VortexResult<ArrowImport> {
        if !matches!(dtype, DType::Variant(_))
            || field
                .metadata()
                .get(EXTENSION_TYPE_NAME_KEY)
                .is_some_and(|ext| ext != PARQUET_VARIANT_ARROW_EXTENSION_NAME)
            || !matches!(array.data_type(), DataType::Struct(_))
        {
            return Ok(ArrowImport::Unsupported(array));
        }

        let arrow_variant = ArrowVariantArray::try_new(array.as_struct())?;
        let imported = if dtype.is_nullable() {
            ParquetVariant::from_arrow_variant_nullable(&arrow_variant)?
        } else {
            ParquetVariant::from_arrow_variant(&arrow_variant)?
        };
        Ok(ArrowImport::Imported(imported.into_array()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array as _;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::StructArray;
    use arrow_array::cast::AsArray;
    use arrow_schema::Field;
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant::VariantBuilder;
    use parquet_variant_compute::VariantArrayBuilder;
    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrow::ArrowExportVTable;
    use vortex_array::arrow::ArrowSessionExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::PARQUET_VARIANT_ARROW_EXTENSION_NAME;
    use crate::ParquetVariant;

    #[fixture]
    fn session() -> VortexSession {
        let session = VortexSession::empty().with::<ArraySession>();
        crate::initialize(&session);
        session
    }

    fn arrow_variant_storage() -> StructArray {
        let mut builder = VariantArrayBuilder::new(3);
        builder.append_variant(PqVariant::from(42i8));
        builder.append_variant(PqVariant::from(true));
        builder.append_variant(PqVariant::from("vortex"));
        builder.build().into_inner()
    }

    fn arrow_variant_field(storage: &StructArray) -> Field {
        Field::new("variant", storage.data_type().clone(), false).with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        )
    }

    fn assert_struct_arrays_eq(actual: &StructArray, expected: &StructArray) {
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual.column_names(), expected.column_names());
        assert_eq!(actual.fields(), expected.fields());
        assert_eq!(actual.nulls(), expected.nulls());
        for (actual, expected) in actual.columns().iter().zip(expected.columns()) {
            assert_eq!(actual.to_data(), expected.to_data());
        }
    }

    #[rstest]
    fn import_parquet_variant_extension_array(session: VortexSession) -> VortexResult<()> {
        let storage = arrow_variant_storage();
        let field = arrow_variant_field(&storage);
        let imported = session
            .arrow()
            .from_arrow_array(Arc::new(storage) as ArrowArrayRef, &field)?;

        assert_eq!(imported.dtype(), &DType::Variant(Nullability::NonNullable));
        assert!(imported.as_opt::<ParquetVariant>().is_some());
        Ok(())
    }

    #[rstest]
    fn roundtrip_parquet_variant_extension_array_from_arrow(
        session: VortexSession,
    ) -> VortexResult<()> {
        let storage = arrow_variant_storage();
        let field = arrow_variant_field(&storage);
        let imported = session
            .arrow()
            .from_arrow_array(Arc::new(storage.clone()) as ArrowArrayRef, &field)?;

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(imported, Some(&field), &mut ctx)?;
        let exported = exported.as_struct();

        assert_struct_arrays_eq(exported, &storage);
        Ok(())
    }

    #[rstest]
    fn roundtrip_parquet_variant_extension_array_from_vortex(
        session: VortexSession,
    ) -> VortexResult<()> {
        let rows = [
            VariantBuilder::new().with_value(42i32).finish(),
            VariantBuilder::new().with_value(true).finish(),
            VariantBuilder::new().with_value("vortex").finish(),
        ];
        let metadata =
            VarBinViewArray::from_iter_bin(rows.iter().map(|(metadata, _)| metadata.as_slice()))
                .into_array();
        let value = VarBinViewArray::from_iter_bin(rows.iter().map(|(_, value)| value.as_slice()))
            .into_array();
        let array = ParquetVariant::try_new(Validity::NonNullable, metadata, Some(value), None)?
            .into_array();
        let expected = array.clone();

        let field = ParquetVariant
            .to_arrow_field("variant", array.dtype(), &session.arrow())?
            .ok_or_else(|| vortex_err!("expected ParquetVariant Arrow field"))?;
        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)?;
        let actual = session
            .arrow()
            .from_arrow_array(Arc::clone(&exported), &field)?;

        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[rstest]
    fn roundtrip_shredded_parquet_variant_extension_array_from_vortex(
        session: VortexSession,
    ) -> VortexResult<()> {
        let rows = [
            VariantBuilder::new().with_value(10i32).finish(),
            VariantBuilder::new().with_value(20i32).finish(),
            VariantBuilder::new().with_value(30i32).finish(),
        ];
        let metadata =
            VarBinViewArray::from_iter_bin(rows.iter().map(|(metadata, _)| metadata.as_slice()))
                .into_array();

        let typed_value = buffer![10i32, 20, 30].into_array();
        let array =
            ParquetVariant::try_new(Validity::NonNullable, metadata, None, Some(typed_value))?
                .into_array();
        let expected = array.clone();

        let field = ParquetVariant
            .to_arrow_field("variant", array.dtype(), &session.arrow())?
            .ok_or_else(|| vortex_err!("expected ParquetVariant Arrow field"))?;
        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)?;
        assert_ne!(
            exported.data_type(),
            field.data_type(),
            "The current arrow field isn't fully validated to the full storage type"
        );

        let actual = session.arrow().from_arrow_array(exported, &field)?;

        assert_arrays_eq!(actual, expected);
        Ok(())
    }
}
