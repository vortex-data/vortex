// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array as _;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::StructArray;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use parquet_variant_compute::GetOptions;
use parquet_variant_compute::VariantArray as ArrowVariantArray;
use parquet_variant_compute::unshred_variant;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Variant;
use vortex_array::arrays::variant::VariantArrayExt;
use vortex_array::arrow::ArrowExport;
use vortex_array::arrow::ArrowExportVTable;
use vortex_array::arrow::ArrowImport;
use vortex_array::arrow::ArrowImportVTable;
use vortex_array::arrow::ArrowSession;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::arrow::to_arrow_null_buffer;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;

/// Arrow canonical extension name for Parquet Variant storage.
const PARQUET_VARIANT_ARROW_EXTENSION_NAME: &str = "arrow.parquet.variant";
static ARROW_PARQUET_VARIANT: CachedId = CachedId::new(PARQUET_VARIANT_ARROW_EXTENSION_NAME);

fn parquet_variant_storage_request(fields: &Fields) -> Option<(bool, bool)> {
    let mut has_metadata = false;
    let mut has_value = false;
    let mut has_typed_value = false;

    for field in fields {
        match field.name().as_str() {
            "metadata" if !has_metadata => has_metadata = true,
            "value" if !has_value => has_value = true,
            "typed_value" if !has_typed_value => has_typed_value = true,
            _ => return None,
        }
    }

    (has_metadata && (has_value || has_typed_value)).then_some((has_value, has_typed_value))
}

fn export_storage_to_target<T: ParquetVariantArrayExt>(
    parquet_array: &T,
    target_fields: &Fields,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let mut arrays = Vec::with_capacity(target_fields.len());

    for field in target_fields {
        let child = match field.name().as_str() {
            "metadata" => Some(parquet_array.metadata_array().clone()),
            "value" => parquet_array.value_array().cloned(),
            "typed_value" => parquet_array.typed_value_array().cloned(),
            _ => unreachable!("storage fields were validated before export"),
        };
        let Some(child) = child else {
            vortex_bail!(
                InvalidArgument: "cannot export Parquet Variant storage field '{}' because source has no {} child",
                field.name(),
                field.name()
            );
        };

        arrays.push(ctx.session().clone().arrow().execute_arrow(
            child,
            Some(field.as_ref()),
            ctx,
        )?);
    }

    let nulls = to_arrow_null_buffer(
        ParquetVariantArrayExt::validity(parquet_array),
        parquet_array.as_ref().len(),
        ctx,
    )?;
    Ok(Arc::new(StructArray::try_new(
        target_fields.clone(),
        arrays,
        nulls,
    )?))
}

fn export_unshredded_storage_to_target<T: ParquetVariantArrayExt>(
    parquet_array: &T,
    target_fields: &Fields,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let arrow_variant = parquet_array.to_arrow(ctx)?;
    let unshredded = unshred_variant(&arrow_variant)?;
    let unshredded_array = if parquet_array.as_ref().dtype().is_nullable() {
        ParquetVariant::from_arrow_variant_nullable(&unshredded)?
    } else {
        ParquetVariant::from_arrow_variant(&unshredded)?
    };
    let unshredded_parquet = unshredded_array.as_::<ParquetVariant>();
    export_storage_to_target(&unshredded_parquet, target_fields, ctx)
}

fn parquet_variant_for_export(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let executed = array.execute_until::<ParquetVariant>(ctx)?;
    if executed.is::<ParquetVariant>() {
        return Ok(executed);
    }

    let variant = executed
        .as_opt::<Variant>()
        .ok_or_else(|| vortex_err!("cannot export Variant without ParquetVariant storage"))?;
    let core_storage = variant
        .core_storage()
        .clone()
        .execute_until::<ParquetVariant>(ctx)?;
    let parquet_core = core_storage
        .as_opt::<ParquetVariant>()
        .ok_or_else(|| vortex_err!("cannot export Variant without ParquetVariant core storage"))?;
    let Some(shredded) = variant.shredded() else {
        return Ok(core_storage);
    };

    ParquetVariant::try_new(
        ParquetVariantArrayExt::validity(&parquet_core),
        parquet_core.metadata_array().clone(),
        parquet_core.value_array().cloned(),
        Some(shredded.clone()),
    )
    .map(IntoArray::into_array)
}

impl ArrowExportVTable for ParquetVariant {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_PARQUET_VARIANT
    }

    fn vortex_id(&self) -> Id {
        ParquetVariant.id()
    }

    fn to_arrow_field(
        &self,
        _name: &str,
        _dtype: &DType,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        // Variant field inference is handled by `ArrowSession::to_arrow_field` directly; this
        // plugin hook is only consulted for `DType::Extension`, never for `DType::Variant`.
        Ok(None)
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

        let parquet_array = parquet_variant_for_export(array, ctx)?;
        let parquet_array = parquet_array.as_::<ParquetVariant>();

        if let DataType::Struct(fields) = target.data_type()
            && let Some((request_has_value, request_has_typed_value)) =
                parquet_variant_storage_request(fields)
        {
            let has_value = parquet_array.value_array().is_some();
            let has_typed_value = parquet_array.typed_value_array().is_some();

            if request_has_value && !request_has_typed_value && has_typed_value {
                return Ok(ArrowExport::Exported(export_unshredded_storage_to_target(
                    &parquet_array,
                    fields,
                    ctx,
                )?));
            }
            if has_value && !request_has_value {
                vortex_bail!(
                    InvalidArgument: "cannot export Parquet Variant storage without losing value child"
                );
            }
            if has_typed_value && !request_has_typed_value {
                vortex_bail!(
                    InvalidArgument: "cannot export Parquet Variant storage without losing typed_value child"
                );
            }

            return Ok(ArrowExport::Exported(export_storage_to_target(
                &parquet_array,
                fields,
                ctx,
            )?));
        }

        let arrow_variant = Arc::new(parquet_array.to_arrow(ctx)?.into_inner()) as ArrowArrayRef;

        if arrow_variant.data_type() == target.data_type() {
            Ok(ArrowExport::Exported(arrow_variant))
        } else {
            Ok(ArrowExport::Exported(parquet_variant_compute::variant_get(
                &arrow_variant,
                GetOptions::new().with_as_type(Some(target.clone().into())),
            )?))
        }
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
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant::VariantBuilder;
    use parquet_variant_compute::VariantArrayBuilder;
    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::VariantArray;
    use vortex_array::arrow::ArrowSessionExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
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

    fn assert_variant_scalars_eq(
        actual: &vortex_array::ArrayRef,
        expected: &vortex_array::ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<()> {
        assert_eq!(actual.len(), expected.len());
        let mut actual_ctx = session.create_execution_ctx();
        let mut expected_ctx = session.create_execution_ctx();
        for index in 0..actual.len() {
            assert_eq!(
                actual.execute_scalar(index, &mut actual_ctx)?,
                expected.execute_scalar(index, &mut expected_ctx)?
            );
        }
        Ok(())
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
    fn export_canonical_variant_with_parquet_variant_core_storage(
        session: VortexSession,
    ) -> VortexResult<()> {
        let storage = arrow_variant_storage();
        let field = arrow_variant_field(&storage);
        let core_storage = session
            .arrow()
            .from_arrow_array(Arc::new(storage.clone()) as ArrowArrayRef, &field)?;
        let canonical = VariantArray::try_new(core_storage, None)?.into_array();

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(canonical, Some(&field), &mut ctx)?;
        let exported = exported.as_struct();

        assert_struct_arrays_eq(exported, &storage);
        Ok(())
    }

    #[rstest]
    fn export_canonical_variant_reattaches_shredded_child(
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
        let expected =
            ParquetVariant::try_new(Validity::NonNullable, metadata, None, Some(typed_value))?
                .into_array();

        let mut ctx = session.create_execution_ctx();
        let canonical = expected
            .clone()
            .execute::<VariantArray>(&mut ctx)?
            .into_array();
        let field = Field::new(
            "variant",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("value", DataType::BinaryView, false),
                ]
                .into(),
            ),
            false,
        )
        .with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );

        let exported = session
            .arrow()
            .execute_arrow(canonical, Some(&field), &mut ctx)?;
        assert_eq!(exported.data_type(), field.data_type());

        let actual = session.arrow().from_arrow_array(exported, &field)?;
        assert_variant_scalars_eq(&actual, &expected, &session)
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

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(array.clone(), None, &mut ctx)?;

        let field = Field::new(
            "",
            exported.data_type().clone(),
            array.dtype().is_nullable(),
        )
        .with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );

        let actual = session
            .arrow()
            .from_arrow_array(Arc::clone(&exported), &field)?;

        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[rstest]
    fn export_fully_shredded_to_inferred_storage_unshreds_value(
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
        let expected =
            ParquetVariant::try_new(Validity::NonNullable, metadata, None, Some(typed_value))?
                .into_array();

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(expected.clone(), None, &mut ctx)?;

        assert_eq!(
            exported.data_type(),
            &DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("value", DataType::BinaryView, false),
                ]
                .into()
            )
        );

        let field = Field::new("", exported.data_type().clone(), false).with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );
        let actual = session.arrow().from_arrow_array(exported, &field)?;
        assert_variant_scalars_eq(&actual, &expected, &session)
    }

    #[rstest]
    fn export_partially_shredded_to_metadata_value_preserves_typed_rows(
        session: VortexSession,
    ) -> VortexResult<()> {
        let (metadata0, value0) = VariantBuilder::new().with_value("fallback-0").finish();
        let (metadata1, _value1) = VariantBuilder::new().with_value(20i32).finish();
        let (metadata2, value2) = VariantBuilder::new().with_value("fallback-2").finish();
        let metadata = VarBinViewArray::from_iter_bin([
            metadata0.as_slice(),
            metadata1.as_slice(),
            metadata2.as_slice(),
        ])
        .into_array();
        let value = VarBinViewArray::from_iter_nullable_bin([
            Some(value0.as_slice()),
            None,
            Some(value2.as_slice()),
        ])
        .into_array();
        let typed_value = PrimitiveArray::from_option_iter([None, Some(20i32), None]).into_array();
        let expected = ParquetVariant::try_new(
            Validity::NonNullable,
            metadata,
            Some(value),
            Some(typed_value),
        )?
        .into_array();
        let field = Field::new(
            "variant",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("value", DataType::BinaryView, true),
                ]
                .into(),
            ),
            false,
        )
        .with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );

        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(expected.clone(), Some(&field), &mut ctx)?;
        assert_eq!(exported.data_type(), field.data_type());

        let actual = session.arrow().from_arrow_array(exported, &field)?;
        assert_variant_scalars_eq(&actual, &expected, &session)
    }

    #[rstest]
    fn export_partial_storage_target_rejects_data_loss(session: VortexSession) -> VortexResult<()> {
        let (metadata0, value0) = VariantBuilder::new().with_value("fallback-0").finish();
        let (metadata1, _value1) = VariantBuilder::new().with_value(20i32).finish();
        let metadata = VarBinViewArray::from_iter_bin([metadata0.as_slice(), metadata1.as_slice()])
            .into_array();
        let value =
            VarBinViewArray::from_iter_nullable_bin([Some(value0.as_slice()), None]).into_array();
        let typed_value = PrimitiveArray::from_option_iter([None, Some(20i32)]).into_array();
        let array = ParquetVariant::try_new(
            Validity::NonNullable,
            metadata,
            Some(value),
            Some(typed_value),
        )?
        .into_array();
        let field = Field::new(
            "variant",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::BinaryView, false),
                    Field::new("typed_value", DataType::Int32, true),
                ]
                .into(),
            ),
            false,
        )
        .with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );

        let mut ctx = session.create_execution_ctx();
        let err = session
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("cannot export Parquet Variant storage without losing value child"),
            "unexpected error: {err}"
        );
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

        let field = Field::new(
            "variant",
            DataType::Struct(
                vec![
                    Field::new("metadata", DataType::Binary, false),
                    Field::new("typed_value", DataType::Int32, false),
                ]
                .into(),
            ),
            false,
        )
        .with_metadata(
            [(
                EXTENSION_TYPE_NAME_KEY.to_string(),
                PARQUET_VARIANT_ARROW_EXTENSION_NAME.to_string(),
            )]
            .into(),
        );
        let mut ctx = session.create_execution_ctx();
        let exported = session
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)?;
        assert_eq!(
            exported.data_type(),
            field.data_type(),
            "Parquet Variant export should honor the requested storage schema"
        );

        let actual = session.arrow().from_arrow_array(exported, &field)?;

        assert_arrays_eq!(actual, expected);
        Ok(())
    }
}
