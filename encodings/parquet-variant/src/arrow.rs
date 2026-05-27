// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array as _;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
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
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;

/// Arrow canonical extension name for Parquet Variant storage.
pub const PARQUET_VARIANT_ARROW_EXTENSION_NAME: &str = "arrow.parquet.variant";

static ARROW_PARQUET_VARIANT: CachedId = CachedId::new(PARQUET_VARIANT_ARROW_EXTENSION_NAME);

impl ArrowExportVTable for ParquetVariant {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_PARQUET_VARIANT
    }

    fn vortex_ext_id(&self) -> Id {
        ParquetVariant.id()
    }

    fn to_arrow_field(
        &self,
        _name: &str,
        _dtype: &ExtDTypeRef,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
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
            .map(String::as_str)
            != Some(PARQUET_VARIANT_ARROW_EXTENSION_NAME)
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
            .map(String::as_str)
            != Some(PARQUET_VARIANT_ARROW_EXTENSION_NAME)
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
                .map(String::as_str)
                != Some(PARQUET_VARIANT_ARROW_EXTENSION_NAME)
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
