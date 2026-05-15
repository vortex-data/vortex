// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::array::arrays::VariantArray;
use vortex::array::arrays::variant::VariantArrayExt;
use vortex::array::validity::Validity;
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex_parquet_variant::ParquetVariant;
use vortex_parquet_variant::ParquetVariantArray;
use vortex_parquet_variant::ParquetVariantArrayExt;

use crate::convert::ToDuckDBScalar;
use crate::duckdb::LogicalType;
use crate::duckdb::Vector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::all_invalid;
use crate::exporter::new_array_exporter;

struct VariantScalarExporter {
    array: VariantArray,
}

struct ParquetVariantExporter {
    metadata: Box<dyn ColumnExporter>,
    metadata_type: LogicalType,
    value: Option<Box<dyn ColumnExporter>>,
    value_type: LogicalType,
    typed_value: Option<Box<dyn ColumnExporter>>,
    typed_value_type: Option<LogicalType>,
    validity: Mask,
}

pub(crate) fn new_exporter(
    array: VariantArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    match array
        .core_storage()
        .clone()
        .try_downcast::<ParquetVariant>()
    {
        Ok(core_storage) => new_parquet_exporter(core_storage, cache, ctx),
        Err(_) => Ok(Box::new(VariantScalarExporter { array })),
    }
}

pub(crate) fn new_parquet_exporter(
    array: ParquetVariantArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let len = array.len();
    let validity = array.validity()?;
    if matches!(validity, Validity::AllInvalid) {
        return Ok(all_invalid::new_exporter());
    }

    let metadata = array.metadata_array().clone();
    let metadata_type = LogicalType::try_from(metadata.dtype())?;
    let metadata = new_array_exporter(metadata, cache, ctx)?;

    let (value, value_type) = match array.value_array() {
        Some(value) => (
            Some(new_array_exporter(value.clone(), cache, ctx)?),
            LogicalType::try_from(value.dtype())?,
        ),
        None => (None, LogicalType::blob()),
    };

    let (typed_value, typed_value_type) = match array.typed_value_array() {
        Some(typed_value) => (
            Some(new_array_exporter(typed_value.clone(), cache, ctx)?),
            Some(LogicalType::try_from(typed_value.dtype())?),
        ),
        None => (None, None),
    };

    let validity = validity.to_array(len).execute::<Mask>(ctx)?;
    Ok(Box::new(ParquetVariantExporter {
        metadata,
        metadata_type,
        value,
        value_type,
        typed_value,
        typed_value_type,
        validity,
    }))
}

impl ColumnExporter for VariantScalarExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        for row in 0..len {
            let scalar = self.array.as_ref().execute_scalar(offset + row, ctx)?;
            let value = scalar.try_to_duckdb_scalar()?;
            vector.set_value(row, &value)?;
        }
        Ok(())
    }
}

impl ColumnExporter for ParquetVariantExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mut metadata = Vector::with_capacity(&self.metadata_type, len);
        self.metadata.export(offset, len, &mut metadata, ctx)?;

        let mut value = Vector::with_capacity(&self.value_type, len);
        if let Some(value_exporter) = &self.value {
            value_exporter.export(offset, len, &mut value, ctx)?;
            apply_outer_nulls(&mut value, &self.validity, offset, len);
        } else {
            value.set_all_false_validity();
        }

        let mut typed_value = self
            .typed_value_type
            .as_ref()
            .map(|logical_type| Vector::with_capacity(logical_type, len));
        if let (Some(exporter), Some(vector)) = (&self.typed_value, typed_value.as_mut()) {
            exporter.export(offset, len, vector, ctx)?;
            apply_outer_nulls(vector, &self.validity, offset, len);
        }

        vector.parquet_variant_to_variant(
            &metadata,
            &value,
            typed_value.as_ref().map(|typed_value| &**typed_value),
            len,
        )
    }
}

fn apply_outer_nulls(vector: &mut VectorRef, mask: &Mask, offset: usize, len: usize) -> bool {
    match mask {
        Mask::AllTrue(_) => false,
        Mask::AllFalse(_) => {
            vector.set_all_false_validity();
            true
        }
        Mask::Values(values) => {
            let validity = values.bit_buffer().slice(offset..offset + len);
            let true_count = validity.true_count();
            if true_count == len {
                return false;
            }
            if true_count == 0 {
                vector.set_all_false_validity();
                return true;
            }

            vector.flatten(len as u64);
            let vector_validity = unsafe { vector.ensure_validity_bitslice(len) };
            for (row, is_valid) in validity.iter().enumerate() {
                if !is_valid {
                    vector_validity.set(row, false);
                }
            }
            false
        }
    }
}
