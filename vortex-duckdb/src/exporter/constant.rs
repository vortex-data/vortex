// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::convert::ToDuckDBScalar;
use crate::duckdb::Value;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::new_array_exporter;
use crate::exporter::validity;

struct ConstantExporter {
    value: Option<Value>,
}

pub fn new_exporter_with_mask(
    array: ConstantArray,
    mask: Mask,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if mask.all_false() {
        return Ok(Box::new(ConstantExporter { value: None }));
    }

    // duckdb cannot have a nullable constant vector, so we create primitive vector with validity mask
    if !mask.all_true() {
        // TODO(joe): we can splat the constant in a specific exporter and save a copy.
        return Ok(validity::new_exporter(
            mask,
            new_array_exporter(
                array.into_array().execute::<Canonical>(ctx)?.into_array(),
                cache,
                ctx,
            )?,
        ));
    }

    new_exporter(array)
}

pub(crate) fn new_exporter(array: ConstantArray) -> VortexResult<Box<dyn ColumnExporter>> {
    let value = if array.scalar().is_null() {
        // If the scalar is null and _not_ of type Null, then we cannot assign a null DuckDB value
        // to a constant vector since DuckDB will complain about a type-mismatch. In these cases,
        // we need to create an all-null flat vector instead.
        None
    } else {
        // For null scalars (that are not explicitly of type Null), we cannot return a
        // constant all-null DuckDB vector.
        Some(array.scalar().try_to_duckdb_scalar()?)
    };

    Ok(Box::new(ConstantExporter { value }))
}

impl ColumnExporter for ConstantExporter {
    fn export(
        &self,
        _offset: usize,
        _len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match self.value.as_ref() {
            None => {
                vector.set_all_false_validity();
            }
            Some(value) => {
                vector.reference_value(value);
            }
        }

        Ok(())
    }
}
