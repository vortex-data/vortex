// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::validity::Validity;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::convert::ToDuckDBScalar;
use crate::duckdb::Value;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::canonical;
use crate::exporter::new_array_exporter;
use crate::exporter::validity;

struct ConstantExporter {
    value: Option<Value>,
}

pub(crate) fn new_exporter_with_validity(
    array: ConstantArray,
    validity: Validity,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    Ok(match validity.execute_mask(array.len(), ctx)? {
        Mask::AllTrue(_) => return new_exporter(array),
        Mask::AllFalse(_) => Box::new(ConstantExporter { value: None }),
        // duckdb cannot have a nullable constant vector
        mask @ Mask::Values(_) => {
            let array = array.into_array().execute::<Canonical>(ctx)?.into_array();
            let exporter = new_array_exporter(array, cache, ctx)?;
            // TODO(joe): we can splat the constant in a specific exporter and save a copy.
            validity::new_exporter_with_mask(mask, exporter)
        }
    })
}

pub(crate) fn new_exporter_with_flatten(
    array: ConstantArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if flatten {
        return canonical::new_exporter(array.into_array(), cache, ctx);
    }
    new_exporter(array)
}

fn new_exporter(array: ConstantArray) -> VortexResult<Box<dyn ColumnExporter>> {
    // If the scalar is null and _not_ of type Null, then we cannot assign a null DuckDB value
    // to a constant vector since DuckDB will complain about a type-mismatch. In these cases,
    // we need to create an all-null flat vector instead.
    let value = if array.scalar().is_null() {
        None
    } else {
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
