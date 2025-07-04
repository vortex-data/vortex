// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::arrays::ConstantArray;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::convert::ToDuckDBScalar;
use crate::duckdb::{Value, Vector};
use crate::exporter::{ColumnExporter, VectorExt};

struct ConstantExporter {
    value: Option<Value>,
}

pub(crate) fn new_exporter(array: &ConstantArray) -> VortexResult<Box<dyn ColumnExporter>> {
    let value = if array.scalar().is_null() && !matches!(array.dtype(), DType::Null) {
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
    fn export(&self, _offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        match self.value.as_ref() {
            None => {
                // TODO(ngates): would be good if DuckDB supported constant null vectors.
                vector.set_validity(&Mask::AllFalse(len), 0, len);
            }
            Some(value) => {
                vector.reference_value(value);
            }
        }

        Ok(())
    }
}
