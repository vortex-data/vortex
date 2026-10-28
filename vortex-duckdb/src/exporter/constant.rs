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

#[cfg(test)]
mod tests {
    use vortex::array::VortexSessionExecute;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::scalar::Scalar;

    use super::*;
    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    #[test]
    fn test_export_constant_fixed_size_list() -> VortexResult<()> {
        let scalar = Scalar::fixed_size_list(
            DType::Primitive(PType::I32, Nullability::NonNullable),
            vec![Scalar::from(1i32), Scalar::from(2i32), Scalar::from(3i32)],
            Nullability::NonNullable,
        );
        let mut chunk = DataChunk::new([LogicalType::new_array(
            cpp::DUCKDB_TYPE::DUCKDB_TYPE_INTEGER,
            3,
        )]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter(ConstantArray::new(scalar, 2))?.export(
            0,
            2,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(2);

        let vector = chunk.get_vector(0);
        let rows = (0..chunk.len())
            .map(|i| vector.get_value(i, chunk.len()).map(|v| v.to_string()))
            .collect::<Option<Vec<_>>>();
        assert_eq!(rows, Some(vec!["[1, 2, 3]".to_string(); 2]));
        Ok(())
    }
}
