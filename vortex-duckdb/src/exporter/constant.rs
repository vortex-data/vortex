use duckdb::core::Value;
use duckdb::vtab::arrow::WritableVector;
use vortex::arrays::ConstantArray;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::exporter::FlatVectorExt;
use crate::{ColumnExporter, ToDuckDBScalar};

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
    fn export(
        &self,
        _offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
    ) -> VortexResult<()> {
        match self.value.as_ref() {
            None => {
                // TODO(ngates): would be good if DucKDB supported constant null vectors.
                vector
                    .flat_vector()
                    .set_validity(&Mask::AllFalse(len), 0, len);
            }
            Some(value) => {
                vector.flat_vector().assign_to_constant(value);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use vortex::arrays::ConstantArray;
    use vortex::dtype::{DType, Nullability, PType};
    use vortex::scalar::Scalar;

    use super::*;

    #[test]
    fn constant_empty_str_array() {
        let len = 100;
        let const_ = ConstantArray::new("", len);

        let chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Varchar)]);
        chunk.set_len(len);

        new_exporter(&const_)
            .unwrap()
            .export(0, len, &mut chunk.flat_vector(0))
            .unwrap();

        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [1 Columns]
- CONSTANT VARCHAR: 100 = [ ]
"#
        );
    }

    #[test]
    fn constant_long_str_array() {
        let len = 100;
        // Create a string longer than the inlined VarBinView length.
        let const_ = ConstantArray::new(
            "long string 100000000000000000000000000000000000000000000000000000000000",
            len,
        );

        let chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Varchar)]);
        chunk.set_len(len);

        new_exporter(&const_)
            .unwrap()
            .export(0, len, &mut chunk.flat_vector(0))
            .unwrap();

        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [1 Columns]
- CONSTANT VARCHAR: 100 = [ long string 100000000000000000000000000000000000000000000000000000000000]
"#
        );
    }

    #[test]
    fn constant_all_null() {
        let len = 10;
        let const_ = ConstantArray::new(
            Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
            len,
        );

        let chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Integer)]);
        chunk.set_len(len);

        new_exporter(&const_)
            .unwrap()
            .export(0, len, &mut chunk.flat_vector(0))
            .unwrap();

        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 10 = [ NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL]
"#
        );
    }
}
