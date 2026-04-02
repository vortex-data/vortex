// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;

use crate::duckdb::LogicalTypeRef;
use crate::duckdb::Value;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;

struct AllInvalidExporter {
    len: usize,
    null_value: Value,
}

pub(crate) fn new_exporter(len: usize, logical_type: &LogicalTypeRef) -> Box<dyn ColumnExporter> {
    Box::new(AllInvalidExporter {
        len,
        null_value: Value::null(logical_type),
    })
}

impl ColumnExporter for AllInvalidExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure!(
            offset + len <= self.len,
            "invalid exporter: offset + len must be less than or equal to len"
        );

        vector.reference_value(&self.null_value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::PrimitiveArray;

    use super::*;
    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    #[test]
    fn all_null_array() {
        let arr = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]);
        let ltype = LogicalType::int32();

        let mut chunk = DataChunk::new([ltype.clone()]);

        new_exporter(arr.len(), &ltype)
            .export(
                0,
                3,
                chunk.get_vector_mut(0),
                &mut SESSION.create_execution_ctx(),
            )
            .unwrap();
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- CONSTANT INTEGER: 3 = [ NULL]
"#
        );
    }
}
