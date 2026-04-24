// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::error::VortexResult;

use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;

struct AllInvalidExporter;

pub(crate) fn new_exporter() -> Box<dyn ColumnExporter> {
    Box::new(AllInvalidExporter {})
}

impl ColumnExporter for AllInvalidExporter {
    fn export(
        &self,
        _offset: usize,
        _len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vector.set_all_false_validity();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::VortexSessionExecute;

    use super::*;
    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    #[test]
    fn all_null_array() {
        let ltype = LogicalType::int32();
        let mut chunk = DataChunk::new([ltype]);

        new_exporter()
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
