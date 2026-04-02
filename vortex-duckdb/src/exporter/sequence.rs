// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bitvec::macros::internal::funty::Fundamental;
use vortex::array::ExecutionCtx;
use vortex::encodings::sequence::SequenceArray;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;

use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;

struct SequenceExporter {
    start: i64,
    step: i64,
}

pub(crate) fn new_exporter(array: &SequenceArray) -> VortexResult<Box<dyn ColumnExporter>> {
    Ok(Box::new(SequenceExporter {
        start: array.base().as_i64().vortex_expect("cannot have null base"),
        step: array
            .multiplier()
            .as_i64()
            .vortex_expect("cannot have null multiplier"),
    }))
}

impl ColumnExporter for SequenceExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let offset = offset.as_i64();
        let start = (offset * self.step) + self.start;

        vector.to_sequence(start, self.step, len.as_u64());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::VortexSessionExecute;
    use vortex::dtype::Nullability;
    use vortex::encodings::sequence::Sequence;

    use super::*;
    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    #[test]
    fn test_sequence() {
        let arr = Sequence::try_new_typed(2, 5, Nullability::NonNullable, 100).unwrap();
        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr)
            .unwrap()
            .export(
                0,
                4,
                chunk.get_vector_mut(0),
                &mut SESSION.create_execution_ctx(),
            )
            .unwrap();
        chunk.set_len(4);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- SEQUENCE INTEGER: 4 = [ 2, 7, 12, 17]
"#
        );
    }
}
