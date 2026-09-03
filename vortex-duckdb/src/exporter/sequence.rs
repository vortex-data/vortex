// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bitvec::macros::internal::funty::Fundamental;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::encodings::sequence::SequenceArray;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;

use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::canonical;

struct SequenceExporter {
    start: i64,
    step: i64,
}

pub(crate) fn new_exporter_with_flatten(
    array: &SequenceArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // DuckDB's sequence vector API takes signed start/step values and rejects unsigned logical
    // types, so unsigned Vortex sequences must be exported as flat primitive vectors.
    if flatten || array.ptype().is_unsigned_int() {
        return canonical::new_exporter(array.clone().into_array(), cache, ctx);
    }
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
        // TODO why don't we apply validity mask here?

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
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter_with_flatten(&arr, &ConversionCache::default(), &mut ctx, false)
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
            String::try_from(&*chunk).unwrap(),
            r#"Chunk - [1 Columns]
- SEQUENCE INTEGER: 4 = [ 2, 7, 12, 17]
"#
        );
    }

    #[test]
    fn test_unsigned_sequence() {
        let arr = Sequence::try_new_typed(2u64, 5u64, Nullability::NonNullable, 100).unwrap();
        let mut chunk = DataChunk::new([LogicalType::uint64()]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter_with_flatten(&arr, &ConversionCache::default(), &mut ctx, false)
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
            String::try_from(&*chunk).unwrap(),
            r#"Chunk - [1 Columns]
- FLAT UBIGINT: 4 = [ 2, 7, 12, 17]
"#
        );
    }
}
