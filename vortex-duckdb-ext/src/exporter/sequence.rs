use bitvec::macros::internal::funty::Fundamental;
use vortex::encodings::sequence::SequenceArray;
use vortex::error::{VortexExpect, VortexResult};

use crate::duckdb::Vector;
use crate::exporter::ColumnExporter;

#[allow(dead_code)]
struct SequenceExporter {
    start: i64,
    step: i64,
}

#[allow(dead_code)]
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
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        let offset = offset.as_i64();
        let start = (offset * self.step) + self.start;
        let end = (len.as_i64() * self.step) + start;

        vector.to_sequence(start, end, len.as_u64());
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::cpp;
    use crate::duckdb::{DataChunk, LogicalType};

    #[test]
    fn test_sequence() {
        let arr = SequenceArray::typed_new(2, 5, 100).unwrap();

        // let chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Integer)]);
        let mut chunk =
            DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)].into_iter());
        chunk.set_len(arr.len());

        new_exporter(&arr)
            .unwrap()
            .export(0, 4, &mut chunk.get_vector(0))
            .unwrap();

        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [1 Columns]
- SEQUENCE INTEGER: 4 = [ 2, 7, 12]
"#
        );
    }
}
