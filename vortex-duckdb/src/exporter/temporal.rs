// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::TemporalArray;
use vortex::error::VortexResult;

use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::primitive;

struct TemporalExporter {
    storage_type_exporter: Box<dyn ColumnExporter>,
}

impl ColumnExporter for TemporalExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.storage_type_exporter.export(offset, len, vector, ctx)
    }
}

// TODO(joe): into_parts
pub(crate) fn new_exporter(
    array: TemporalArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    Ok(Box::new(TemporalExporter {
        storage_type_exporter: primitive::new_exporter(
            array
                .temporal_values()
                .clone()
                .execute::<Canonical>(ctx)?
                .into_primitive(),
            ctx,
        )?,
    }))
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray as _;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::TemporalArray;
    use vortex::buffer::buffer;
    use vortex::extension::datetime::TimeUnit;

    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::temporal::new_exporter;

    #[test]
    fn test_timestamp_s() {
        let arr = TemporalArray::new_timestamp(
            buffer![1750265024i64..(1750265024 + 10)].into_array(),
            TimeUnit::Seconds,
            None,
        );
        let mut chunk =
            DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_TIMESTAMP_S)]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter(arr, &mut ctx)
            .unwrap()
            .export(1, 5, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(5);

        assert_eq!(
            String::try_from(&*chunk).unwrap(),
            r#"Chunk - [1 Columns]
- FLAT TIMESTAMP_S: 5 = [ 2025-06-18 16:43:45, 2025-06-18 16:43:46, 2025-06-18 16:43:47, 2025-06-18 16:43:48, 2025-06-18 16:43:49]
"#
        );
    }
}
