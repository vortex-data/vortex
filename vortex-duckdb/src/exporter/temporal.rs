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
    use vortex::array::arrays::PrimitiveArray;
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
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT TIMESTAMP_S: 2 = [ 2025-06-18 16:43:45, 2025-06-18 16:43:46]
"#
        );
    }

    #[test]
    fn test_timestamp_us() {
        let arr = TemporalArray::new_timestamp(
            PrimitiveArray::from_iter((0..10).map(|i| 1_000_000 * i + 1750265188000001i64))
                .into_array(),
            TimeUnit::Microseconds,
            None,
        );
        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_TIMESTAMP)]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter(arr, &mut ctx)
            .unwrap()
            .export(1, 5, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(4);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT TIMESTAMP: 4 = [ 2025-06-18 16:46:29.000001, 2025-06-18 16:46:30.000001, 2025-06-18 16:46:31.000001, 2025-06-18 16:46:32.000001]
"#
        );
    }

    #[test]
    fn test_timestamp_time_us() {
        let arr = TemporalArray::new_time(
            PrimitiveArray::from_iter((1i64..10).map(|i| 1_000_000 * i)).into_array(),
            TimeUnit::Microseconds,
        );

        let mut chunk = DataChunk::new([LogicalType::try_from(arr.dtype()).unwrap()]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter(arr, &mut ctx)
            .unwrap()
            .export(1, 5, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();

        chunk.set_len(4);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT TIME: 4 = [ 00:00:02, 00:00:03, 00:00:04, 00:00:05]
"#
        );
    }
}
