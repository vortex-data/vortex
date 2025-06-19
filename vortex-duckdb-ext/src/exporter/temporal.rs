use vortex::ToCanonical;
use vortex::arrays::TemporalArray;
use vortex::error::VortexResult;

use crate::duckdb::Vector;
use crate::exporter::{ColumnExporter, primitive};

struct TemporalExporter {
    storage_type_exporter: Box<dyn ColumnExporter>,
}

pub(crate) fn new_exporter(array: &TemporalArray) -> VortexResult<Box<dyn ColumnExporter>> {
    Ok(Box::new(TemporalExporter {
        storage_type_exporter: primitive::new_exporter(
            &array.temporal_values().clone().to_primitive()?,
        )?,
    }))
}

impl ColumnExporter for TemporalExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        self.storage_type_exporter.export(offset, len, vector)
    }
}

#[cfg(test)]
mod tests {
    use vortex::arrays::{PrimitiveArray, TemporalArray};
    use vortex::dtype::datetime::TimeUnit;

    use crate::cpp;
    use crate::duckdb::{DataChunk, LogicalType};
    use crate::exporter::temporal::new_exporter;

    #[test]
    fn test_timestamp_s() {
        let arr = TemporalArray::new_timestamp(
            PrimitiveArray::from_iter(1750265024i64..(1750265024 + 10)).to_array(),
            TimeUnit::S,
            None,
        );
        let mut chunk =
            DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_TIMESTAMP_S)]);

        new_exporter(&arr)
            .unwrap()
            .export(1, 5, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT TIMESTAMP_S: 2 = [ 2025-06-18 16:43:45, 2025-06-18 16:43:46]
"#
        );
    }

    #[test]
    fn test_timestamp_us() {
        let arr = TemporalArray::new_timestamp(
            PrimitiveArray::from_iter((0..10).map(|i| 1_000_000 * i + 1750265188000001i64))
                .to_array(),
            TimeUnit::Us,
            None,
        );
        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_TIMESTAMP)]);

        new_exporter(&arr)
            .unwrap()
            .export(1, 5, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(4);

        assert_eq!(
            format!("{}", String::try_from(chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT TIMESTAMP: 4 = [ 2025-06-18 16:46:29.000001, 2025-06-18 16:46:30.000001, 2025-06-18 16:46:31.000001, 2025-06-18 16:46:32.000001]
"#
        );
    }

    #[test]
    fn test_timestamp_time_us() {
        let arr = TemporalArray::new_time(
            PrimitiveArray::from_iter((1i64..10).map(|i| 1_000_000 * i)).to_array(),
            TimeUnit::Us,
        );

        let mut chunk = DataChunk::new([LogicalType::try_from(arr.dtype()).unwrap()]);

        new_exporter(&arr)
            .unwrap()
            .export(1, 5, &mut chunk.get_vector(0))
            .unwrap();

        chunk.set_len(4);

        assert_eq!(
            format!("{}", String::try_from(chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT TIME: 4 = [ 00:00:02, 00:00:03, 00:00:04, 00:00:05]
"#
        );
    }
}
