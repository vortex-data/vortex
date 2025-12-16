// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex::array::ArrayRef;
use vortex::array::VectorExecutor;
use vortex::array::arrays::BoolArray;
use vortex::buffer::BitBuffer;
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex::session::VortexSession;

use crate::LogicalType;
use crate::duckdb::Vector;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;

struct BoolExporter {
    array: BoolArray,
    validity_mask: Mask,
}

pub(crate) fn new_exporter(array: &BoolArray) -> VortexResult<Box<dyn ColumnExporter>> {
    let validity_mask = array.validity_mask();
    if validity_mask.all_false() {
        return Ok(all_invalid::new_exporter(
            array.len(),
            &array.dtype().try_into()?,
        ));
    }
    Ok(Box::new(BoolExporter {
        array: array.clone(),
        validity_mask,
    }))
}

impl ColumnExporter for BoolExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Set validity if necessary.
        if unsafe { vector.set_validity(&self.validity_mask, offset, len) } {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        // DuckDB uses byte bools, not bit bools.
        // maybe we can convert into these from a compressed array sometimes?.
        unsafe { vector.as_slice_mut(len) }.copy_from_slice(
            &self
                .array
                .bit_buffer()
                .slice(offset..(offset + len))
                .iter()
                .collect_vec(),
        );

        Ok(())
    }
}

struct BoolVectorExporter {
    buffer: BitBuffer,
    mask: Mask,
}

pub(crate) fn new_vector_exporter(
    array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let vector = array.execute_vector(session)?.into_bool();
    let (buffer, mask) = vector.into_parts();
    if mask.all_false() {
        return Ok(all_invalid::new_exporter(
            buffer.len(),
            &LogicalType::bool(),
        ));
    }
    Ok(Box::new(BoolVectorExporter { buffer, mask }))
}

impl ColumnExporter for BoolVectorExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Set validity if necessary.
        if unsafe { vector.set_validity(&self.mask, offset, len) } {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        // DuckDB uses byte bools, not bit bools.
        // maybe we can convert into these from a compressed array sometimes?.
        unsafe { vector.as_slice_mut(len) }.copy_from_slice(
            &self
                .buffer
                .slice(offset..(offset + len))
                .iter()
                .collect_vec(),
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;

    use super::*;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    #[test]
    fn test_bool() {
        let arr = BoolArray::from_iter([true, false, true]);
        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_BOOLEAN)]);

        new_exporter(&arr)
            .unwrap()
            .export(1, 2, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT BOOLEAN: 2 = [ false, true]
"#
        );
    }

    #[test]
    fn test_bool_long() {
        let arr = BoolArray::from_iter([true; 128]);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_BOOLEAN)]);

        new_exporter(&arr)
            .unwrap()
            .export(1, 66, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(65);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            format!(
                r#"Chunk - [1 Columns]
- FLAT BOOLEAN: 65 = [ {}]
"#,
                iter::repeat_n("true", 65).join(", ")
            )
        );
    }

    #[test]
    fn test_bool_vector_long() {
        let arr = BoolArray::from_iter([true; 128]);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_BOOLEAN)]);

        new_vector_exporter(arr.into_array(), &VortexSession::default())
            .unwrap()
            .export(1, 66, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(65);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            format!(
                r#"Chunk - [1 Columns]
- FLAT BOOLEAN: 65 = [ {}]
"#,
                iter::repeat_n("true", 65).join(", ")
            )
        );
    }
}
