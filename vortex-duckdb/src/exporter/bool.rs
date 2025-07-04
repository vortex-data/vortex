// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex::arrays::BoolArray;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::duckdb::Vector;
use crate::exporter::{ColumnExporter, VectorExt};

struct BoolExporter {
    array: BoolArray,
    validity: Mask,
}

pub(crate) fn new_exporter(array: &BoolArray) -> VortexResult<Box<dyn ColumnExporter>> {
    Ok(Box::new(BoolExporter {
        array: array.clone(),
        validity: array.validity_mask()?,
    }))
}

impl ColumnExporter for BoolExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Set validity if necessary.
        if vector.set_validity(&self.validity, offset, len) {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        // DuckDB uses byte bools, not bit bools.
        // maybe we can convert into these from a compressed array sometimes?.
        unsafe { vector.as_slice_mut(len) }.copy_from_slice(
            &self
                .array
                .boolean_buffer()
                .slice(offset, len)
                .iter()
                .collect_vec(),
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::cpp;
    use crate::duckdb::{DataChunk, LogicalType};

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
}
