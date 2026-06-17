// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::bool::BoolArrayExt;
use vortex::buffer::BitBuffer;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::validity;

struct BoolExporter {
    bit_buffer: BitBuffer,
}

pub(crate) fn new_exporter(
    array: BoolArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let len = array.len();
    let bits = array.to_bit_buffer();

    let validity = array.validity()?;
    if validity.definitely_all_null() {
        return Ok(all_invalid::new_exporter());
    }
    let validity = validity.to_array(len).execute::<Mask>(ctx)?;

    Ok(validity::new_exporter(
        validity,
        Box::new(BoolExporter { bit_buffer: bits }),
    ))
}

impl ColumnExporter for BoolExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // DuckDB uses byte bools, not bit bools.
        // maybe we can convert into these from a compressed array sometimes?.
        unsafe { vector.as_slice_mut(len) }.copy_from_slice(
            &self
                .bit_buffer
                .slice(offset..(offset + len))
                .iter()
                .collect::<Vec<bool>>(),
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use vortex_array::VortexSessionExecute;

    use super::*;
    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    #[test]
    fn test_bool() {
        let arr = BoolArray::from_iter([true, false, true]);
        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_BOOLEAN)]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter(arr, &mut ctx)
            .unwrap()
            .export(1, 2, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT BOOLEAN: 2 = [ false, true]
"#
        );
    }

    #[test]
    fn test_bool_long() {
        let arr = BoolArray::from_iter([true; 128]);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_BOOLEAN)]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter(arr, &mut ctx)
            .unwrap()
            .export(1, 66, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(65);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            format!(
                r#"Chunk - [1 Columns]
- FLAT BOOLEAN: 65 = [ {}]
"#,
                iter::repeat_n("true", 65).collect::<Vec<&str>>().join(", ")
            )
        );
    }

    #[test]
    fn test_bool_nullable() {
        let arr = BoolArray::from_iter([Some(true), None, Some(false)]);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_BOOLEAN)]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter(arr, &mut ctx)
            .unwrap()
            .export(1, 2, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT BOOLEAN: 2 = [ NULL, false]
"#
        );
    }

    #[test]
    fn test_bool_all_invalid() {
        let arr = BoolArray::from_iter([None; 3]);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_BOOLEAN)]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter(arr, &mut ctx)
            .unwrap()
            .export(1, 2, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- CONSTANT BOOLEAN: 2 = [ NULL]
"#
        );
    }
}
