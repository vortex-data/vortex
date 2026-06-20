// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;

use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::canonical;
use crate::exporter::new_array_exporter;

struct ChunkedExporter {
    chunk_offsets: Vec<usize>,
    chunks: Vec<Box<dyn ColumnExporter>>,
}

pub(crate) fn new_exporter_with_flatten(
    array: ChunkedArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if flatten {
        return canonical::new_exporter(array.into_array(), cache, ctx);
    }

    let chunk_offsets = array.chunk_offsets().to_vec();
    let chunks = array
        .chunks()
        .iter()
        .map(|chunk| new_array_exporter(chunk.clone(), cache, ctx))
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(Box::new(ChunkedExporter {
        chunk_offsets,
        chunks,
    }))
}

impl ChunkedExporter {
    fn chunk_index(&self, offset: usize) -> usize {
        self.chunk_offsets
            .partition_point(|&chunk_offset| chunk_offset <= offset)
            .saturating_sub(1)
    }
}

impl ColumnExporter for ChunkedExporter {
    fn preferred_batch_len(&self, offset: usize, max_len: usize) -> usize {
        if max_len == 0 || self.chunks.is_empty() {
            return 0;
        }

        let chunk_idx = self.chunk_index(offset);
        let chunk_start = self.chunk_offsets[chunk_idx];
        let chunk_end = self.chunk_offsets[chunk_idx + 1];
        let len = (chunk_end - offset).min(max_len);
        self.chunks[chunk_idx].preferred_batch_len(offset - chunk_start, len)
    }

    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if len == 0 {
            return Ok(());
        }

        let chunk_idx = self.chunk_index(offset);
        let chunk_start = self.chunk_offsets[chunk_idx];
        let chunk_end = self.chunk_offsets[chunk_idx + 1];
        let offset_in_chunk = offset - chunk_start;
        vortex_ensure!(
            offset + len <= chunk_end,
            "chunked DuckDB export range {offset}..{} crosses chunk boundary at {chunk_end}",
            offset + len
        );

        self.chunks[chunk_idx].export(offset_in_chunk, len, vector, ctx)
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::ChunkedArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::buffer::buffer;
    use vortex::error::VortexResult;

    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::ArrayExporter;
    use crate::exporter::ConversionCache;

    #[test]
    fn chunked_exporter_emits_chunk_aligned_vectors() -> VortexResult<()> {
        let values0 = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
        let chunk0 = DictArray::try_new(buffer![0u8, 1].into_array(), values0)?.into_array();
        let dtype = chunk0.dtype().clone();

        let values1 = VarBinViewArray::from_iter_str(["c", "d", "e"]).into_array();
        let chunk1 = DictArray::try_new(buffer![0u8, 1, 2].into_array(), values1)?.into_array();

        let field = ChunkedArray::try_new(vec![chunk0, chunk1], dtype)?.into_array();
        let array = StructArray::from_fields(&[("field", field)])?;
        let mut exporter = ArrayExporter::try_new(
            &array,
            &ConversionCache::default(),
            SESSION.create_execution_ctx(),
        )?;
        let mut chunk = DataChunk::new([LogicalType::varchar()]);

        assert!(exporter.export(&mut chunk, None, None)?);
        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- DICTIONARY VARCHAR: 2 = [ a, b]
"#
        );

        assert!(exporter.export(&mut chunk, None, None)?);
        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- DICTIONARY VARCHAR: 3 = [ c, d, e]
"#
        );

        assert!(!exporter.export(&mut chunk, None, None)?);
        Ok(())
    }
}
