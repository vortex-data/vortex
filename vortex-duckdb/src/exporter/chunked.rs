// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::error::VortexResult;

use crate::duckdb::SelectionVector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::ConversionCache;
use crate::exporter::DictionaryParts;
use crate::exporter::canonical;
use crate::exporter::new_array_exporter_with_flatten;

/// Exports a chunked array with one exporter per chunk.
///
/// A range inside one chunk goes to that chunk's exporter and keeps its encoding. A range that
/// crosses chunk boundaries is exported as follows:
///
/// - If every chunk in the range is a dictionary over the same values, the range becomes one
///   dictionary vector over those values, with no copy. DuckDB sees the same dictionary as for
///   the single-chunk ranges, so its dictionary caches still apply.
/// - Otherwise, the range is sliced and canonicalized, and exported as a flat vector. Joining
///   different dictionaries would give each range a new dictionary, which DuckDB cannot cache.
struct ChunkedExporter {
    array: ArrayRef,
    chunk_offsets: Vec<usize>,
    chunks: Vec<Box<dyn ColumnExporter>>,
}

/// The part of an export range that one chunk holds.
struct Piece {
    chunk_idx: usize,
    /// The first row of the piece in its chunk.
    chunk_offset: usize,
    /// The first row of the piece in the target vector.
    target_offset: usize,
    len: usize,
}

pub(crate) fn new_exporter_with_flatten(
    array: ChunkedArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let chunk_offsets = array.chunk_offset_values().to_vec();
    let chunks = array
        .chunks()
        .iter()
        .map(|chunk| new_array_exporter_with_flatten(chunk.clone(), cache, ctx, flatten))
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(Box::new(ChunkedExporter {
        array: array.into_array(),
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

    /// The non-empty pieces of the range `offset..offset + len`, in row order.
    fn pieces(&self, offset: usize, len: usize) -> Vec<Piece> {
        let end = offset + len;
        let mut pieces = Vec::new();
        let mut position = offset;
        let mut chunk_idx = self.chunk_index(offset);

        while position < end {
            let chunk_start = self.chunk_offsets[chunk_idx];
            let piece_end = self.chunk_offsets[chunk_idx + 1].min(end);

            if piece_end > position {
                pieces.push(Piece {
                    chunk_idx,
                    chunk_offset: position - chunk_start,
                    target_offset: position - offset,
                    len: piece_end - position,
                });
            }

            position = piece_end;
            chunk_idx += 1;
        }

        pieces
    }

    /// Export `pieces` as one dictionary vector, or return `false` if the chunks are not all
    /// dictionaries over the same values.
    fn export_dictionary(&self, pieces: &[Piece], len: usize, vector: &mut VectorRef) -> bool {
        let Some(dicts) = pieces
            .iter()
            .map(|piece| self.chunks[piece.chunk_idx].as_dictionary())
            .collect::<Option<Vec<&dyn DictionaryParts>>>()
        else {
            return false;
        };

        let first_key = dicts[0].values_key();
        if !dicts.iter().all(|dict| dict.values_key() == first_key) {
            return false;
        }

        let mut sel_vec = SelectionVector::with_capacity(len);
        // SAFETY: the selection vector has capacity for `len` entries, and the pieces write all
        // of them.
        let codes = unsafe { sel_vec.as_slice_mut(len) };
        for (piece, dict) in pieces.iter().zip(&dicts) {
            let target = piece.target_offset..piece.target_offset + piece.len;
            dict.write_codes(piece.chunk_offset, &mut codes[target]);
        }

        vector.reuse_dictionary(dicts[0].values(), &sel_vec, len);
        true
    }
}

impl ColumnExporter for ChunkedExporter {
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

        let pieces = self.pieces(offset, len);
        if let [piece] = pieces.as_slice() {
            return self.chunks[piece.chunk_idx].export(piece.chunk_offset, len, vector, ctx);
        }

        if self.export_dictionary(&pieces, len, vector) {
            return Ok(());
        }

        // Vortex joins canonical chunks without copying string data, so canonicalizing the range
        // costs less than exporting each chunk and copying it into the target. The sliced range
        // is a new array that the scan's cache cannot hold, so it gets its own cache.
        let range = self.array.slice(offset..offset + len)?;
        canonical::new_exporter(range, &ConversionCache::default(), ctx)?
            .export(0, len, vector, ctx)
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
    use crate::exporter::new_array_exporter;

    fn dict_chunks() -> VortexResult<ChunkedArray> {
        let values0 = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
        let chunk0 = DictArray::try_new(buffer![0u8, 1].into_array(), values0)?.into_array();
        let dtype = chunk0.dtype().clone();

        let values1 = VarBinViewArray::from_iter_str(["c", "d", "e"]).into_array();
        let chunk1 = DictArray::try_new(buffer![0u8, 1, 2].into_array(), values1)?.into_array();

        ChunkedArray::try_new(vec![chunk0, chunk1], dtype)
    }

    /// The scan fills each DuckDB vector to its capacity, so a batch can cross chunk boundaries.
    /// The chunks have different dictionaries, so the batch is canonicalized.
    #[test]
    fn chunked_exporter_fills_batch_across_chunks() -> VortexResult<()> {
        let array = StructArray::from_fields(&[("field", dict_chunks()?.into_array())])?;
        let mut exporter = ArrayExporter::try_new(
            &array,
            &ConversionCache::default(),
            SESSION.create_execution_ctx(),
        )?;
        let mut chunk = DataChunk::new([LogicalType::varchar()]);

        assert!(exporter.export(&mut chunk, None)?);
        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- FLAT VARCHAR: 5 = [ a, b, c, d, e]
"#
        );

        assert!(!exporter.export(&mut chunk, None)?);
        Ok(())
    }

    /// A range inside one chunk keeps the encoding of that chunk.
    #[test]
    fn chunked_exporter_keeps_encoding_inside_chunk() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let exporter = new_array_exporter(
            dict_chunks()?.into_array(),
            &ConversionCache::default(),
            &mut ctx,
        )?;
        let mut chunk = DataChunk::new([LogicalType::varchar()]);

        exporter.export(2, 3, chunk.get_vector_mut(0), &mut ctx)?;
        chunk.set_len(3);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- DICTIONARY VARCHAR: 3 = [ c, d, e]
"#
        );
        Ok(())
    }

    /// Chunks over the same values export a range across them as one dictionary over those values.
    ///
    /// Each chunk has at least as many codes as values. A chunk with fewer codes than values is
    /// exported as plain values, not as a dictionary.
    #[test]
    fn chunked_exporter_shares_dictionary_across_chunks() -> VortexResult<()> {
        let values = VarBinViewArray::from_iter_str(["x", "y", "z"]).into_array();
        let chunk0 =
            DictArray::try_new(buffer![2u8, 0, 1].into_array(), values.clone())?.into_array();
        let chunk1 = DictArray::try_new(buffer![1u8, 2, 0].into_array(), values)?.into_array();
        let dtype = chunk0.dtype().clone();
        let array = ChunkedArray::try_new(vec![chunk0, chunk1], dtype)?.into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let exporter = new_array_exporter(array, &ConversionCache::default(), &mut ctx)?;
        let mut chunk = DataChunk::new([LogicalType::varchar()]);

        exporter.export(1, 4, chunk.get_vector_mut(0), &mut ctx)?;
        chunk.set_len(4);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- DICTIONARY VARCHAR: 4 = [ x, y, y, z]
"#
        );
        Ok(())
    }

    /// A range across chunks with different dictionaries is canonicalized and exported as a flat
    /// vector.
    #[test]
    fn chunked_exporter_exports_range_across_chunks() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let exporter = new_array_exporter(
            dict_chunks()?.into_array(),
            &ConversionCache::default(),
            &mut ctx,
        )?;
        let mut chunk = DataChunk::new([LogicalType::varchar()]);

        exporter.export(1, 3, chunk.get_vector_mut(0), &mut ctx)?;
        chunk.set_len(3);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- FLAT VARCHAR: 3 = [ b, c, d]
"#
        );
        Ok(())
    }
}
