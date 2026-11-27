// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decode compact Pco pages and scatter their valid values directly into blocks.

use std::ops::Range;

use pco::data_types::Number;
use pco::data_types::NumberType;
use pco::match_number_enum;
use pco::wrapped::ChunkDecompressor;
use pco::wrapped::FileDecompressor;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::block::BlockBuf;
use vortex_array::block::BlockSource;
use vortex_array::block::BlockValidity;
use vortex_array::block::SourceBlock;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::half;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::Pco;
use crate::PcoArray;
use crate::PcoArrayExt;
use crate::array::number_type_from_ptype;
use crate::array::vortex_err_from_pco;

pub(crate) fn pco_source(
    array: ArrayView<'_, Pco>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<Box<dyn BlockSource>>> {
    let unsliced_validity = array.unsliced_validity();
    // Dense Pco already decodes directly into a buffer that its parent can reuse in place.
    if unsliced_validity.definitely_no_nulls() {
        return Ok(None);
    }
    let mask = unsliced_validity.execute_mask(array.unsliced_n_rows(), ctx)?;
    let next_value = mask.valid_counts_for_indices(&[array.slice_start()])[0];
    let validity = mask.slice(array.slice_start()..array.slice_stop());
    if matches!(validity, Mask::AllTrue(_)) {
        return Ok(None);
    }
    let (file, _) =
        FileDecompressor::new(array.metadata.header.as_slice()).map_err(vortex_err_from_pco)?;
    Ok(Some(
        match_number_enum!(number_type_from_ptype(array.dtype().as_ptype()), NumberType<T> => {
            Box::new(Pages::<T> {
                array: array.into_owned(),
                file,
                chunk: None,
                chunk_idx: 0,
                page_in_chunk: 0,
                next_page: 0,
                decoded: Vec::new(),
                range: 0..0,
                validity,
                next_value,
            }) as Box<dyn BlockSource>
        }),
    ))
}

struct Pages<T: Number> {
    array: PcoArray,
    file: FileDecompressor,
    chunk: Option<ChunkDecompressor<T>>,
    chunk_idx: usize,
    page_in_chunk: usize,
    next_page: usize,
    decoded: Vec<T>,
    /// Next position in the stream of non-null values, in unsliced page coordinates.
    next_value: usize,
    range: Range<usize>,
    validity: Mask,
}

impl<T: Number + NativePType> Pages<T> {
    /// Position at the page containing `row`, skipping whole pages before a slice.
    fn seek(&mut self, row: usize) -> VortexResult<()> {
        while row >= self.range.end {
            let meta = self
                .array
                .metadata
                .chunks
                .get(self.chunk_idx)
                .ok_or_else(|| vortex_err!("Pco pages end before requested row {row}"))?;
            let Some(page) = meta.pages.get(self.page_in_chunk) else {
                self.chunk_idx += 1;
                self.page_in_chunk = 0;
                self.chunk = None;
                continue;
            };
            let len = page.n_values as usize;
            self.range = self.range.end..self.range.end + len;
            let page_idx = self.next_page;
            self.page_in_chunk += 1;
            self.next_page += 1;
            if self.range.end <= row {
                continue;
            }
            let chunk = match &mut self.chunk {
                Some(chunk) => chunk,
                slot @ None => slot.insert(
                    self.file
                        .chunk_decompressor::<T, _>(
                            self.array.chunk_metas[self.chunk_idx].as_slice(),
                        )
                        .map_err(vortex_err_from_pco)?
                        .0,
                ),
            };
            let mut page = chunk
                .page_decompressor(self.array.pages[page_idx].as_slice(), len)
                .map_err(vortex_err_from_pco)?;
            self.decoded.resize(len, T::default());
            let progress = page.read(&mut self.decoded).map_err(vortex_err_from_pco)?;
            vortex_ensure_eq!(progress.n_processed, len);
        }
        Ok(())
    }
}

impl<T: Number + NativePType> BlockSource for Pages<T> {
    fn validity(&self) -> &Mask {
        &self.validity
    }

    fn fill<'s>(
        &'s mut self,
        rows: Range<usize>,
        scratch: &mut BlockBuf,
        _validity: &mut BlockValidity<'_>,
    ) -> VortexResult<SourceBlock<'s>> {
        let output = scratch.values_mut::<T>(rows.len());
        output.fill(T::default());
        if let Mask::Values(mask) = &self.validity {
            let bits = mask.bit_buffer().slice(rows.clone());
            let mut indices = bits.set_indices().peekable();
            while indices.peek().is_some() {
                self.seek(self.next_value)?;
                let remaining = &self.decoded[self.next_value - self.range.start..];
                for (&value, index) in remaining.iter().zip(&mut indices) {
                    output[index] = value;
                    self.next_value += 1;
                }
            }
        }
        Ok(SourceBlock::Scratch {
            offset: 0,
            len: rows.len(),
        })
    }
}
