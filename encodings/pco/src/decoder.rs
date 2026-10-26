// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! File and chunk decoding setup amortized over all selected pages in an operation.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use pco::data_types::Number;
use pco::wrapped::ChunkDecompressor;
use pco::wrapped::FileDecompressor;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::Pco;
use crate::array::vortex_err_from_pco;
use crate::page_index::LogicalPageIndex;
use crate::page_index::PhysicalPage;

pub(crate) struct PageDecoder<T: Number> {
    file: FileDecompressor,
    chunks: BTreeMap<usize, ChunkDecompressor<T>>,
}

impl<T: Number + NativePType> PageDecoder<T> {
    pub fn new(header: &[u8]) -> VortexResult<Self> {
        let (file, _) = FileDecompressor::new(header).map_err(vortex_err_from_pco)?;
        Ok(Self {
            file,
            chunks: BTreeMap::new(),
        })
    }

    pub fn read(
        &mut self,
        chunk: usize,
        chunk_meta: &[u8],
        n_values: usize,
        buffer: &[u8],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        let mut values = BufferMut::<T>::with_capacity_in(n_values, ctx.allocator().clone());
        unsafe {
            values.set_len(n_values);
        }
        self.read_into(chunk, chunk_meta, n_values, buffer, &mut values)?;
        Ok(PrimitiveArray::new(values.freeze(), Validity::NonNullable))
    }
    pub fn read_into(
        &mut self,
        chunk: usize,
        chunk_meta: &[u8],
        n_values: usize,
        buffer: &[u8],
        output: &mut [T],
    ) -> VortexResult<()> {
        let chunk = match self.chunks.entry(chunk) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let (decoder, _) = self
                    .file
                    .chunk_decompressor::<T, _>(chunk_meta)
                    .map_err(vortex_err_from_pco)?;
                entry.insert(decoder)
            }
        };
        let mut decoder = chunk
            .page_decompressor(buffer, n_values)
            .map_err(vortex_err_from_pco)?;
        decoder.read(output).map_err(vortex_err_from_pco)?;
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct PageLookup {
    last: Option<PhysicalPage>,
}

impl PageLookup {
    pub fn resolve(
        &mut self,
        index: &LogicalPageIndex,
        array: ArrayView<'_, Pco>,
        page: usize,
    ) -> VortexResult<(usize, usize)> {
        if !self
            .last
            .as_ref()
            .is_some_and(|last| last.pages.contains(&page))
        {
            let info = index.chunk_for_page(page)?;
            let result = (info.chunk, info.n_values);
            self.last = Some(info);
            return Ok(result);
        }
        let last = self.last.as_ref().vortex_expect("chunk lookup initialized");
        let info = array.metadata.chunks[last.chunk]
            .pages
            .get(page - last.pages.start)
            .ok_or_else(|| vortex_err!("Pco page metadata is missing"))?;
        Ok((last.chunk, info.n_values as usize))
    }
}

/// Operation-owned decoded pages for compact repeated/range selections. The physical
/// descriptor is derived from the chosen mode only when a page is decoded for the first time.
pub(crate) struct DecodedPages<T: Number> {
    decoder: PageDecoder<T>,
    lookup: PageLookup,
    pages: BTreeMap<usize, PrimitiveArray>,
    last: Option<(usize, PrimitiveArray)>,
}

impl<T: Number + NativePType> DecodedPages<T> {
    pub fn new(array: ArrayView<'_, Pco>) -> VortexResult<Self> {
        Ok(Self {
            decoder: PageDecoder::new(array.metadata.header.as_slice())?,
            lookup: PageLookup::default(),
            pages: BTreeMap::new(),
            last: None,
        })
    }
    pub fn last_values(&self, page: usize) -> Option<&PrimitiveArray> {
        self.last
            .as_ref()
            .filter(|(last, _)| *last == page)
            .map(|(_, values)| values)
    }
    pub fn values(
        &mut self,
        view: &LogicalPageIndex,
        array: ArrayView<'_, Pco>,
        page: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<&PrimitiveArray> {
        if !self.last.as_ref().is_some_and(|(last, _)| *last == page) {
            let decoded = if let Some(decoded) = self.pages.remove(&page) {
                decoded
            } else {
                let (chunk_id, n_values) = self.lookup.resolve(view, array, page)?;
                let buffer = array
                    .pages
                    .get(page)
                    .ok_or_else(|| vortex_err!("Pco compressed page is missing"))?;
                let chunk = array
                    .chunk_metas
                    .get(chunk_id)
                    .ok_or_else(|| vortex_err!("Pco chunk metadata is missing"))?;
                self.decoder.read(
                    chunk_id,
                    chunk.as_slice(),
                    n_values,
                    buffer.as_slice(),
                    ctx,
                )?
            };
            if let Some((page, previous)) = self.last.replace((page, decoded)) {
                self.pages.insert(page, previous);
            }
        }
        let entry = &self
            .last
            .as_ref()
            .vortex_expect("decoded page prepared above")
            .1;
        Ok(entry)
    }
}
