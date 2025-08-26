// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use fsst::{Compressor, Symbol};
use vortex_array::arrays::VarBinViewArray;
use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, ValidityHelper};
use vortex_array::{Canonical, DeserializeMetadata, EmptyMetadata, IntoArray};
use vortex_buffer::{Buffer, BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_bail, vortex_ensure};

use crate::fsst_view::{FSSTViewArray, FSSTViewEncoding, FSSTViewVTable, OutlinedStr, View};
use crate::{FSSTArray, fsst_compress, fsst_train_compressor};

impl SerdeVTable<FSSTViewVTable> for FSSTViewVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_: &FSSTViewArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _: &FSSTViewEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FSSTViewArray> {
        // If the DType is nullable, we need to validate the validity information
        vortex_ensure!(buffers.len() == , );

        vortex_ensure!(
            dtype.is_utf8() || dtype.is_binary(),
            "FSSTViewArray can only be built for utf8 or binary data type, not {dtype}"
        );

        // First buffer are the views.
        let views = buffers[0].clone();
        let views = Buffer::<View>::from_byte_buffer(views);

        vortex_ensure!(
            views.len() == len,
            "FSSTViewArray: views expected to have length {len}, was {}",
            views.len()
        );

        // Second buffer are the symbol table
        let symbols = Buffer::<Symbol>::from_byte_buffer(buffers[1].clone());
        // Third buffer: symbol lengths
        let symbol_lengths = Buffer::<u8>::from_byte_buffer(buffers[2].clone());

        vortex_ensure!(
            symbols.len() <= 254,
            "FSSTViewArray: symbol table too large: {}",
            symbols.len()
        );
        vortex_ensure!(
            symbols.len() == symbol_lengths.len(),
            "FSSTViewArray: symbols (len={}) and symbol_lengths (len={}) misaligned",
            symbols.len(),
            symbol_lengths.len()
        );

        // Fourth buffer is the actual string data. Must be FSST encoded.

        // We should make sure all the external offsets point into the output buffer node.
        // Access a child array of buffers.
        // We can directly bitpack encode the buffers instead.
        let uncompressed_offsets = Buffer::<u32>::from_byte_buffer(buffers[3].clone());
    }
}

impl EncodeVTable<FSSTViewVTable> for FSSTViewVTable {
    // Write into the encoding using the canonical elements.
    fn encode(
        _: &FSSTViewEncoding,
        canonical: &Canonical,
        like: Option<&FSSTViewArray>,
    ) -> VortexResult<Option<FSSTViewArray>> {
        let Canonical::VarBinView(strings) = canonical else {
            vortex_bail!("FSSTViewVTable can only encode from VarBinView")
        };

        // Reuse the compressor from the other array to compress our array.
        let compressor = match like {
            None => Arc::new(fsst_train_compressor(strings.as_ref())?),
            Some(original) => original.compressor.clone(),
        };

        let compressed = fsst_compress(strings, &compressor)?;

        todo!()
    }
}

/// Compress from an iterator of bytestrings using FSST.
fn compress_from_canonical(array: &VarBinViewArray, compressor: &Arc<Compressor>) -> FSSTViewArray {
    // Pre-allocate a reusable buffer for compression
    let mut reuse = Vec::with_capacity(16 * 1024 * 1024);
    let mut codes = ByteBufferMut::with_capacity(16 * 1024 * 1024);
    let mut uncompressed_offsets: BufferMut<u32> = BufferMut::with_capacity(array.views().len());
    let mut compressed_offsets: BufferMut<u32> = BufferMut::with_capacity(array.views().len());

    uncompressed_offsets.push(0);
    compressed_offsets.push(0);

    let mut views = BufferMut::with_capacity(array.views().len());
    let mut index = 0;

    for idx in 0..array.len() {
        let view = array.views()[idx];
        if view.is_inlined() {
            // We only copy the outlined strings
            // Copy the inlined string.
            views.push(View::new_inlined(&view.as_inlined().data));

            continue;
        }

        // Outlined views:
        //  1. Compress the bytes, copy them into the final buffer
        //  2. Implement the bytes
        let view = view.as_view();
        let buffer = array.buffer(view.buffer_index as usize);
        let start = view.offset() as usize;
        let end = start + view.size as usize;

        // TODO(aduffy): handle strings larger than 8MB
        assert!(
            end - start < 8 * 1024 * 1024,
            "FSST cannot handle strings larger than 8MB at this time"
        );

        unsafe { compressor.compress_into(&buffer[start..end], &mut reuse) };

        let begin =
            u32::try_from(codes.len()).vortex_expect("FSST compressed buffer overflowed u32 range");
        let length =
            u32::try_from(reuse.len()).vortex_expect("FSST compressed strings cannot exceed 8MB");
        codes.extend_from_slice(reuse.as_slice());

        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&reuse[..8]);

        views.push(View {
            outline: OutlinedStr {
                len: length,
                index,
                prefix,
            },
        });

        index += 1;

        compressed_offsets
            .push(compressed_offsets.last().copied().unwrap_or(0) + reuse.len() as u32);
        uncompressed_offsets
            .push(uncompressed_offsets.last().copied().unwrap_or(0) + (end - start) as u32);
    }

    let uncompressed_offsets = uncompressed_offsets.freeze().into_array();
    let compressed_offsets = compressed_offsets.freeze().into_array();

    FSSTViewArray {
        views,
        compressor: compressor.clone(),
        validity: array.validity().clone(),
        compressed_offsets,
        uncompressed_offsets,
        dtype: array.dtype().clone(),
        stats_set: Default::default(),
    }
}
