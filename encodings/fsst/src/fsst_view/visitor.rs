// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::vtable::VisitorVTable;
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor};
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use zstd::zstd_safe::CompressionLevel;

use crate::fsst_view::{FSSTViewArray, FSSTViewVTable};

const ZSTD_COMPRESS_LEVEL: CompressionLevel = 6;

impl VisitorVTable<FSSTViewVTable> for FSSTViewVTable {
    fn visit_buffers(array: &FSSTViewArray, visitor: &mut dyn ArrayBufferVisitor) {
        // Access the Views and FSST buffers.
        let zstd_views = zstd::bulk::compress(
            array.views.clone().into_byte_buffer().as_slice(),
            ZSTD_COMPRESS_LEVEL,
        )
        .vortex_expect("failed to ZSTD compress data");

        visitor.visit_buffer(&ByteBuffer::from(zstd_views));
        visitor.visit_buffer(&array.symbols.clone().into_byte_buffer());
        visitor.visit_buffer(&array.symbol_lengths);
        visitor.visit_buffer(&array.fsst_buffer);
    }

    fn visit_children(array: &FSSTViewArray, visitor: &mut dyn ArrayChildVisitor) {
        // Child arrays: uncompressed offsets, compressed offsets, validity (if present)
        visitor.visit_child("compressed_offsets", &array.compressed_offsets);
        visitor.visit_child("uncompressed_offsets", &array.uncompressed_offsets);
        visitor.visit_validity(&array.validity, array.len());
    }
}
