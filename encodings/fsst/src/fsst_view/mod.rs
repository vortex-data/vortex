// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FSST View encoding, an analog to raw FSST encoding.

use fsst::Compressor;
use vortex_array::arrays::{BinaryView, Inlined, Ref};
use vortex_array::Canonical;
use vortex_buffer::{Buffer, ByteBuffer};

const MAX_INLINE_STR: usize = 12;

#[repr(C, align(8))]
#[derive(Copy, Clone)]
union View {
    inline: InlinedStr,
    outline: OutlinedStr,
}

impl View {
    fn is_inlined(&self) -> bool {
        let inner = unsafe { self.inline };
        inner.len as usize <= MAX_INLINE_STR
    }
}

#[repr(C, align(8))]
#[derive(Debug, Copy, Clone)]
struct InlinedStr {
    /// Uncompressed string length
    len: u32,
    /// Raw string bytes
    bytes: [u8; 12],
}

#[repr(C, align(8))]
#[derive(Debug, Copy, Clone)]
struct OutlinedStr {
    /// Uncompressed string length
    len: u32,
    /// 8 bytes of prefix, more than StringView!
    prefix: [u8; 8],
    /// Index into the buffer
    index: u32,
}

pub struct FSSTViewArray {
    /// A set of 16-byte views into the fsst_buffer
    views: Buffer<View>,
    /// A packed buffer containing FSST-encoded string data without any internal padding
    fsst_buffer: ByteBuffer,
    /// Offset of the beginning of the n-th encoded string in the fsst_buffer
    compressed_offsets: Buffer<u32>,
    /// Offsets of all the uncompressed strings, in the original order based on the buffer
    /// type instead.
    uncompressed_offsets: Buffer<u32>,
    /// FSST compressor used to encode/decode the strings in the fsst_buffer
    compressor: Compressor,
}

impl FSSTViewArray {
    pub fn bytes_at(&self, index: usize) -> ByteBuffer {
        // If the value is a slice, we want to convert using the offset instead here.
        let view = self.views[index];
        if view.is_inlined() {
            let inlined = unsafe { view.inline };
            let len = inlined.len as usize;

            let start = index * size_of::<View>() + 4;
            let end = start + len;
            // Return a handle to bytes pointing into the `views` buffer
            self.views.clone().into_byte_buffer().slice(start..end)
        } else {
            // Return a ByteBuffer wrapping the vector
            let outline = unsafe { view.outline };
            let buf_index = outline.index as usize;

            let encoded = self.fsst_buffer.slice(
                self.compressed_offsets[buf_index] as usize
                    ..self.compressed_offsets[buf_index + 1] as usize,
            );

            let result = self
                .compressor
                .decompressor()
                .decompress(encoded.as_slice());

            ByteBuffer::from(result)
        }
    }
}

impl FSSTViewArray {
    pub fn into_canonical(self) -> Canonical {
        let decoder = self.compressor.decompressor();

        let buffer: ByteBuffer = decoder.decompress(self.fsst_buffer.as_slice()).into();
        // Rebuild the views, to point at all of our views instead.
        let views: Buffer<BinaryView> = self
            .views
            .into_iter()
            .map(|view| {
                if view.is_inlined() {
                    let inlined = unsafe { view.inline };
                    // Propagate the inlining directly
                    BinaryView {
                        inlined: Inlined {
                            size: inlined.len,
                            data: inlined.bytes,
                        },
                    }
                } else {
                    let outlined = unsafe { view.outline };
                    let index = outlined.index as usize;

                    // Start and end of the offset instead here.
                    // The trouble with this is that we don't know apriori that the buffer is
                    // packed, and generally we really want to encode things

                    // All of the uncompressed lengths
                    let mut reference = Ref {
                        size: outlined.len,
                        prefix: [0; 4],
                        buffer_index: 0,
                        offset: self.uncompressed_offsets[index],
                    };
                    reference.prefix.copy_from_slice(&outlined.prefix[..4]);

                    BinaryView { _ref: reference }
                }
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_basic() {
        let plaintext: Vec<&[u8]> = vec![
            b"blog.spiraldb.com/1",
            b"blog.spiraldb.com/23",
            b"docs.vortex.dev",
            b"bench.vortex.dev",
        ];
        let compressor = fsst::Compressor::train(&plaintext);

        let mut bytes = Vec::with_capacity(1024);
    }
}
