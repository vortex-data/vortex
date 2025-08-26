// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use fsst::Compressor;
use vortex_array::arrays::{BinaryView, Inlined, Ref, VarBinViewArray};
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::validity::Validity;
use vortex_array::vtable::{ArrayVTable, NotSupported, VTable, ValidityVTableFromValidityHelper};
use vortex_array::{vtable, Array, ArrayRef, Canonical, EncodingId, EncodingRef, ToCanonical};
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::{DType, Nullability};
use vortex_error::VortexExpect;

use crate::fsst_view::View;
#[derive(Debug, Copy, Clone)]
pub struct FSSTViewEncoding;

vtable!(FSSTView);

impl VTable for FSSTViewVTable {
    type Array = FSSTViewArray;
    type Encoding = FSSTViewEncoding;
    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;
    type PipelineVTable = NotSupported;

    fn id(_: &Self::Encoding) -> EncodingId {
        FSSTViewEncoding.id()
    }

    fn encoding(_: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(FSSTViewEncoding.as_ref())
    }
}

impl ArrayVTable<FSSTViewVTable> for FSSTViewVTable {
    fn len(array: &FSSTViewArray) -> usize {
        array.views.len()
    }

    fn dtype(array: &FSSTViewArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &FSSTViewArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

#[derive(Clone)]
pub struct FSSTViewArray {
    /// A list of 16-byte views into the FSST buffer
    pub(crate) views: Buffer<View>,
    pub(crate) dtype: DType,
    /// A packed buffer containing FSST-encoded string data without any internal padding
    pub(crate) fsst_buffer: ByteBuffer,
    /// `compressed_offsets[i]` is the offset into `fsst_buffer` where the `i`-th compressed
    /// string starts.
    pub(crate) compressed_offsets: ArrayRef,
    /// Offsets of all the uncompressed strings, in the original order based on the buffer
    /// type instead.
    pub(crate) uncompressed_offsets: ArrayRef,
    /// FSST compressor used to encode/decode the strings in the fsst_buffer
    pub(crate) compressor: Arc<Compressor>,
    /// Validity information, dictating presence of nulls
    pub(crate) validity: Validity,
    pub(crate) stats_set: ArrayStats,
}

impl fmt::Debug for FSSTViewArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FSSTViewArray")
            .field("views_length", &self.views.len())
            .field("fsst_buffer_size", &self.fsst_buffer.len())
            .field("validity", &self.validity)
            .finish()
    }
}

impl FSSTViewArray {
    pub fn bytes_at(&self, index: usize) -> ByteBuffer {
        let view = self.views[index];
        // If view is a pointer to the slice, ignore it
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

            let start = self
                .compressed_offsets
                .scalar_at(buf_index)
                .as_primitive()
                .as_::<u32>()
                .unwrap_or_default() as usize;
            let end = self
                .compressed_offsets
                .scalar_at(buf_index + 1)
                .as_primitive()
                .as_::<u32>()
                .unwrap_or_default() as usize;

            let encoded = self.fsst_buffer.slice(start..end);

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

        let uncompressed_offsets = self
            .uncompressed_offsets
            .to_primitive()
            .vortex_expect("must implement ToCanonical as Primitive")
            .buffer::<u32>();

        // Rebuild the views to point at the decoded data instead.
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

                    // All the uncompressed lengths
                    let mut reference = Ref {
                        size: outlined.len,
                        prefix: [0; 4],
                        buffer_index: 0,
                        offset: uncompressed_offsets[index],
                    };
                    reference.prefix.copy_from_slice(&outlined.prefix[..4]);

                    BinaryView { _ref: reference }
                }
            })
            .collect();

        Canonical::VarBinView(VarBinViewArray::new(
            views,
            Arc::new([buffer]),
            DType::Utf8(Nullability::NonNullable),
            self.validity,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::validity::Validity;
    use vortex_array::IntoArray;
    use vortex_buffer::{BufferMut, ByteBufferMut};

    use crate::fsst_view::array::FSSTViewArray;
    use crate::fsst_view::{OutlinedStr, View};

    #[test]
    fn test_basic() {
        let plaintext: Vec<&[u8]> = vec![
            b"blog.spiraldb.com/1",
            b"blog.spiraldb.com/23",
            b"docs.vortex.dev",
            b"bench.vortex.dev",
        ];
        let compressor = fsst::Compressor::train(&plaintext);

        let mut uncompressed_offsets = BufferMut::with_capacity(1024);
        uncompressed_offsets.push(0);

        let mut compressed_offsets = BufferMut::with_capacity(1024);
        compressed_offsets.push(0);

        // Compress the values, make a new FSSTViewArray from it.
        let mut views = BufferMut::with_capacity(1024);
        let mut buffer = ByteBufferMut::with_capacity(1024);
        for (index, &text) in plaintext.iter().enumerate() {
            uncompressed_offsets.push(uncompressed_offsets.last().unwrap() + text.len() as u32);

            let mut prefix = [0u8; 8];
            prefix.copy_from_slice(&text[0..8]);

            let view = View {
                outline: OutlinedStr {
                    len: u32::try_from(text.len()).unwrap(),
                    prefix,
                    index: u32::try_from(index).unwrap(),
                },
            };
            views.push(view);

            let compressed = compressor.compress(text);
            buffer.extend_from_slice(&compressed);
            compressed_offsets.push(
                compressed_offsets.last().unwrap() + u32::try_from(compressed.len()).unwrap(),
            );
        }

        // Uncompressed offsets.

        let views = views.freeze();
        let uncompressed_offsets = uncompressed_offsets.freeze().into_array();
        let compressed_offsets = compressed_offsets.freeze().into_array();
        let fsst_buffer = buffer.freeze();

        let array = FSSTViewArray {
            views,
            fsst_buffer,
            compressed_offsets,
            uncompressed_offsets,
            compressor: Arc::new(compressor),
            validity: Validity::NonNullable,
        };

        for (idx, &expected) in plaintext.iter().enumerate() {
            let actual = array.bytes_at(idx);
            assert_eq!(actual.as_slice(), expected, "idx: {idx}");
        }
    }
}
