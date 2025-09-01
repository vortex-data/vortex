// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_array::arrays::{BinaryView, Inlined, Ref, VarBinViewArray};
use vortex_array::vtable::CanonicalVTable;
use vortex_array::{Canonical, ToCanonical};
use vortex_buffer::{Buffer, BufferMut, ByteBuffer};
use vortex_dtype::{NativePType, match_each_unsigned_integer_ptype};

use crate::{FSSTViewArray, FSSTViewVTable, View};

impl CanonicalVTable<FSSTViewVTable> for FSSTViewVTable {
    fn canonicalize(array: &FSSTViewArray) -> Canonical {
        let decoder = array.compressor.decompressor();

        let buffer: ByteBuffer = decoder.decompress(array.fsst_buffer.as_slice()).into();

        let uncompressed_offsets = array.uncompressed_offsets.to_primitive();

        // Rebuild the views to point at the decoded data instead.
        let views: Buffer<BinaryView> =
            match_each_unsigned_integer_ptype!(uncompressed_offsets.ptype(), |P| {
                let uncompressed_offsets = uncompressed_offsets.as_slice::<P>();
                build_views::<P>(uncompressed_offsets, array.views().as_slice())
            });

        // SAFETY: handled in build_views constructing valid view pointers
        unsafe {
            Canonical::VarBinView(VarBinViewArray::new_unchecked(
                views,
                Arc::new([buffer]),
                array.dtype.clone(),
                array.validity.clone(),
            ))
        }
    }
}

#[inline(always)]
fn build_views<T: NativePType + AsPrimitive<u32>>(
    uncompressed_offsets: &[T],
    fsst_views: &[View],
) -> Buffer<BinaryView> {
    let mut views = BufferMut::with_capacity(fsst_views.len());

    let iter = fsst_views.iter().map(|view| {
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
                offset: uncompressed_offsets[index].as_(),
            };
            reference.prefix.copy_from_slice(&outlined.prefix[..4]);

            BinaryView { _ref: reference }
        }
    });

    views.extend_trusted(iter);
    views.freeze()
}
