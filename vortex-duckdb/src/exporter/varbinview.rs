// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_char;

use itertools::Itertools;
use vortex::arrays::{BinaryView, Inlined, VarBinViewArray};
use vortex::buffer::{Buffer, ByteBuffer};
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::duckdb::{Vector, VectorBuffer};
use crate::exporter::{ColumnExporter, all_invalid};

struct VarBinViewExporter {
    views: Buffer<BinaryView>,
    buffers: Vec<ByteBuffer>,
    vector_buffers: Vec<VectorBuffer>,
    validity: Mask,
}

pub(crate) fn new_exporter(array: &VarBinViewArray) -> VortexResult<Box<dyn ColumnExporter>> {
    let validity = array.validity_mask();
    if validity.all_false() {
        return Ok(all_invalid::new_exporter(
            array.len(),
            &array.dtype().try_into()?,
        ));
    }

    Ok(Box::new(VarBinViewExporter {
        views: array.views().clone(),
        buffers: array.buffers().to_vec(),
        vector_buffers: array
            .buffers()
            .iter()
            .cloned()
            .map(VectorBuffer::new)
            .collect_vec(),
        validity: array.validity_mask(),
    }))
}

impl ColumnExporter for VarBinViewExporter {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Copy the views into place.
        for (mut_view, view) in unsafe { vector.as_slice_mut::<PtrBinaryView>(len) }
            .iter_mut()
            .zip(to_ptr_binary_view(
                self.views[offset..offset + len].iter(),
                &self.buffers,
            ))
        {
            *mut_view = view;
        }

        // Update the validity mask.
        unsafe { vector.set_validity(&self.validity, offset, len) };

        // We register our buffers zero-copy with DuckDB and re-use them in each vector.
        for buffer in &self.vector_buffers {
            vector.add_string_vector_buffer(buffer);
        }

        Ok(())
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
// See `BinaryView`
union PtrBinaryView {
    // Numeric representation. This is logically `u128`, but we split it into the high and low
    // bits to preserve the alignment.
    le_bytes: [u8; 16],

    // Inlined representation: strings <= 12 bytes
    inlined: Inlined,

    // Reference type: strings > 12 bytes.
    _ref: PtrRef,
}

#[derive(Clone, Copy, Debug)]
#[repr(C, align(8))]
struct PtrRef {
    size: u32,
    prefix: [u8; 4],
    ptr: *const c_char,
}

fn to_ptr_binary_view<'a>(
    view: impl Iterator<Item = &'a BinaryView>,
    buffers: &[ByteBuffer],
) -> impl Iterator<Item = PtrBinaryView> {
    view.map(|v| {
        if v.is_inlined() {
            PtrBinaryView {
                inlined: *v.as_inlined(),
            }
        } else {
            let view = v.as_view();
            PtrBinaryView {
                _ref: PtrRef {
                    size: v.len(),
                    prefix: *view.prefix(),
                    // TODO(joe) verify this.
                    ptr: unsafe {
                        buffers[view.buffer_index() as usize]
                            .as_ptr()
                            .add(view.offset() as usize)
                            .cast()
                    },
                },
            }
        }
    })
}
