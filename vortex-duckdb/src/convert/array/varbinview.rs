use std::ffi::c_char;

use duckdb::vtab::arrow::WritableVector;
use itertools::Itertools;
use vortex_array::arrays::{BinaryView, Inlined, VarBinViewArray};
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::ToDuckDB;
use crate::buffer::{
    AssignBufferToVec, FFIDuckDBBuffer, FFIDuckDBBufferInternal, NewCppVectorBuffer,
};

// This is the C++ string view struct
// private:
// 	union {
// 		struct {
// 			uint32_t length;
// 			char prefix[4];
// 			char *ptr;
// 		} pointer;
// 		struct {
// 			uint32_t length;
// 			char inlined[12];
// 		} inlined;
// 	} value;
// };

#[derive(Clone, Copy)]
#[repr(C, align(16))]
// See `BinaryView`
pub union PtrBinaryView {
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
pub struct PtrRef {
    size: u32,
    prefix: [u8; 4],
    ptr: *const c_char,
}

fn binary_view_to_ptr_binary_view(
    view: impl Iterator<Item = BinaryView>,
    buffers: &[ByteBuffer],
    used_buffers: &mut Vec<bool>,
) -> Vec<PtrBinaryView> {
    view.map(|v| {
        if v.is_inlined() {
            PtrBinaryView {
                inlined: v.as_inlined().clone(),
            }
        } else {
            let view = v.as_view();
            used_buffers[view.buffer_index() as usize] = true;
            PtrBinaryView {
                _ref: PtrRef {
                    size: v.len(),
                    prefix: view.prefix().clone(),
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
    .collect_vec()
}

impl ToDuckDB for VarBinViewArray {
    fn to_duckdb(&self, chunk: &mut dyn WritableVector) -> VortexResult<()> {
        let buffers = self.buffers();
        let mut buffer_used = vec![false; buffers.len()];

        let views: Vec<PtrBinaryView> =
            binary_view_to_ptr_binary_view(self.views().iter(), buffers, &mut buffer_used);

        let vec = chunk.flat_vector();
        buffers
            .iter()
            .enumerate()
            .filter_map(|(idx, buf)| {
                if buffer_used[idx] {
                    Some(buf.clone())
                } else {
                    None
                }
            })
            .for_each(|b| {
                // Each buffer is wrapped with a C++ VectorBuffer wrapper which will
                // in turn call `FFIDuckDBBuffer_free` when it is cleaned up in C++ land.
                // Once all ptrs to the bytes are free the bytes can be freed.
                let buffer: *mut FFIDuckDBBuffer = Box::into_raw(Box::new(
                    FFIDuckDBBufferInternal { inner: Box::new(b) }.into(),
                ));
                let extern_buf = unsafe { NewCppVectorBuffer(buffer) };
                // Adds an extra ref to the buffer which will outlive the `views`
                unsafe { AssignBufferToVec(vec.unowned_ptr(), extern_buf) };
            });

        chunk.flat_vector().copy(views.as_slice());

        Ok(())
    }
}
