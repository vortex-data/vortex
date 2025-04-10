use std::ffi::c_char;

use duckdb::vtab::arrow::WritableVector;
use itertools::Itertools;
use vortex_array::Array;
use vortex_array::arrays::{BinaryView, Inlined, VarBinViewArray};
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::ToDuckDB;
use crate::buffer::{
    AssignBufferToVec, CppVectorBuffer, ExternalBuffer, FFIDuckDBBufferInternal,
    new_cpp_vector_buffer,
};
use crate::convert::array::ConversionCache;
use crate::convert::array::validity::write_validity_from_mask;
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

fn binary_view_to_ptr_binary_view<'a>(
    view: impl Iterator<Item = &'a BinaryView>,
    buffers: &[ByteBuffer],
    used_buffers: &mut [bool],
) -> Vec<PtrBinaryView> {
    view.map(|v| {
        if v.is_inlined() {
            PtrBinaryView {
                inlined: *v.as_inlined(),
            }
        } else {
            let view = v.as_view();
            used_buffers[view.buffer_index() as usize] = true;
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
    .collect_vec()
}

impl ToDuckDB for VarBinViewArray {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        _: &mut ConversionCache,
    ) -> VortexResult<()> {
        let buffers = self.buffers();
        let mut buffer_used = vec![false; buffers.len()];

        let views: Vec<PtrBinaryView> = binary_view_to_ptr_binary_view(
            self.views().iter(),
            buffers,
            buffer_used.as_mut_slice(),
        );

        let vec = chunk.flat_vector();
        buffers
            .iter()
            .enumerate()
            .filter(|&(idx, _buf)| buffer_used[idx])
            .map(|(_idx, buf)| buf.clone())
            .for_each(|b| {
                // Each buffer is wrapped with a C++ VectorBuffer wrapper which will
                // in turn call `FFIDuckDBBuffer_free` when it is cleaned up in C++ land.
                // Once all ptrs to the bytes are free the bytes can be freed.
                let buffer: *mut ExternalBuffer = Box::into_raw(Box::new(
                    FFIDuckDBBufferInternal { inner: Box::new(b) }.into(),
                ));
                let extern_buf: *mut CppVectorBuffer = unsafe { new_cpp_vector_buffer(buffer) };
                // Adds an extra ref to the buffer which will outlive the `views`
                // Note this fn takes ownership of the cpp vector buffer.
                // TODO(joe): create a free_cpp_vector_buffer method, maybe?
                unsafe { AssignBufferToVec(vec.unowned_ptr(), extern_buf) };
            });

        let mut vector = chunk.flat_vector();
        vector.copy(views.as_slice());
        write_validity_from_mask(self.validity_mask()?, &mut vector);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use vortex_array::arrays::{ConstantArray, VarBinViewArray};
    use vortex_array::compute::slice;
    use vortex_array::{Array, ToCanonical};

    use crate::ToDuckDB;
    use crate::convert::array::ConversionCache;
    use crate::convert::array::array_ref::to_duckdb;
    use crate::convert::array::data_chunk_adaptor::DataChunkHandleSlice;

    #[test]
    fn constant_empty_str_array() {
        let len = 100;
        let const_ = ConstantArray::new("", len).to_array();
        let mut chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Varchar)]);
        to_duckdb(
            &const_,
            &mut DataChunkHandleSlice::new(&mut chunk, 0),
            &mut ConversionCache::default(),
        )
        .unwrap();
        chunk.set_len(len);
        chunk.verify();
        assert_eq!(
            format!("{:?}", chunk),
            r#"Chunk - [1 Columns]
- CONSTANT VARCHAR: 100 = [ ]
"#
        );
    }

    #[test]
    fn constant_long_str_array() {
        let len = 100;
        let const_ = ConstantArray::new(
            "long string 100000000000000000000000000000000000000000000000000000000000",
            len,
        )
        .to_array();
        let mut chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Varchar)]);
        to_duckdb(
            &const_,
            &mut DataChunkHandleSlice::new(&mut chunk, 0),
            &mut ConversionCache::default(),
        )
        .unwrap();
        chunk.set_len(len);
        chunk.verify();
        assert_eq!(
            format!("{:?}", chunk),
            r#"Chunk - [1 Columns]
- CONSTANT VARCHAR: 100 = [ long string 100000000000000000000000000000000000000000000000000000000000]
"#
        );
    }

    // This tests the sharing of buffers between data chunk, while dropping these buffers early.
    #[test]
    fn test_multi_buffer_ref() {
        let varbin = VarBinViewArray::from_iter_str(["a", "ab", "abc", "abcd", "abcde"]);
        {
            let start_view = slice(&varbin, 0, 2).unwrap().to_varbinview().unwrap();
            let mut chunk =
                DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Varchar)]);
            chunk.set_len(start_view.len());
            start_view
                .to_duckdb(
                    &mut DataChunkHandleSlice::new(&mut chunk, 0),
                    &mut ConversionCache::default(),
                )
                .unwrap();

            chunk.verify();
            assert_eq!(
                format!("{:?}", chunk),
                r#"Chunk - [1 Columns]
- FLAT VARCHAR: 2 = [ a, ab]
"#
            );
            drop(chunk)
        }
        {
            let end_view = slice(&varbin, 2, 5).unwrap().to_varbinview().unwrap();
            drop(varbin);
            let mut chunk =
                DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Varchar)]);
            chunk.set_len(end_view.len());
            end_view
                .to_duckdb(
                    &mut DataChunkHandleSlice::new(&mut chunk, 0),
                    &mut ConversionCache::default(),
                )
                .unwrap();
            drop(end_view);

            chunk.verify();
            assert_eq!(
                format!("{:?}", chunk),
                r#"Chunk - [1 Columns]
- FLAT VARCHAR: 3 = [ abc, abcd, abcde]
"#
            );

            drop(chunk)
        }
    }
}
