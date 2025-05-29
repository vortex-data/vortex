use std::ffi::c_char;

use duckdb::ffi::duckdb_assign_buffer_to_vector;
use duckdb::vtab::arrow::WritableVector;
use vortex::arrays::{BinaryView, Inlined, VarBinViewArray};
use vortex::buffer::{Buffer, ByteBuffer};
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::ColumnExporter;
use crate::buffer::new_buffer;
use crate::exporter::FlatVectorExt;

struct VarBinViewExporter {
    views: Buffer<BinaryView>,
    buffers: Vec<ByteBuffer>,
    validity: Mask,
}

pub(crate) fn new_exporter(array: &VarBinViewArray) -> VortexResult<Box<dyn ColumnExporter>> {
    Ok(Box::new(VarBinViewExporter {
        views: array.views().clone(),
        buffers: array.buffers().to_vec(),
        validity: array.validity_mask()?,
    }))
}

impl ColumnExporter for VarBinViewExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
    ) -> VortexResult<()> {
        let mut vector = vector.flat_vector();

        // Copy the views into place.
        for (mut_view, view) in vector
            .as_mut_slice_with_len::<PtrBinaryView>(len)
            .iter_mut()
            .zip(to_ptr_binary_view(
                self.views[offset..offset + len].iter(),
                &self.buffers,
            ))
        {
            *mut_view = view;
        }

        // Update the validity mask.
        vector.set_validity(&self.validity, offset, len);

        // We register our buffers zero-copy with DuckDB and re-use them in each vector.
        for buffer in &self.buffers {
            let duckdb_buffer = new_buffer(buffer.clone());
            unsafe { duckdb_assign_buffer_to_vector(vector.unowned_ptr(), duckdb_buffer) };
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

#[cfg(test)]
mod tests {
    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use vortex::ToCanonical;
    use vortex::arrays::VarBinViewArray;

    use crate::exporter::varbinview::new_exporter;

    // This tests the sharing of buffers between data chunk, while dropping these buffers early.
    #[test]
    fn test_multi_buffer_ref() {
        let varbin = VarBinViewArray::from_iter_str(["a", "ab", "abc", "abcd", "abcde"]);

        let start_view = varbin.slice(0, 2).unwrap().to_varbinview().unwrap();
        let chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Varchar)]);
        chunk.set_len(start_view.len());

        new_exporter(&varbin)
            .unwrap()
            .export(0, start_view.len(), &mut chunk.flat_vector(0))
            .unwrap();
        drop(start_view);

        chunk.verify();
        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [1 Columns]
- FLAT VARCHAR: 2 = [ a, ab]
"#
        );
        drop(chunk);

        let end_view = varbin.slice(2, 5).unwrap().to_varbinview().unwrap();
        drop(varbin);
        let chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Varchar)]);
        chunk.set_len(end_view.len());

        new_exporter(&end_view)
            .unwrap()
            .export(0, end_view.len(), &mut chunk.flat_vector(0))
            .unwrap();
        drop(end_view);

        chunk.verify();
        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [1 Columns]
- FLAT VARCHAR: 3 = [ abc, abcd, abcde]
"#
        );
    }
}
