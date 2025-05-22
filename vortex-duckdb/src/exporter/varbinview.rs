use duckdb::ffi::duckdb_assign_buffer_to_vector;
use duckdb::vtab::arrow::WritableVector;
use vortex_array::arrays::{BinaryView, VarBinViewArray};
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::buffer::new_buffer;
use crate::exporter::FlatVectorExt;
use crate::{ColumnExporter, PtrBinaryView, to_ptr_binary_view};

struct VarBinViewExporter {
    views: Buffer<BinaryView>,
    buffers: Vec<ByteBuffer>,
    validity: Mask,
}

pub(crate) fn new_exporter(array: VarBinViewArray) -> VortexResult<Box<dyn ColumnExporter>> {
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
