use std::marker::PhantomData;

use duckdb::vtab::arrow::WritableVector;
use vortex::arrays::PrimitiveArray;
use vortex::dtype::{NativePType, match_each_native_ptype};
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::ColumnExporter;
use crate::exporter::FlatVectorExt;

struct PrimitiveExporter<T: NativePType> {
    array: PrimitiveArray,
    array_type: PhantomData<T>,
    validity: Mask,
}

pub(crate) fn new_exporter(array: &PrimitiveArray) -> VortexResult<Box<dyn ColumnExporter>> {
    Ok(match_each_native_ptype!(array.ptype(), |T| {
        Box::new(PrimitiveExporter {
            array: array.clone(),
            array_type: PhantomData::<T>,
            validity: array.validity_mask()?,
        })
    }))
}

impl<T: NativePType> ColumnExporter for PrimitiveExporter<T> {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
    ) -> VortexResult<()> {
        let mut vector = vector.flat_vector();

        // Set validity if necessary.
        if vector.set_validity(&self.validity, offset, len) {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        // Copy the values from the Vortex array to the DuckDB vector.
        vector
            .as_mut_slice_with_len(len)
            .copy_from_slice(&self.array.as_slice::<T>()[offset..offset + len]);

        Ok(())
    }
}
