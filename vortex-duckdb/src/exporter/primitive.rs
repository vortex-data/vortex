// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex::arrays::PrimitiveArray;
use vortex::dtype::{NativePType, match_each_native_ptype};
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::duckdb::Vector;
use crate::exporter::{ColumnExporter, VectorExt};

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
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Set validity if necessary.
        if vector.set_validity(&self.validity, offset, len) {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        // Copy the values from the Vortex array to the DuckDB vector.
        unsafe { vector.as_slice_mut(len) }
            .copy_from_slice(&self.array.as_slice::<T>()[offset..offset + len]);

        Ok(())
    }
}
