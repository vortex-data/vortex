// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex::arrays::PrimitiveArray;
use vortex::dtype::{NativePType, match_each_native_ptype};
use vortex::error::VortexResult;

use crate::duckdb::Vector;
use crate::exporter::{ColumnExporter, validity};

struct PrimitiveExporter<T: NativePType> {
    array: PrimitiveArray,
    array_type: PhantomData<T>,
}

pub(crate) fn new_exporter(array: &PrimitiveArray) -> VortexResult<Box<dyn ColumnExporter>> {
    let prim = match_each_native_ptype!(array.ptype(), |T| {
        Box::new(PrimitiveExporter {
            array: array.clone(),
            array_type: PhantomData::<T>,
        }) as Box<dyn ColumnExporter>
    });
    Ok(if array.dtype().is_nullable() {
        validity::new_exporter(array.validity_mask(), prim)
    } else {
        prim
    })
}

impl<T: NativePType> ColumnExporter for PrimitiveExporter<T> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Copy the values from the Vortex array to the DuckDB vector.
        unsafe { vector.as_slice_mut(len) }
            .copy_from_slice(&self.array.as_slice::<T>()[offset..offset + len]);

        Ok(())
    }
}
