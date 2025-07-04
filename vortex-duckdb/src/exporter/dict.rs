// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex::ToCanonical;
use vortex::arrays::PrimitiveArray;
use vortex::dtype::{NativePType, match_each_integer_ptype};
use vortex::encodings::dict::DictArray;
use vortex::error::VortexResult;

use crate::duckdb::{SelectionVector, Vector};
use crate::exporter::cache::ConversionCache;
use crate::exporter::{ColumnExporter, new_array_exporter};

struct DictExporter<I: NativePType> {
    // Store the dictionary values once and export the same dictionary with each codes chunk.
    values_vector: Vector, // NOTE(ngates): not actually flat...
    codes: PrimitiveArray,
    codes_type: PhantomData<I>,
}

pub(crate) fn new_exporter(
    array: &DictArray,
    cache: &mut ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Grab the cache dictionary values.
    let values = array.values();
    let values_key = Arc::as_ptr(values).addr();
    let values_vector = match cache.values_cache.get(&values_key) {
        None => {
            // Create a new DuckDB vector for the values.
            let mut vector = Vector::with_capacity(values.dtype().try_into()?, values.len());
            new_array_exporter(values, cache)?.export(0, values.len(), &mut vector)?;
            let unowned = vector.clone();
            cache
                .values_cache
                .insert(values_key, (values.clone(), vector));
            unowned
        }
        Some((_array, vector)) => vector.clone(),
    };

    let codes = array.codes().to_primitive()?;
    match_each_integer_ptype!(codes.ptype(), |I| {
        Ok(Box::new(DictExporter {
            values_vector,
            codes,
            codes_type: PhantomData::<I>,
        }))
    })
}

impl<I: NativePType + AsPrimitive<u32>> ColumnExporter for DictExporter<I> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Copy across the dictionary values.
        vector.reference(&self.values_vector);

        // Slice with a selection vector from the codes.
        let mut sel_vec = SelectionVector::with_capacity(len);
        let mut_sel_vec = unsafe { sel_vec.as_slice_mut(len) };
        for (dst, src) in mut_sel_vec.iter_mut().zip(
            self.codes.as_slice::<I>()[offset..offset + len]
                .iter()
                .map(|v| v.as_()),
        ) {
            *dst = src
        }

        vector.slice_to_dictionary(sel_vec, len);

        Ok(())
    }
}
