// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use parking_lot::Mutex;
use std::marker::PhantomData;
use std::sync::Arc;

use bitvec::macros::internal::funty::Fundamental;
use num_traits::AsPrimitive;
use vortex::arrays::PrimitiveArray;
use vortex::dtype::{NativePType, match_each_integer_ptype};
use vortex::encodings::dict::DictArray;
use vortex::error::VortexResult;
use vortex::{Array, ToCanonical};

use crate::duckdb::{SelectionVector, Vector};
use crate::exporter::cache::ConversionCache;
use crate::exporter::{ColumnExporter, new_array_exporter};

struct DictExporter<I: NativePType> {
    // Store the dictionary values once and export the same dictionary with each codes chunk.
    values_vector: Arc<Mutex<Vector>>, // NOTE(ngates): not actually flat...
    values_len: u32,
    codes: PrimitiveArray,
    codes_type: PhantomData<I>,
    cache_id: u64,
    value_id: usize,
}

pub(crate) fn new_exporter(
    array: &DictArray,
    cache: &ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Grab the cache dictionary values.
    let values = array.values();
    let values_key = Arc::as_ptr(values).addr();

    // Check if we have a cached vector and extract it if we do.
    let cached_vector = cache
        .values_cache
        .get(&values_key)
        .map(|entry| entry.value().1.clone());

    let values_vector = match cached_vector {
        Some(vector) => vector,
        None => {
            // Create a new DuckDB vector for the values.
            let mut vector = Vector::with_capacity(values.dtype().try_into()?, values.len());
            new_array_exporter(values, cache)?.export(0, values.len(), &mut vector)?;

            let vector = Arc::new(Mutex::new(vector));
            cache
                .values_cache
                .insert(values_key, (values.clone(), vector.clone()));

            vector
        }
    };

    let codes = array.codes().to_primitive()?;
    match_each_integer_ptype!(codes.ptype(), |I| {
        Ok(Box::new(DictExporter {
            values_vector,
            values_len: values.len().as_u32(),
            codes,
            codes_type: PhantomData::<I>,
            cache_id: cache.instance_id(),
            value_id: values_key,
        }))
    })
}

impl<I: NativePType + AsPrimitive<u32>> ColumnExporter for DictExporter<I> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Copy across the dictionary values.
        vector.reference(&self.values_vector.lock());

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
        // Use a unique id to each dictionary data array -- telling duckdb that the dict value vector
        // is the same as reuse the hash in a join.
        vector.set_dictionary_id(format!("{}-{}", self.cache_id, self.value_id));
        vector.set_dictionary_len(self.values_len);

        Ok(())
    }
}
