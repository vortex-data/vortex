use std::marker::PhantomData;
use std::sync::Arc;

use duckdb::core::{FlatVector, SelectionVector};
use duckdb::vtab::arrow::WritableVector;
use num_traits::AsPrimitive;
use vortex_array::ToCanonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_dict::DictArray;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::VortexResult;

use crate::exporter::create_exporter;
use crate::{ArrayExporter, ConversionCache, ToDuckDBType};

struct DictExporter<I: NativePType> {
    // Store the dictionary values once and export the same dictionary with each codes chunk.
    values_vector: FlatVector, // NOTE(ngates): not actually flat...
    values_len: usize,
    codes: PrimitiveArray,
    codes_type: PhantomData<I>,
}

pub(crate) fn new_exporter(
    array: &DictArray,
    cache: &mut ConversionCache,
) -> VortexResult<Box<dyn ArrayExporter>> {
    // Grab the cache dictionary values.
    let values = array.values();
    let values_key = Arc::as_ptr(values).addr();
    let values_vector = match cache.values_cache.get(&values_key) {
        None => {
            // Create a new DuckDB vector for the values.
            let mut vector = FlatVector::allocate_new_vector_with_capacity(
                values.dtype().to_duckdb_type()?,
                values.len(),
            );
            create_exporter(values, cache)?.export(0, values.len(), &mut vector)?;
            let unowned = vector.clone();
            cache
                .values_cache
                .insert(values_key, (values.clone(), vector));
            unowned
        }
        Some((_array, vector)) => vector.clone(),
    };

    let codes = array.codes().to_primitive()?;
    match_each_integer_ptype!(codes.ptype(), |$I| {
        Ok(Box::new(DictExporter {
            values_vector,
            values_len: array.values().len(),
            codes,
            codes_type: PhantomData::<$I>,
        }))
    })
}

impl<I: NativePType + AsPrimitive<u32>> ArrayExporter for DictExporter<I> {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
    ) -> VortexResult<()> {
        let mut vector = vector.flat_vector();
        // Copy across the dictionary values.
        vector.reference(&self.values_vector);

        // Slice with a selection vector from the codes.
        let sel_vec = SelectionVector::from_iter(
            self.codes.as_slice::<I>()[offset..offset + len]
                .iter()
                .map(|v| v.as_()),
        );

        vector.slice(self.values_len as u64, sel_vec);

        Ok(())
    }
}
