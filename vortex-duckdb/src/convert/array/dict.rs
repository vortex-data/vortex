use std::sync::Arc;

use duckdb::core::{FlatVector, SelectionVector};
use duckdb::vtab::arrow::WritableVector;
use num_traits::AsPrimitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::take;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_dict::DictArray;
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};

use crate::convert::array::array_ref::to_duckdb;
use crate::convert::array::cache::ConversionCache;
use crate::{DUCKDB_STANDARD_VECTOR_SIZE, ToDuckDB, ToDuckDBType};

impl ToDuckDB for DictArray {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()> {
        let values = self.values();

        // Note you can only have nullable values (not codes/selection vectors),
        // so we cannot assign a selection vector.
        if !self.codes().all_valid()? {
            let values = take(values, self.codes())?;
            return to_duckdb(&values, chunk, cache);
        };

        let value_ptr = Arc::as_ptr(values).addr();

        let mut vector: FlatVector = if self.values().len() <= DUCKDB_STANDARD_VECTOR_SIZE {
            // If the values fit into a single vector, put the values in the pre-allocated vector.
            to_duckdb(values, chunk, cache)?;
            chunk.flat_vector()
        } else {
            // If the values don't fit allocated a larger vector and that the data chunk vector
            // reference this new one.
            let entry = cache.values_cache.get(&value_ptr);
            let value_vector = match entry {
                None => {
                    create_and_insert_duckdb_dict_value_array_into_cache(values, cache, value_ptr)?
                }
                Some((cached_array_ref, entry)) => {
                    if Arc::ptr_eq(cached_array_ref, values) {
                        entry.clone()
                    } else {
                        create_and_insert_duckdb_dict_value_array_into_cache(
                            values, cache, value_ptr,
                        )?
                    }
                }
            };

            let mut vector = chunk.flat_vector();
            vector.reference(&value_vector);
            vector
        };
        let sel = selection_vector_from_array(self.codes().to_primitive()?);
        vector.slice(values.len() as u64, sel);
        vector.set_dictionary_id(format!("{}-{}", cache.instance_id, value_ptr));
        Ok(())
    }
}

pub fn selection_vector_from_array(prim: PrimitiveArray) -> SelectionVector {
    match_each_integer_ptype!(prim.ptype(), |$P| {
        selection_vector_from_slice(prim.as_slice::<$P>())
    })
}

pub fn selection_vector_from_slice<P: NativePType + AsPrimitive<u32>>(
    slice: &[P],
) -> SelectionVector {
    slice.iter().map(|v| (*v).as_()).collect()
}

fn create_and_insert_duckdb_dict_value_array_into_cache(
    values: &ArrayRef,
    cache: &mut ConversionCache,
    value_ptr: usize,
) -> VortexResult<FlatVector> {
    let mut value_vector = FlatVector::allocate_new_vector_with_capacity(
        values.dtype().to_duckdb_type()?,
        values.len(),
    );
    let cached_array = cache.cached_array(values)?;
    to_duckdb(&cached_array, &mut value_vector, cache)?;
    cache
        .values_cache
        .insert(value_ptr, (values.clone(), value_vector));
    Ok(cache
        .values_cache
        .get(&value_ptr)
        .vortex_expect("just added")
        .1
        .clone())
}
