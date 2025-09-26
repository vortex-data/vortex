// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use bitvec::macros::internal::funty::Fundamental;
use num_traits::AsPrimitive;
use parking_lot::Mutex;
use vortex::arrays::{ConstantArray, ConstantVTable, PrimitiveArray};
use vortex::dtype::{NativePType, match_each_integer_ptype};
use vortex::encodings::dict::DictArray;
use vortex::error::VortexResult;
use vortex::{Array, ToCanonical};

use crate::duckdb::{SelectionVector, Vector};
use crate::exporter::cache::ConversionCache;
use crate::exporter::{ColumnExporter, constant, new_array_exporter};

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
    if let Some(constant) = values.as_opt::<ConstantVTable>() {
        return constant::new_exporter_with_mask(
            &ConstantArray::new(constant.scalar().clone(), array.codes().len()),
            array.codes().validity_mask(),
            cache,
        );
    }

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

            // This is a bit of a hack, but we need to return the values vector into a dictionary
            // typed vector, where we can later set different selection vectors.
            // If this is not done here the threads will race to convert the value into a dictionary.
            Vector::with_capacity(vector.logical_type(), 0).dictionary(
                &vector,
                values.len(),
                &SelectionVector::with_capacity(0),
                0,
            );

            let vector = Arc::new(Mutex::new(vector));
            cache
                .values_cache
                .insert(values_key, (values.clone(), vector.clone()));

            vector
        }
    };

    let new_values_vector = {
        let values_vector = values_vector.lock();
        let mut new_values_vector = Vector::new(values_vector.logical_type());
        // Shares the underlying data which determines the vectors length.
        new_values_vector.reference(&values_vector);
        Vector::with_capacity(new_values_vector.logical_type(), 0).dictionary(
            &new_values_vector,
            values.len(),
            &SelectionVector::with_capacity(0),
            0,
        );
        Arc::new(Mutex::new(new_values_vector))
    };

    let codes = array.codes().to_primitive();
    match_each_integer_ptype!(codes.ptype(), |I| {
        Ok(Box::new(DictExporter {
            values_vector: new_values_vector,
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
        // Create a selection vector from the codes.
        let mut sel_vec = SelectionVector::with_capacity(len);
        let mut_sel_vec = unsafe { sel_vec.as_slice_mut(len) };
        for (dst, src) in mut_sel_vec.iter_mut().zip(
            // FIXME(joe): we ignore nullability in codes, fix with a specific null value in the values vector.
            self.codes.as_slice::<I>()[offset..offset + len]
                .iter()
                .map(|v| v.as_()),
        ) {
            *dst = src
        }

        {
            let values = self.values_vector.lock();
            let mut other = Vector::new(values.logical_type());
            other.reference(&*values);
            vector.dictionary(
                &other,
                self.values_len as usize,
                &sel_vec,
                len,
            );

            // Use a unique id used for each dictionary data array -- informing duckdb that the dict value
            // vector is the same, used for reuse of the hash in a join.
            vector.set_dictionary_id(format!("{}-{}", self.cache_id, self.value_id));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::IntoArray;
    use vortex::arrays::{ConstantArray, PrimitiveArray};
    use vortex::encodings::dict::DictArray;

    use crate::cpp;
    use crate::duckdb::{DataChunk, LogicalType};
    use crate::exporter::ConversionCache;
    use crate::exporter::dict::new_exporter;

    #[test]
    fn test_constant_dict() {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None, Some(0u32)]).into_array(),
            ConstantArray::new(10, 1).into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr, &ConversionCache::default())
            .unwrap()
            .export(0, 2, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 2 = [ NULL, 10]
"#
        );
    }
}
