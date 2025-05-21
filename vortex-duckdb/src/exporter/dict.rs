use std::marker::PhantomData;

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

pub(crate) fn new_exporter(array: &DictArray) -> VortexResult<Box<dyn ArrayExporter>> {
    let mut values_vector = FlatVector::allocate_new_vector_with_capacity(
        array.values().dtype().to_duckdb_type()?,
        array.values().len(),
    );
    create_exporter(array.values())?.export(
        0,
        array.values().len(),
        &mut values_vector,
        &mut ConversionCache::default(),
    )?;

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
        _cache: &mut ConversionCache,
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
