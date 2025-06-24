use std::marker::PhantomData;
use std::sync::Arc;

use duckdb::core::{FlatVector, SelectionVector};
use duckdb::vtab::arrow::WritableVector;
use num_traits::AsPrimitive;
use vortex::ToCanonical;
use vortex::arrays::PrimitiveArray;
use vortex::dtype::{NativePType, match_each_integer_ptype};
use vortex::encodings::dict::DictArray;
use vortex::error::VortexResult;

use crate::exporter::new_array_exporter;
use crate::{ColumnExporter, ConversionCache, ToDuckDBType};

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
) -> VortexResult<Box<dyn ColumnExporter>> {
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
            values_len: array.values().len(),
            codes,
            codes_type: PhantomData::<I>,
        }))
    })
}

impl<I: NativePType + AsPrimitive<u32>> ColumnExporter for DictExporter<I> {
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

#[cfg(test)]
mod tests {
    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use vortex::IntoArray;
    use vortex::arrays::ConstantArray;
    use vortex::buffer::buffer;
    use vortex::encodings::dict::DictArray;
    use vortex::scalar::Scalar;

    use super::*;
    use crate::ConversionCache;

    #[test]
    fn test_dict() {
        let arr = DictArray::try_new(
            buffer![3u32, 2, 1, 0].into_array(),
            buffer![0i32, 1, 2, 3].into_array(),
        )
        .unwrap();

        let chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Integer)]);
        chunk.set_len(arr.len());

        new_exporter(&arr, &mut ConversionCache::default())
            .unwrap()
            .export(0, 4, &mut chunk.flat_vector(0))
            .unwrap();

        chunk.verify();
        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 4 = [ 3, 2, 1, 0]
"#
        );
    }

    /// DuckDB doesn't permit constant values in a dictionary, which makes sense, since then the
    /// whole vector is constant anyway.
    #[test]
    fn test_dict_constant_values() {
        let arr = DictArray::try_new(
            buffer![0u32, 0, 1, 3].into_array(),
            ConstantArray::new(Scalar::from(1i32), 4).into_array(),
        )
        .unwrap();

        let chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Integer)]);
        chunk.set_len(arr.len());

        new_exporter(&arr, &mut ConversionCache::default())
            .unwrap()
            .export(0, 4, &mut chunk.flat_vector(0))
            .unwrap();

        chunk.verify();
        assert_eq!(
            format!("{chunk:?}"),
            r#"Chunk - [1 Columns]
- CONSTANT INTEGER: 4 = [ 1]
"#
        );
    }
}
