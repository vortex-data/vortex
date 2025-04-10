use duckdb::core::{FlatVector, SelectionVector};
use duckdb::ffi::{idx_t, sel_t};
use duckdb::vtab::arrow::WritableVector;
use num_traits::AsPrimitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::scalar_at;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{Array, ToCanonical};
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_runend::{RunEndArray, trimmed_ends_iter};

use crate::convert::array::array_ref::to_duckdb;
use crate::convert::scalar::ToDuckDBScalar;
use crate::{ConversionCache, DUCKDB_STANDARD_VECTOR_SIZE, ToDuckDB, ToDuckDBType};

pub fn selection_vector_from_ends_array(
    ends: PrimitiveArray,
    offset: usize,
    length: usize,
) -> VortexResult<SelectionVector> {
    match_each_integer_ptype!(ends.ptype(), |$E| {
        selection_vector_from_ends_slice(
            ends.as_slice::<$E>(),
            offset,
            length,
        )
    })
}

pub fn selection_vector_from_ends_slice<E: NativePType + AsPrimitive<usize> + Ord>(
    ends: &[E],
    offset: usize,
    length: usize,
) -> VortexResult<SelectionVector> {
    assert!(length <= DUCKDB_STANDARD_VECTOR_SIZE);

    let mut selection = SelectionVector::new(length as idx_t);
    let data_slice = selection.as_data_slice();

    let mut start = 0;
    for (value, end) in trimmed_ends_iter(ends, offset, length).enumerate() {
        assert!(end <= length, "Runend end must be less than overall length");

        // SAFETY:
        // We preallocate enough capacity because we know the total length
        unsafe {
            data_slice
                .get_unchecked_mut(start..end)
                .fill(sel_t::try_from(value)?);
        }
        start = end;
    }
    Ok(selection)
}

// We can convert a run end array into a dictionary like array and pass that to duckdb.
impl ToDuckDB for RunEndArray {
    fn to_duckdb(
        &self,
        chunk: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()> {
        if self.values().len() == 1 {
            let constant = scalar_at(self.values(), 0)?;
            let value = constant.try_to_duckdb_scalar()?;
            chunk.flat_vector().assign_to_constant(&value);
            return Ok(());
        }

        let mut vector: FlatVector = if self.values().len() <= DUCKDB_STANDARD_VECTOR_SIZE {
            to_duckdb(self.values(), chunk, cache)?;
            chunk.flat_vector()
        } else {
            // If the values don't fit allocated a larger vector and that the data chunk vector
            // reference this new one.
            let mut value_vector = FlatVector::allocate_new_vector_with_capacity(
                self.values().dtype().to_duckdb_type()?,
                self.values().len(),
            );
            to_duckdb(self.values(), &mut value_vector, cache)?;

            let mut vector = chunk.flat_vector();
            vector.reference(&value_vector);
            vector
        };
        let sel = selection_vector_from_ends_array(
            self.ends().to_primitive()?,
            self.offset(),
            self.len(),
        )?;
        vector.slice(self.values().len() as u64, sel);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use duckdb::core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId};
    use itertools::Itertools;
    use vortex_array::arrays::StructArray;
    use vortex_array::compute::slice;
    use vortex_array::{Array, IntoArray};
    use vortex_buffer::buffer;
    use vortex_runend::RunEndArray;

    use crate::{ConversionCache, to_duckdb_chunk};

    #[test]
    fn test_run_end_array_to_duckdb() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10, 14].into_array(),
            buffer![1i32, 2, 3, 4].into_array(),
        )
        .unwrap();

        let arr = slice(arr.to_array().as_ref(), 1, 5).unwrap();

        let struct_ = StructArray::from_fields(&[("a", arr)]).unwrap();

        let mut chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Integer)]);
        to_duckdb_chunk(&struct_, &mut chunk, &mut ConversionCache::default()).unwrap();

        chunk.verify();
        assert_eq!(
            format!("{:?}", chunk),
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 4 = [ 1, 2, 2, 2]
"#
        );
    }

    #[test]
    fn test_run_end_array_large_to_duckdb() {
        let arr = RunEndArray::try_new(
            buffer![1000u32, 2000, 3000].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap();

        let arr = slice(arr.to_array().as_ref(), 900, 2948).unwrap();

        let struct_ = StructArray::from_fields(&[("a", arr)]).unwrap();

        let mut chunk = DataChunkHandle::new(&[LogicalTypeHandle::from(LogicalTypeId::Integer)]);
        to_duckdb_chunk(&struct_, &mut chunk, &mut ConversionCache::default()).unwrap();

        chunk.verify();
        assert_eq!(
            format!("{:?}", chunk),
            format!(
                r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 2048 = [ {}, {}, {}]
"#,
                (0..100).map(|_| "1").join(", "),
                (0..1000).map(|_| "2").join(", "),
                (0..948).map(|_| "3").join(", "),
            ),
        );
    }
}
