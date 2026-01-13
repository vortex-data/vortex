// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex::array::arrays::PrimitiveArray;
use vortex::dtype::NativePType;
use vortex::dtype::match_each_native_ptype;
use vortex::error::VortexResult;

use crate::duckdb::Vector;
use crate::duckdb::VectorBuffer;
use crate::exporter::ColumnExporter;
use crate::exporter::validity;

struct PrimitiveExporter<T: NativePType> {
    len: usize,
    start: *const T,
    shared_buffer: VectorBuffer,
    _phantom_type: PhantomData<T>,
}

pub fn new_exporter(array: PrimitiveArray) -> VortexResult<Box<dyn ColumnExporter>> {
    match_each_native_ptype!(array.ptype(), |T| {
        let buffer = array.buffer::<T>();
        let prim = Box::new(PrimitiveExporter {
            len: buffer.len(),
            start: buffer.as_ptr(),
            shared_buffer: VectorBuffer::new(buffer),
            _phantom_type: Default::default(),
        });
        Ok(validity::new_exporter(array.validity_mask(), prim))
    })
}

impl<T: NativePType> ColumnExporter for PrimitiveExporter<T> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        assert!(self.len >= offset + len);

        let pos = unsafe { self.start.add(offset) };
        unsafe { vector.set_vector_buffer(&self.shared_buffer) };
        // While we are setting a *mut T this is an artifact of the C API, this is in fact const.
        unsafe { vector.set_data_ptr(pos as *mut T) };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use vortex::error::VortexExpect;

    use super::*;
    use crate::cpp;
    use crate::duckdb::DUCKDB_STANDARD_VECTOR_SIZE;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;

    #[test]
    fn test_primitive_exporter() {
        let arr = PrimitiveArray::from_iter(0..10);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(arr)
            .unwrap()
            .export(0, 3, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 3 = [ 0, 1, 2]
"#
        );
    }

    #[test]
    fn test_long_primitive_exporter() {
        const ARRAY_COUNT: usize = 2;
        const LEN: usize = DUCKDB_STANDARD_VECTOR_SIZE * ARRAY_COUNT;
        let arr = PrimitiveArray::from_iter(0..i32::try_from(LEN).vortex_expect(""));

        {
            let mut chunk = (0..ARRAY_COUNT)
                .map(|_| DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]))
                .collect_vec();

            for i in 0..ARRAY_COUNT {
                new_exporter(arr.clone())
                    .unwrap()
                    .export(
                        i * DUCKDB_STANDARD_VECTOR_SIZE,
                        DUCKDB_STANDARD_VECTOR_SIZE,
                        &mut chunk[i].get_vector(0),
                    )
                    .unwrap();
                chunk[i].set_len(DUCKDB_STANDARD_VECTOR_SIZE);

                assert_eq!(
                    format!("{}", String::try_from(&chunk[i]).unwrap()),
                    format!(
                        r#"Chunk - [1 Columns]
- FLAT INTEGER: {DUCKDB_STANDARD_VECTOR_SIZE} = [ {}]
"#,
                        &(i * DUCKDB_STANDARD_VECTOR_SIZE..(i + 1) * DUCKDB_STANDARD_VECTOR_SIZE)
                            .map(|i| i.to_string())
                            .join(", ")
                    )
                );
            }
        }
    }
}
