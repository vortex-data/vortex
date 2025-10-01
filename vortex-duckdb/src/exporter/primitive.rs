// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex::arrays::PrimitiveArray;
use vortex::buffer::{Buffer, ByteBuffer};
use vortex::dtype::{NativePType, match_each_native_ptype};
use vortex::error::VortexResult;

use crate::duckdb::Vector;
use crate::exporter::{ColumnExporter, validity};

struct PrimitiveExporter<T: NativePType> {
    buffer: Buffer<u8>,
    array_type: PhantomData<T>,
}

pub fn new_exporter(array: &PrimitiveArray) -> VortexResult<Box<dyn ColumnExporter>> {
    let buffer: ByteBuffer = array.byte_buffer().clone();

    let prim = match_each_native_ptype!(array.ptype(), |T| {
        Box::new(PrimitiveExporter {
            buffer,
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
        assert!(self.buffer.len() * size_of::<T>() >= offset + len);

        let pos = unsafe { (self.buffer.as_ptr() as *const T).add(offset) };
        unsafe { vector.set_data_buffer(self.buffer.clone()) };
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
    use crate::duckdb::{DUCKDB_STANDARD_VECTOR_SIZE, DataChunk, LogicalType};

    #[test]
    fn test_primitive_exporter() {
        let arr = PrimitiveArray::from_iter(0..10);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr)
            .unwrap()
            .export(0, 3, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(3);

        // some-invalid codes cannot be exported as a dictionary.
        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 3 = [ 0, 1, 2]
"#
        );
    }

    #[test]
    fn test_long_primitive_exporter() {
        const VECTOR_COUNT: usize = 2;
        const LEN: usize = DUCKDB_STANDARD_VECTOR_SIZE * VECTOR_COUNT;
        let arr = PrimitiveArray::from_iter(0..i32::try_from(LEN).vortex_expect(""));

        {
            let mut chunk = (0..VECTOR_COUNT)
                .map(|_| DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]))
                .collect_vec();

            for i in 0..VECTOR_COUNT {
                new_exporter(&arr)
                    .unwrap()
                    .export(
                        i * DUCKDB_STANDARD_VECTOR_SIZE,
                        DUCKDB_STANDARD_VECTOR_SIZE,
                        &mut chunk[i].get_vector(0),
                    )
                    .unwrap();
                chunk[i].set_len(DUCKDB_STANDARD_VECTOR_SIZE);

                // some-invalid codes cannot be exported as a dictionary.
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
