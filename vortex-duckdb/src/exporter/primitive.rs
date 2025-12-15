// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex::array::ArrayRef;
use vortex::array::VectorExecutor;
use vortex::array::arrays::PrimitiveArray;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::dtype::PTypeDowncastExt;
use vortex::dtype::match_each_native_ptype;
use vortex::error::VortexResult;
use vortex::session::VortexSession;

use crate::duckdb::Vector;
use crate::duckdb::VectorBuffer;
use crate::exporter::ColumnExporter;
use crate::exporter::validity;

struct PrimitiveExporter<T: NativePType> {
    buffer: Buffer<T>,
    shared_buffer: VectorBuffer,
}

struct PrimitiveVectorExporter<T: NativePType> {
    len: usize,
    start: *const T,
    shared_buffer: VectorBuffer,
    _phantom_type: PhantomData<T>,
}

pub fn new_exporter(array: &PrimitiveArray) -> VortexResult<Box<dyn ColumnExporter>> {
    let prim = match_each_native_ptype!(array.ptype(), |T| {
        let buffer = array.buffer::<T>();
        Box::new(PrimitiveExporter {
            buffer: buffer.clone(),
            shared_buffer: VectorBuffer::new(buffer),
        }) as Box<dyn ColumnExporter>
    });
    Ok(if array.dtype().is_nullable() {
        validity::new_exporter(array.validity_mask(), prim)
    } else {
        prim
    })
}

pub fn new_vector_exporter(
    array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let vector = array.execute_vector(session)?.into_primitive();
    match_each_native_ptype!(vector.ptype(), |T| {
        let vector = vector.downcast::<T>();
        let (buffer, mask) = vector.into_parts();
        let prim = Box::new(PrimitiveVectorExporter {
            len: buffer.len(),
            start: buffer.as_ptr(),
            shared_buffer: VectorBuffer::new(buffer),
            _phantom_type: Default::default(),
        });
        Ok(validity::new_exporter(mask, prim))
    })
}

impl<T: NativePType> ColumnExporter for PrimitiveExporter<T> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        assert!(self.buffer.len() >= offset + len);

        let pos = unsafe { self.buffer.as_ptr().add(offset) };
        unsafe { vector.set_vector_buffer(&self.shared_buffer) };
        // While we are setting a *mut T this is an artifact of the C API, this is in fact const.
        unsafe { vector.set_data_ptr(pos as *mut T) };

        Ok(())
    }
}

impl<T: NativePType> ColumnExporter for PrimitiveVectorExporter<T> {
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
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
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

        new_exporter(&arr)
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
    fn test_primitive_vector_exporter() {
        let arr = PrimitiveArray::from_iter(0..10);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_vector_exporter(arr.into_array(), &VortexSession::default())
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

    #[test]
    fn test_long_primitive_vector_exporter() -> VortexResult<()> {
        const VECTOR_COUNT: usize = 2;
        const LEN: usize = DUCKDB_STANDARD_VECTOR_SIZE * VECTOR_COUNT;
        let arr = PrimitiveArray::from_iter(0..i32::try_from(LEN).vortex_expect(""));

        {
            let mut chunk = (0..VECTOR_COUNT)
                .map(|_| DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]))
                .collect_vec();

            let exporter = new_vector_exporter(arr.into_array(), &VortexSession::default())?;

            for i in 0..VECTOR_COUNT {
                exporter.export(
                    i * DUCKDB_STANDARD_VECTOR_SIZE,
                    DUCKDB_STANDARD_VECTOR_SIZE,
                    &mut chunk[i].get_vector(0),
                )?;
                chunk[i].set_len(DUCKDB_STANDARD_VECTOR_SIZE);

                assert_eq!(
                    format!("{}", String::try_from(&chunk[i])?),
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

        Ok(())
    }
}
