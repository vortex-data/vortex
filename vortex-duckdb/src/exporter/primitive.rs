// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex::array::ExecutionCtx;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::vtable::ValidityHelper;
use vortex::dtype::NativePType;
use vortex::dtype::match_each_native_ptype;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::LogicalType;
use crate::duckdb::Vector;
use crate::duckdb::VectorBuffer;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::validity;

struct PrimitiveExporter<T: NativePType> {
    len: usize,
    start: *const T,
    shared_buffer: VectorBuffer,
    _phantom_type: PhantomData<T>,
}

pub fn new_exporter(
    array: PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let validity = array
        .validity()
        .to_array(array.len())
        .execute::<Mask>(ctx)?;

    if validity.all_false() {
        return Ok(all_invalid::new_exporter(
            array.len(),
            &LogicalType::try_from(array.ptype())?,
        ));
    }

    match_each_native_ptype!(array.ptype(), |T| {
        let buffer = array.to_buffer::<T>();
        let prim = Box::new(PrimitiveExporter {
            len: buffer.len(),
            start: buffer.as_ptr(),
            shared_buffer: VectorBuffer::new(buffer),
            _phantom_type: Default::default(),
        });
        Ok(validity::new_exporter(validity, prim))
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
    use vortex_array::VortexSessionExecute;

    use super::*;
    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::duckdb::duckdb_vector_size;

    #[test]
    fn test_primitive_exporter() {
        let arr = PrimitiveArray::from_iter(0..10);

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(arr, &mut SESSION.create_execution_ctx())
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
        let vector_size = duckdb_vector_size();
        const ARRAY_COUNT: usize = 2;
        let len = vector_size * ARRAY_COUNT;
        #[expect(clippy::cast_possible_truncation, reason = "test data fits in i32")]
        let arr = PrimitiveArray::from_iter(0..len as i32);

        {
            let mut chunk = (0..ARRAY_COUNT)
                .map(|_| DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]))
                .collect_vec();

            for i in 0..ARRAY_COUNT {
                new_exporter(arr.clone(), &mut SESSION.create_execution_ctx())
                    .unwrap()
                    .export(i * vector_size, vector_size, &mut chunk[i].get_vector(0))
                    .unwrap();
                chunk[i].set_len(vector_size);

                assert_eq!(
                    format!("{}", String::try_from(&chunk[i]).unwrap()),
                    format!(
                        r#"Chunk - [1 Columns]
- FLAT INTEGER: {vector_size} = [ {}]
"#,
                        &(i * vector_size..(i + 1) * vector_size)
                            .map(|i| i.to_string())
                            .join(", ")
                    )
                );
            }
        }
    }
}
