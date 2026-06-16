// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use num_traits::AsPrimitive;
use parking_lot::Mutex;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::list::ListDataParts;
use vortex::array::match_each_integer_ptype;
use vortex::dtype::IntegerPType;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::mask::Mask;

use super::ConversionCache;
use super::all_invalid;
use super::new_array_exporter_with_flatten;
use super::validity;
use crate::cpp;
use crate::duckdb::LogicalType;
use crate::duckdb::Vector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;

struct ListExporter<O> {
    /// We cache the child elements of our list array so that we don't have to export it every time,
    /// and we also share it across any other exporters who want to export this array.
    ///
    /// Note that we are trading less compute for more memory here, as we will export the entire
    /// array in the constructor of the exporter (`new_exporter`) even if some of the elements are
    /// unreachable.
    duckdb_elements: Arc<Mutex<Vector>>,
    offsets: PrimitiveArray,
    num_elements: usize,
    offset_type: PhantomData<O>,
}

pub(crate) fn new_exporter(
    array: ListArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let array_len = array.len();
    // Cache an `elements` vector up front so that future exports can reference it.
    let ListDataParts {
        elements,
        offsets,
        validity,
        dtype: _dtype,
    } = array.into_data_parts();
    let num_elements = elements.len();

    if validity.definitely_all_null() {
        return Ok(all_invalid::new_exporter());
    }
    let validity = validity.to_array(array_len).execute::<Mask>(ctx)?;

    let values_key = elements.addr();
    // Check if we have a cached vector and extract it if we do.
    let cached_elements = cache
        .values_cache
        .get(&values_key)
        .map(|entry| Arc::clone(&entry.value().1));

    let shared_elements = match cached_elements {
        Some(elements) => elements,
        None => {
            // We have no cached the vector yet, so create a new DuckDB vector for the elements.
            let elements_type: LogicalType = elements.dtype().try_into()?;
            let mut duckdb_elements = Vector::with_capacity(&elements_type, num_elements);
            let elements_exporter =
                new_array_exporter_with_flatten(elements.clone(), cache, ctx, true)?;

            if num_elements != 0 {
                elements_exporter.export(0, num_elements, &mut duckdb_elements, ctx)?;
            }

            let shared_elements = Arc::new(Mutex::new(duckdb_elements));
            cache
                .values_cache
                .insert(values_key, (elements, Arc::clone(&shared_elements)));

            shared_elements
        }
    };

    let offsets = offsets.execute::<PrimitiveArray>(ctx)?;

    let boxed = match_each_integer_ptype!(offsets.ptype(), |O| {
        Box::new(ListExporter {
            duckdb_elements: shared_elements,
            offsets,
            num_elements,
            offset_type: PhantomData::<O>,
        }) as Box<dyn ColumnExporter>
    });

    Ok(validity::new_exporter(validity, boxed))
}

impl<O: IntegerPType> ColumnExporter for ListExporter<O> {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if len == 0 {
            return vector.list_vector_set_size(0);
        }

        // SAFETY: TODO(connor): Pretty sure that `export` needs to be `unsafe`.
        let duckdb_list_views: &mut [cpp::duckdb_list_entry] =
            unsafe { vector.as_slice_mut::<cpp::duckdb_list_entry>(len) };
        let offsets = &self.offsets.as_slice::<O>()[offset..offset + len + 1];

        let offset_start: u64 = offsets[0].as_().as_();
        let offset_end = offsets[len].as_().as_();
        vortex_ensure!(offset_end <= self.num_elements as u64);

        for i in 0..len {
            let offset: u64 = offsets[i].as_().as_();
            let next_offset: u64 = offsets[i + 1].as_().as_();
            duckdb_list_views[i] = cpp::duckdb_list_entry {
                offset: offset - offset_start,
                length: next_offset - offset,
            };
        }

        let sliced = {
            let elements = &self.duckdb_elements.lock();
            Vector::slice(elements, offset_start..offset_end)
        };

        let child_len = offset_end - offset_start;
        vector.list_vector_get_child_mut().reference(&sliced);
        vector.list_vector_set_size(child_len)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use num_traits::AsPrimitive;
    use vortex::array::IntoArray as _;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::dtype::DType;
    use vortex::dtype::PType;
    use vortex::encodings::runend::RunEnd;
    use vortex::error::VortexResult;

    use super::*;
    use crate::SESSION;
    use crate::convert::FromLogicalType;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::new_array_exporter;

    #[test]
    fn test_export_empty_list() -> VortexResult<()> {
        let list = ListArray::try_new(
            Buffer::<u32>::empty().into_array(),
            buffer![0u32].into_array(),
            Validity::AllValid,
        )?
        .into_array();

        let list_type = LogicalType::list_type(LogicalType::uint32())?;
        let mut chunk = DataChunk::new([list_type]);

        let mut ctx = SESSION.create_execution_ctx();
        new_array_exporter(list, &ConversionCache::default(), &mut ctx)?.export(
            0,
            0,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(0);

        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT UINTEGER[]: 0 = [ ]
"#
        );
        Ok(())
    }

    #[test]
    fn test_export_u64_list() -> VortexResult<()> {
        let list = ListArray::try_new(
            buffer![1u64, 2, 3, 4, 5].into_array(),
            buffer![0u8, 1, 2, 3, 4, 5].into_array(),
            Validity::AllValid,
        )?
        .into_array();
        assert_eq!(
            list.dtype(),
            &DType::List(
                Arc::new(DType::Primitive(PType::U64, false.into())),
                true.into()
            )
        );

        let list_type = LogicalType::list_type(LogicalType::uint64())?;
        let mut chunk = DataChunk::new([list_type]);

        let mut ctx = SESSION.create_execution_ctx();
        new_array_exporter(list, &ConversionCache::default(), &mut ctx)?.export(
            0,
            5,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(5);

        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT UBIGINT[]: 5 = [ [1], [2], [3], [4], [5]]
"#
        );
        Ok(())
    }

    #[test]
    fn test_export_u64_list_two_pass() -> VortexResult<()> {
        // [1], [2, 8], [3], [4], [5]
        let elements = buffer![1u64, 2, 8, 3, 4, 5].into_array();
        let offsets = buffer![0u8, 1, 3, 4, 5, 6].into_array();
        let list = ListArray::try_new(elements, offsets, Validity::AllValid)?.into_array();

        let u64_type = LogicalType::uint64();
        let list_type = LogicalType::list_type(u64_type)?;
        let mut chunk = DataChunk::new([list_type]);

        let mut ctx = SESSION.create_execution_ctx();
        let exporter = new_array_exporter(list, &ConversionCache::default(), &mut ctx)?;

        exporter.export(0, 2, chunk.get_vector_mut(0), &mut ctx)?;
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT UBIGINT[]: 2 = [ [1], [2, 8]]
"#
        );

        let u64_type = LogicalType::uint64();
        let list_vec = chunk.get_vector(0);
        let list_child = list_vec.list_vector_get_child();
        assert_eq!(
            DType::from_logical_type(&list_child.logical_type(), false.into())?,
            DType::from_logical_type(&u64_type, false.into())?
        );
        let child_len: usize = list_vec.list_vector_get_size().as_();
        assert_eq!(child_len, 3);
        let child_values = list_child.as_slice_with_len::<u64>(child_len);
        assert_eq!(child_values, [1, 2, 8]);

        exporter.export(2, 3, chunk.get_vector_mut(0), &mut ctx)?;
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT UBIGINT[]: 3 = [ [3], [4], [5]]
"#
        );

        let list_vec = chunk.get_vector(0);
        let child_len: usize = list_vec.list_vector_get_size().as_();
        assert_eq!(child_len, 3);
        let list_child = list_vec.list_vector_get_child();
        let child_values = list_child.as_slice_with_len::<u64>(child_len);
        assert_eq!(child_values, [3, 4, 5]);

        Ok(())
    }

    #[test]
    fn test_export_runend_list() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements_buffer = buffer![100f32, 100f32, 200f32, 200f32, 200f32].into_array();
        let elements = RunEnd::encode(elements_buffer, &mut ctx)?.into_array();
        let offsets = buffer![0u32, 2, 5].into_array();
        let list = ListArray::try_new(elements, offsets, Validity::AllValid)?.into_array();

        let list_type = LogicalType::list_type(LogicalType::float32())?;
        let mut chunk = DataChunk::new([list_type]);

        new_array_exporter(list, &ConversionCache::default(), &mut ctx)?.export(
            0,
            2,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT FLOAT[]: 2 = [ [100.0, 100.0], [200.0, 200.0, 200.0]]
"#
        );

        Ok(())
    }

    #[test]
    fn test_export_string_list() -> VortexResult<()> {
        let list = ListArray::try_new(
            <VarBinArray as FromIterator<_>>::from_iter([
                Some("abc"),
                Some("def"),
                None,
                Some("ghi"),
            ])
            .into_array(),
            buffer![0u8, 1, 2, 3, 4].into_array(),
            Validity::from_iter([true, true, false, true]),
        )?
        .into_array();

        let list_type = LogicalType::list_type(LogicalType::varchar())?;
        let mut chunk = DataChunk::new([list_type]);

        let mut ctx = SESSION.create_execution_ctx();
        new_array_exporter(list, &ConversionCache::default(), &mut ctx)?.export(
            0,
            4,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(4);

        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT VARCHAR[]: 4 = [ [abc], [def], NULL, [ghi]]
"#
        );

        Ok(())
    }
}
