// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::list::ListDataParts;
use vortex::array::match_each_integer_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::IntegerPType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
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

    if matches!(validity, Validity::AllInvalid) {
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
        let offsets = &self.offsets.as_slice::<O>()[offset..offset + len + 1];
        debug_assert_eq!(offsets.len(), len + 1);

        // SAFETY: TODO(connor): Pretty sure that `export` needs to be `unsafe`.
        let duckdb_list_views: &mut [cpp::duckdb_list_entry] =
            unsafe { vector.as_slice_mut::<cpp::duckdb_list_entry>(len) };
        debug_assert_eq!(duckdb_list_views.len(), len);

        for i in 0..len {
            let offset = offsets[i]
                .to_u64()
                .ok_or_else(|| vortex_err!("somehow unable to convert an offset to u64"))?;
            let length = (offsets[i + 1] - offsets[i])
                .to_u64()
                .ok_or_else(|| vortex_err!("somehow unable to convert an offset to u64"))?;

            debug_assert!(offset + length <= self.num_elements as u64);

            duckdb_list_views[i] = cpp::duckdb_list_entry { offset, length };
        }

        let child = vector.list_vector_get_child_mut();
        child.reference(&self.duckdb_elements.lock());

        vector.list_vector_set_size(self.num_elements as u64)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray as _;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::error::VortexExpect;

    use super::*;
    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::new_array_exporter;

    #[test]
    #[ignore = "TODO(connor)[4809]: Exporters do not correctly handle empty vectors"]
    fn test_export_empty_list() {
        let list = unsafe {
            ListArray::new_unchecked(
                Buffer::<u32>::empty().into_array(),
                Buffer::<u32>::empty().into_array(),
                Validity::AllValid,
            )
        }
        .into_array();

        let list_type = LogicalType::list_type(LogicalType::int32())
            .vortex_expect("LogicalTypeRef creation should succeed for test data");
        let mut chunk = DataChunk::new([list_type]);

        let mut ctx = SESSION.create_execution_ctx();
        new_array_exporter(list, &ConversionCache::default(), &mut ctx)
            .unwrap()
            .export(0, 0, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(0);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT INTEGER[]: 0 = [ ]
"#
        );
    }

    #[test]
    fn test_export_non_empty_list_of_strings() {
        let list = unsafe {
            ListArray::new_unchecked(
                <VarBinArray as FromIterator<_>>::from_iter([
                    Some("abc"),
                    Some("def"),
                    None,
                    Some("ghi"),
                ])
                .into_array(),
                buffer![0u8, 1, 2, 3, 4].into_array(),
                Validity::from_iter([true, true, false, true]),
            )
        }
        .into_array();

        let list_type = LogicalType::list_type(LogicalType::varchar())
            .vortex_expect("LogicalTypeRef creation should succeed for test data");
        let mut chunk = DataChunk::new([list_type]);

        let mut ctx = SESSION.create_execution_ctx();
        new_array_exporter(list, &ConversionCache::default(), &mut ctx)
            .unwrap()
            .export(0, 4, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(4);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT VARCHAR[]: 4 = [ [abc], [def], NULL, [ghi]]
"#
        );
    }
}
