// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::listview::ListViewArrayParts;
use vortex::array::match_each_integer_ptype;
use vortex::dtype::DType;
use vortex::dtype::IntegerPType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::mask::Mask;

use super::ConversionCache;
use super::all_invalid;
use super::new_array_exporter_with_flatten;
use crate::cpp;
use crate::duckdb::LogicalType;
use crate::duckdb::Vector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;

struct ListViewExporter<O, S> {
    validity: Mask,
    /// We cache the child elements of our list array so that we don't have to export it every time,
    /// and we also share it across any other exporters who want to export this array.
    ///
    /// Note that we are trading less compute for more memory here, as we will export the entire
    /// array in the constructor of the exporter (`new_exporter`) even if some of the elements are
    /// unreachable.
    duckdb_elements: Arc<Mutex<Vector>>,
    offsets: PrimitiveArray,
    sizes: PrimitiveArray,
    num_elements: usize,
    offset_type: PhantomData<O>,
    size_type: PhantomData<S>,
}

pub(crate) fn new_exporter(
    array: ListViewArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let len = array.len();
    let ListViewArrayParts {
        elements_dtype,
        elements,
        offsets,
        sizes,
        validity,
    } = array.into_data().into_parts();
    // Cache an `elements` vector up front so that future exports can reference it.
    let num_elements = elements.len();
    let nullability = validity.nullability();
    let validity = validity.to_array(len).execute::<Mask>(ctx)?;

    if validity.all_false() {
        let ltype = LogicalType::try_from(DType::List(elements_dtype, nullability))?;
        return Ok(all_invalid::new_exporter(len, &ltype));
    }

    let values_key = elements.addr();
    // Check if we have a cached vector and extract it if we do.
    let cached_elements = cache
        .values_cache
        .get(&values_key)
        .map(|entry| entry.value().1.clone());

    let shared_elements = match cached_elements {
        Some(elements) => elements,
        None => {
            // We have no cached the vector yet, so create a new DuckDB vector for the elements.
            let elements_type: LogicalType = elements_dtype.as_ref().try_into()?;
            let mut duckdb_elements = Vector::with_capacity(&elements_type, elements.len());
            let elements_exporter =
                new_array_exporter_with_flatten(elements.clone(), cache, ctx, true)?;

            if !elements.is_empty() {
                elements_exporter.export(0, elements.len(), &mut duckdb_elements, ctx)?;
            }

            let shared_elements = Arc::new(Mutex::new(duckdb_elements));
            cache
                .values_cache
                .insert(values_key, (elements, shared_elements.clone()));

            shared_elements
        }
    };

    let offsets = offsets.execute::<PrimitiveArray>(ctx)?;
    let sizes = sizes.execute::<PrimitiveArray>(ctx)?;

    let boxed = match_each_integer_ptype!(offsets.ptype(), |O| {
        match_each_integer_ptype!(sizes.ptype(), |S| {
            Box::new(ListViewExporter {
                validity,
                duckdb_elements: shared_elements,
                offsets,
                sizes,
                num_elements,
                offset_type: PhantomData::<O>,
                size_type: PhantomData::<S>,
            }) as Box<dyn ColumnExporter>
        })
    });

    Ok(boxed)
}

impl<O: IntegerPType, S: IntegerPType> ColumnExporter for ListViewExporter<O, S> {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Verify that offset + len doesn't exceed the validity mask length.
        assert!(
            offset + len <= self.validity.len(),
            "Export range [{}, {}) exceeds validity mask length {}",
            offset,
            offset + len,
            self.validity.len()
        );

        // Set validity if necessary.
        if unsafe { vector.set_validity(&self.validity, offset, len) } {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        let offsets = &self.offsets.as_slice::<O>()[offset..offset + len];
        let sizes = &self.sizes.as_slice::<S>()[offset..offset + len];
        debug_assert_eq!(offsets.len(), len);
        debug_assert_eq!(sizes.len(), len);

        // SAFETY: TODO(connor): Pretty sure that `export` needs to be `unsafe`.
        let duckdb_list_views: &mut [cpp::duckdb_list_entry] =
            unsafe { vector.as_slice_mut::<cpp::duckdb_list_entry>(len) };
        debug_assert_eq!(duckdb_list_views.len(), len);

        for i in 0..len {
            let offset = offsets[i]
                .to_u64()
                .ok_or_else(|| vortex_err!("somehow unable to convert an offset to u64"))?;
            let length = sizes[i]
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
            ListViewArray::new_unchecked(
                Buffer::<u32>::empty().into_array(),
                Buffer::<u32>::empty().into_array(),
                Buffer::<u32>::empty().into_array(),
                Validity::AllValid,
            )
            .with_zero_copy_to_list(true)
        }
        .into_array();

        let list_type = LogicalType::list_type(LogicalType::varchar())
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
    fn test_export_non_empty_list_with_preceding_and_trailing_garbage() {
        let list = unsafe {
            ListViewArray::new_unchecked(
                buffer![0, 1, 2, 3, 4, 5].into_array(),
                buffer![1u8, 2, 3].into_array(),
                buffer![1u8, 1, 1].into_array(),
                Validity::AllValid,
            )
            .with_zero_copy_to_list(true)
        }
        .into_array();

        let list_type = LogicalType::list_type(LogicalType::int32())
            .vortex_expect("LogicalTypeRef creation should succeed for test data");
        let mut chunk = DataChunk::new([list_type]);

        let mut ctx = SESSION.create_execution_ctx();
        new_array_exporter(list, &ConversionCache::default(), &mut ctx)
            .unwrap()
            .export(0, 3, chunk.get_vector_mut(0), &mut ctx)
            .unwrap();
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&*chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT INTEGER[]: 3 = [ [1], [2], [3]]
"#
        );
    }

    #[test]
    fn test_export_non_empty_list_of_strings() {
        let list = unsafe {
            ListViewArray::new_unchecked(
                <VarBinArray as FromIterator<_>>::from_iter([
                    Some("abc"),
                    Some("def"),
                    None,
                    Some("ghi"),
                ])
                .into_array(),
                buffer![0u8, 0, 3, 4].into_array(),
                buffer![0u8, 3, 1, 0].into_array(),
                Validity::from_iter([true, true, false, true]),
            )
            .with_zero_copy_to_list(true)
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
- FLAT VARCHAR[]: 4 = [ [], [abc, def, NULL], NULL, []]
"#
        );
    }
}
