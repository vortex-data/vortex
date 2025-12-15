// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::array::ToCanonical;
use vortex::array::VectorExecutor;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::dtype::IntegerPType;
use vortex::dtype::PTypeDowncastExt;
use vortex::dtype::match_each_integer_ptype;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::mask::Mask;
use vortex::session::VortexSession;
use vortex_vector::primitive::PVector;

use super::ConversionCache;
use super::new_array_exporter_with_flatten;
use super::new_array_vector_exporter_with_flatten;
use crate::cpp;
use crate::duckdb::Vector;
use crate::exporter::ColumnExporter;

struct ListExporter<O, S> {
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
    array: &ListViewArray,
    cache: &ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Cache an `elements` vector up front so that future exports can reference it.
    let elements = array.elements();
    let num_elements = elements.len();

    let values_key = Arc::as_ptr(elements).addr();
    // Check if we have a cached vector and extract it if we do.
    let cached_elements = cache
        .values_cache
        .get(&values_key)
        .map(|entry| entry.value().1.clone());

    let shared_elements = match cached_elements {
        Some(elements) => elements,
        None => {
            // We have no cached the vector yet, so create a new DuckDB vector for the elements.
            let mut duckdb_elements =
                Vector::with_capacity(elements.dtype().try_into()?, elements.len());
            let elements_exporter = new_array_exporter_with_flatten(array.elements(), cache, true)?;

            if !elements.is_empty() {
                elements_exporter.export(0, elements.len(), &mut duckdb_elements)?;
            }

            let shared_elements = Arc::new(Mutex::new(duckdb_elements));
            cache
                .values_cache
                .insert(values_key, (elements.clone(), shared_elements.clone()));

            shared_elements
        }
    };

    let offsets = array.offsets().to_primitive();
    let sizes = array.sizes().to_primitive();

    let boxed = match_each_integer_ptype!(offsets.ptype(), |O| {
        match_each_integer_ptype!(sizes.ptype(), |S| {
            Box::new(ListExporter {
                validity: array.validity_mask(),
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

impl<O: IntegerPType, S: IntegerPType> ColumnExporter for ListExporter<O, S> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
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

        let mut child = vector.list_vector_get_child();
        child.reference(&self.duckdb_elements.lock());

        vector.list_vector_set_size(self.num_elements as u64)?;

        Ok(())
    }
}

struct ListVectorExporter<O, S> {
    validity: Mask,
    /// We cache the child elements of our list array so that we don't have to export it every time,
    /// and we also share it across any other exporters who want to export this array.
    ///
    /// Note that we are trading less compute for more memory here, as we will export the entire
    /// array in the constructor of the exporter (`new_exporter`) even if some of the elements are
    /// unreachable.
    duckdb_elements: Arc<Mutex<Vector>>,
    offsets: PVector<O>,
    sizes: PVector<S>,
    num_elements: usize,
}

pub(crate) fn new_vector_exporter(
    array: ArrayRef,
    cache: &ConversionCache,
    session: &VortexSession,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let array = array.to_listview();
    // Cache an `elements` vector up front so that future exports can reference it.
    let elements = array.elements();
    let num_elements = elements.len();

    let values_key = Arc::as_ptr(elements).addr();
    // Check if we have a cached vector and extract it if we do.
    let cached_elements = cache
        .values_cache
        .get(&values_key)
        .map(|entry| entry.value().1.clone());

    let shared_elements = match cached_elements {
        Some(elements) => elements,
        None => {
            // We have no cached the vector yet, so create a new DuckDB vector for the elements.
            let mut duckdb_elements =
                Vector::with_capacity(elements.dtype().try_into()?, elements.len());
            let elements_exporter = new_array_vector_exporter_with_flatten(
                array.elements().clone(),
                cache,
                session,
                true,
            )?;

            if !elements.is_empty() {
                elements_exporter.export(0, elements.len(), &mut duckdb_elements)?;
            }

            let shared_elements = Arc::new(Mutex::new(duckdb_elements));
            cache
                .values_cache
                .insert(values_key, (elements.clone(), shared_elements.clone()));

            shared_elements
        }
    };

    let offsets = array.offsets().execute_vector(session)?.into_primitive();
    let sizes = array.sizes().execute_vector(session)?.into_primitive();

    let boxed = match_each_integer_ptype!(offsets.ptype(), |O| {
        match_each_integer_ptype!(sizes.ptype(), |S| {
            Box::new(ListVectorExporter {
                validity: array.validity_mask(),
                duckdb_elements: shared_elements,
                offsets: offsets.downcast::<O>(),
                sizes: sizes.downcast::<O>(),
                num_elements,
            }) as Box<dyn ColumnExporter>
        })
    });

    Ok(boxed)
}

impl<O: IntegerPType, S: IntegerPType> ColumnExporter for ListVectorExporter<O, S> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
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

        let offsets = &self.offsets.as_ref()[offset..offset + len];
        let sizes = &self.sizes.as_ref()[offset..offset + len];
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

        let mut child = vector.list_vector_get_child();
        child.reference(&self.duckdb_elements.lock());

        vector.list_vector_set_size(self.num_elements as u64)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray as _;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::error::VortexUnwrap;

    use super::*;
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

        let list_type = LogicalType::list_type(LogicalType::int32()).vortex_unwrap();
        let mut chunk = DataChunk::new([list_type]);

        new_array_exporter(&list, &ConversionCache::default())
            .unwrap()
            .export(0, 0, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(0);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
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

        let list_type = LogicalType::list_type(LogicalType::int32()).vortex_unwrap();
        let mut chunk = DataChunk::new([list_type]);

        new_array_exporter(&list, &ConversionCache::default())
            .unwrap()
            .export(0, 3, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
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

        let list_type = LogicalType::list_type(LogicalType::varchar()).vortex_unwrap();
        let mut chunk = DataChunk::new([list_type]);

        new_array_exporter(&list, &ConversionCache::default())
            .unwrap()
            .export(0, 4, &mut chunk.get_vector(0))
            .unwrap();
        chunk.set_len(4);

        assert_eq!(
            format!("{}", String::try_from(&chunk).unwrap()),
            r#"Chunk - [1 Columns]
- FLAT VARCHAR[]: 4 = [ [], [abc, def, NULL], NULL, []]
"#
        );
    }
}
