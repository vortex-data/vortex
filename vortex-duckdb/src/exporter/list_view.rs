// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::marker::PhantomData;
use std::sync::Arc;

use num_traits::AsPrimitive;
use parking_lot::Mutex;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::ListViewArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::listview::DEFAULT_REBUILD_DENSITY_THRESHOLD;
use vortex::array::arrays::listview::DEFAULT_TRIM_ELEMENTS_THRESHOLD;
use vortex::array::arrays::listview::ListViewArrayExt;
use vortex::array::arrays::listview::ListViewDataParts;
use vortex::array::arrays::listview::ListViewRebuildMode;
use vortex::array::match_each_integer_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::IntegerPType;
use vortex::error::VortexExpect;
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

struct ListViewExporter<O, S> {
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
    // If the array is sufficiently sparse, rebuild. Otherwise the DuckDB vector will
    // hold an elements buffer containing unreferenced data in memory indefinitely,
    // and any compute pass over that buffer wastes work on data nothing references.
    let array = if array.is_zero_copy_to_list() {
        // A zctl array has no overlaps and no interior gaps, so the only unreferenced
        // elements are leading and trailing. Trimming them is much cheaper than a full rebuild.
        // Compute the referenced bounds once and reuse them for both the decision and the trim.
        let n_elts = array.elements().len();
        if n_elts == 0 || array.is_empty() {
            array
        } else {
            let (start, end) = array.referenced_element_bounds(ctx)?;
            let waste = (n_elts - (end - start)) as f32 / n_elts as f32;
            if waste > DEFAULT_TRIM_ELEMENTS_THRESHOLD {
                // SAFETY: we calculated valid start and end bounds
                unsafe { array.trim_elements(start, end)? }
            } else {
                array
            }
        }
    } else if array.upper_bound_density(ctx)? < DEFAULT_REBUILD_DENSITY_THRESHOLD {
        // Overlaps, gaps, or garbage may be present, so a full rebuild is needed to reclaim waste.
        array.rebuild(ListViewRebuildMode::MakeZeroCopyToList, ctx)?
    } else {
        array
    };

    let len = array.len();

    let ListViewDataParts {
        elements_dtype,
        elements,
        offsets,
        sizes,
        validity,
    } = array.into_data_parts();
    // Cache an `elements` vector up front so that future exports can reference it.
    let num_elements = elements.len();

    if matches!(validity, Validity::AllInvalid) {
        return Ok(all_invalid::new_exporter());
    }
    let validity = validity.to_array(len).execute::<Mask>(ctx)?;

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
                .insert(values_key, (elements, Arc::clone(&shared_elements)));

            shared_elements
        }
    };

    let offsets = offsets.execute::<PrimitiveArray>(ctx)?;
    let sizes = sizes.execute::<PrimitiveArray>(ctx)?;

    let boxed = match_each_integer_ptype!(offsets.ptype(), |O| {
        match_each_integer_ptype!(sizes.ptype(), |S| {
            Box::new(ListViewExporter {
                duckdb_elements: shared_elements,
                offsets,
                sizes,
                num_elements,
                offset_type: PhantomData::<O>,
                size_type: PhantomData::<S>,
            }) as Box<dyn ColumnExporter>
        })
    });

    Ok(validity::new_exporter(validity, boxed))
}

impl<O: IntegerPType, S: IntegerPType> ColumnExporter for ListViewExporter<O, S> {
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
        let offsets = &self.offsets.as_slice::<O>()[offset..offset + len];
        let sizes = &self.sizes.as_slice::<S>()[offset..offset + len];

        // SAFETY: TODO(connor): Pretty sure that `export` needs to be `unsafe`.
        let duckdb_list_views: &mut [cpp::duckdb_list_entry] =
            unsafe { vector.as_slice_mut::<cpp::duckdb_list_entry>(len) };

        let offset_start: u64 = offsets
            .iter()
            .min()
            .vortex_expect("offsets array is empty")
            .as_()
            .as_();

        let mut offset_end: u64 = 0;
        for i in 0..len {
            let offset: u64 = offsets[i].as_().as_();
            let length: u64 = sizes[i].as_().as_();
            offset_end = max(offset_end, offset + length);
            duckdb_list_views[i] = cpp::duckdb_list_entry {
                offset: offset - offset_start,
                length,
            };
        }
        vortex_ensure!(offset_start <= offset_end);
        vortex_ensure!(offset_end <= self.num_elements as u64);

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
    use vortex::array::IntoArray as _;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex_runend::RunEnd;

    use super::*;
    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::new_array_exporter;

    #[test]
    fn test_export_empty_list() -> VortexResult<()> {
        let elements = Buffer::<u32>::empty().into_array();
        let offsets = Buffer::<u32>::empty().into_array();
        let sizes = Buffer::<u32>::empty().into_array();
        let list = ListViewArray::try_new(elements, offsets, sizes, Validity::AllValid)?;
        let list = unsafe { list.with_zero_copy_to_list(true) };
        let list = list.into_array();

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
    fn test_export_list_non_ordered_elements() -> VortexResult<()> {
        let elements = buffer![0, 1, 2, 3, 4, 5].into_array();
        let offsets = buffer![1u8, 0, 3].into_array();
        let sizes = buffer![1u8, 1, 1].into_array();
        let list =
            ListViewArray::try_new(elements, offsets, sizes, Validity::AllValid)?.into_array();

        let list_type = LogicalType::list_type(LogicalType::int32())?;
        let mut chunk = DataChunk::new([list_type]);

        let mut ctx = SESSION.create_execution_ctx();
        new_array_exporter(list, &ConversionCache::default(), &mut ctx)?.export(
            0,
            3,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT INTEGER[]: 3 = [ [1], [0], [3]]
"#
        );

        Ok(())
    }

    #[test]
    fn test_export_non_empty_list_with_preceding_and_trailing_garbage() -> VortexResult<()> {
        let elements = buffer![0, 1, 2, 3, 4, 5].into_array();
        let offsets = buffer![1u8, 2, 3].into_array();
        let sizes = buffer![1u8, 1, 1].into_array();
        let list = ListViewArray::try_new(elements, offsets, sizes, Validity::AllValid)?;
        let list = unsafe { list.with_zero_copy_to_list(true) };
        let list = list.into_array();

        let list_type = LogicalType::list_type(LogicalType::int32())?;
        let mut chunk = DataChunk::new([list_type]);

        let mut ctx = SESSION.create_execution_ctx();
        new_array_exporter(list, &ConversionCache::default(), &mut ctx)?.export(
            0,
            3,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT INTEGER[]: 3 = [ [1], [2], [3]]
"#
        );

        Ok(())
    }

    #[test]
    fn test_export_string_list() -> VortexResult<()> {
        let elements = <VarBinArray as FromIterator<_>>::from_iter([
            Some("abc"),
            Some("def"),
            None,
            Some("ghi"),
        ])
        .into_array();
        let offsets = buffer![0u8, 0, 3, 4].into_array();
        let sizes = buffer![0u8, 3, 1, 0].into_array();
        let validities = Validity::from_iter([true, true, false, true]);
        let list = ListViewArray::try_new(elements, offsets, sizes, validities)?;
        let list = unsafe { list.with_zero_copy_to_list(true) };
        let list = list.into_array();

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
- FLAT VARCHAR[]: 4 = [ [], [abc, def, NULL], NULL, []]
"#
        );

        Ok(())
    }

    #[test]
    fn test_export_runend_list() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements_buffer = buffer![100f32, 101f32, 200f32, 202f32, 203f32].into_array();
        let elements = RunEnd::encode(elements_buffer, &mut ctx)?.into_array();

        let offsets = buffer![1u32, 2, 4].into_array();
        let sizes = buffer![1u8, 2, 1].into_array();
        let list =
            ListViewArray::try_new(elements, offsets, sizes, Validity::AllValid)?.into_array();

        let list_type = LogicalType::list_type(LogicalType::float32())?;
        let mut chunk = DataChunk::new([list_type]);

        new_array_exporter(list, &ConversionCache::default(), &mut ctx)?.export(
            0,
            3,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT FLOAT[]: 3 = [ [101.0], [200.0, 202.0], [203.0]]
"#
        );

        Ok(())
    }
}
