// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex::arrays::{ListViewArray, PrimitiveArray};
use vortex::dtype::match_each_integer_ptype;
use vortex::error::{VortexResult, vortex_err};
use vortex::mask::Mask;
use vortex::{OffsetPType, ToCanonical as _};

use super::{ConversionCache, new_array_exporter_with_flatten};
use crate::cpp;
use crate::duckdb::Vector;
use crate::exporter::ColumnExporter;

struct ListExporter<O, S> {
    validity: Mask,
    /// We cache the child elements of our list array so that we don't have to export it every time.
    duckdb_elements: Vector,
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
    let offsets = array.offsets().to_primitive();
    let sizes = array.sizes().to_primitive();

    // Create a duckdb elements vector up front so that future exports can reference it.
    let elements = array.elements();

    let mut duckdb_elements = Vector::with_capacity(elements.dtype().try_into()?, elements.len());
    let elements_exporter = new_array_exporter_with_flatten(array.elements(), cache, true)?;
    elements_exporter.export(0, elements.len(), &mut duckdb_elements)?;

    let boxed = match_each_integer_ptype!(offsets.ptype(), |O| {
        match_each_integer_ptype!(sizes.ptype(), |S| {
            Box::new(ListExporter {
                validity: array.validity_mask(),
                duckdb_elements,
                offsets,
                sizes,
                num_elements: elements.len(),
                offset_type: PhantomData::<O>,
                size_type: PhantomData::<S>,
            }) as Box<dyn ColumnExporter>
        })
    });

    Ok(boxed)
}

impl<O: OffsetPType, S: OffsetPType> ColumnExporter for ListExporter<O, S> {
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
        child.reference(&self.duckdb_elements);

        vector.list_vector_set_size(self.num_elements as u64)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::IntoArray as _;
    use vortex::arrays::VarBinArray;
    use vortex::buffer::{Buffer, buffer};
    use vortex::validity::Validity;

    use super::*;
    use crate::cpp;
    use crate::duckdb::{DataChunk, LogicalType};
    use crate::exporter::new_array_exporter;

    #[test]
    #[ignore = "TODO(connor): Exporters do not correctly handle empty vectors"]
    fn test_export_empty_list() {
        let list = ListViewArray::try_new(
            Buffer::<u32>::empty().into_array(),
            Buffer::<u32>::empty().into_array(),
            Buffer::<u32>::empty().into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let list_type = LogicalType::new_list(cpp::duckdb_type::DUCKDB_TYPE_INTEGER);
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
        let list = ListViewArray::try_new(
            buffer![0, 1, 2, 3, 4, 5].into_array(),
            buffer![1u8, 2, 3].into_array(),
            buffer![1u8, 1, 1].into_array(),
            Validity::AllValid,
        )
        .unwrap()
        .into_array();

        let list_type = LogicalType::new_list(cpp::duckdb_type::DUCKDB_TYPE_INTEGER);
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
        let list = ListViewArray::try_new(
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
        .unwrap()
        .into_array();

        let list_type = LogicalType::new_list(cpp::duckdb_type::DUCKDB_TYPE_VARCHAR);
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
