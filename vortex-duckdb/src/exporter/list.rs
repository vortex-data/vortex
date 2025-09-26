// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use itertools::Itertools as _;
use vortex::ToCanonical as _;
use vortex::arrays::{ListArray, PrimitiveArray};
use vortex::dtype::{IntegerPType, match_each_integer_ptype};
use vortex::error::{VortexResult, vortex_err};
use vortex::mask::Mask;

use super::{ConversionCache, new_array_exporter_with_flatten};
use crate::cpp;
use crate::duckdb::Vector;
use crate::exporter::ColumnExporter;

struct ListExporter<T> {
    validity: Mask,
    elements_exporter: Box<dyn ColumnExporter>,
    offsets: PrimitiveArray,
    offset_type: PhantomData<T>,
}

pub(crate) fn new_exporter(
    array: &ListArray,
    cache: &ConversionCache,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let elements_exporter = new_array_exporter_with_flatten(array.elements(), cache, true)?;
    let offsets = array.offsets().to_primitive();
    let boxed = match_each_integer_ptype!(offsets.ptype(), |T| {
        Box::new(ListExporter {
            validity: array.validity_mask(),
            elements_exporter,
            offsets,
            offset_type: PhantomData::<T>,
        }) as Box<dyn ColumnExporter>
    });
    Ok(boxed)
}

impl<T: IntegerPType> ColumnExporter for ListExporter<T> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
        // Set validity if necessary.
        if unsafe { vector.set_validity(&self.validity, offset, len) } {
            // All values are null, so no point copying the data.
            return Ok(());
        }

        let offsets = &self.offsets.as_slice::<T>()[offset..][..(len + 1)];
        let start_offset = offsets[0]
            .to_u64()
            .ok_or_else(|| vortex_err!("list offsets must fit in u64"))?;
        let mut sum_of_list_lens = 0_u64;

        let offsets_slice: &mut [cpp::duckdb_list_entry] =
            unsafe { vector.as_slice_mut::<cpp::duckdb_list_entry>(len) };

        for (window, destination) in offsets.windows(2).zip_eq(offsets_slice) {
            let start = window[0]
                .to_u64()
                .ok_or_else(|| vortex_err!("list offsets must fit in u64"))?;
            let end = window[1]
                .to_u64()
                .ok_or_else(|| vortex_err!("list offsets must fit in u64"))?;
            let len = end - start;
            sum_of_list_lens += len;
            *destination = cpp::duckdb_list_entry {
                offset: start - start_offset,
                length: len,
            };
        }

        // TODO(DK): This calls `list_vector_reserve` once for each call to `export`. Moreover, we
        // copy slices of the elements once for each call to `export`. Would `export` be faster if
        // we copied all the elements on the first call to `export` so that subsequent calls need
        // only copy the correct slice of offsets?
        vector.list_vector_reserve(sum_of_list_lens)?;
        let mut elements_vector = vector.list_vector_get_child();
        vector.list_vector_set_size(sum_of_list_lens)?;
        self.elements_exporter.export(
            usize::try_from(start_offset)?,
            usize::try_from(sum_of_list_lens)?,
            &mut elements_vector,
        )
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
    fn test_export_empty_list() {
        let list = ListArray::try_new(
            Buffer::<u32>::empty().into_array(),
            buffer![0u8].into_array(),
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
        let list = ListArray::try_new(
            buffer![0, 1, 2, 3, 4, 5].into_array(),
            buffer![1u8, 2, 3, 4].into_array(),
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
        let list = ListArray::try_new(
            <VarBinArray as FromIterator<_>>::from_iter([
                Some("abc"),
                Some("def"),
                None,
                Some("ghi"),
            ])
            .into_array(),
            buffer![0u8, 0, 3, 4, 4].into_array(),
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
