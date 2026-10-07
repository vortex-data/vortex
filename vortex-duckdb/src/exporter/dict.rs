// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use num_traits::AsPrimitive;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::Constant;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::array::match_each_integer_ptype;
use vortex::buffer::BitBuffer;
use vortex::dtype::DType;
use vortex::dtype::IntegerPType;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::mask::Mask;

use crate::duckdb::ReusableDict;
use crate::duckdb::SelectionVector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::cache::ConversionCache;
use crate::exporter::cached_values_dict;
use crate::exporter::cached_values_dict_with_null;
use crate::exporter::constant;
use crate::exporter::new_array_exporter;

struct DictExporter<I: IntegerPType> {
    // Store the dictionary values once and export the same dictionary with each codes chunk.
    values: ReusableDict,
    codes: PrimitiveArray,
    codes_validity: Option<BitBuffer>,
    null_index: u32,
    codes_type: PhantomData<I>,
}

pub(crate) fn new_exporter_with_flatten(
    array: &DictArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    // Whether to return a duckdb flat vector or not.
    flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Grab the cache dictionary values.
    let values = array.values();
    let codes = array.codes();
    let codes_len = codes.len();

    let codes_mask = codes.validity()?.execute_mask(codes_len, ctx)?;
    if matches!(codes_mask, Mask::AllFalse(_)) {
        return Ok(all_invalid::new_exporter());
    }

    let values_key = values.addr();

    if let Some(constant) = values.as_opt::<Constant>() {
        return constant::new_exporter_with_mask(
            ConstantArray::new(constant.scalar().clone(), codes_len),
            codes_mask,
            cache,
            ctx,
        );
    }

    // DuckDB dictionary vectors do not support STRUCT children.
    if flatten || matches!(values.dtype(), DType::Struct(..)) {
        let canonical = cache
            .canonical_cache
            .get(&values_key)
            .map(|entry| entry.value().1.clone());
        let canonical = match canonical {
            Some(c) => c,
            None => {
                let canonical = values.clone().execute::<Canonical>(ctx)?;
                cache
                    .canonical_cache
                    .insert(values_key, (values.clone(), canonical.clone()));
                canonical
            }
        };
        return new_array_exporter(
            DictArray::new(array.codes().clone(), canonical.into_array())
                .into_array()
                .execute::<Canonical>(ctx)?
                .into_array(),
            cache,
            ctx,
        );
    }

    let values_len = u32::try_from(values.len())
        .map_err(|_| vortex_err!("DuckDB dictionary length {} exceeds u32", values.len()))?;
    let codes = array.codes().clone().execute::<PrimitiveArray>(ctx)?;
    let (reusable_dict, codes_validity) = if codes_mask.all_true() {
        (cached_values_dict(values.clone(), cache, ctx)?, None)
    } else {
        (
            cached_values_dict_with_null(values.clone(), cache, ctx)?,
            Some(codes_mask.into_bit_buffer()),
        )
    };
    let null_index = values_len;

    match_each_integer_ptype!(codes.ptype(), |I| {
        Ok(Box::new(DictExporter {
            values: reusable_dict,
            codes,
            codes_validity,
            null_index,
            codes_type: PhantomData::<I>,
        }))
    })
}

impl<I: IntegerPType + AsPrimitive<u32>> ColumnExporter for DictExporter<I> {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        // Create a selection vector from the codes.
        let mut sel_vec = SelectionVector::with_capacity(len);
        let mut_sel_vec = unsafe { sel_vec.as_slice_mut(len) };
        let codes = &self.codes.as_slice::<I>()[offset..offset + len];
        if let Some(validity) = &self.codes_validity {
            assert!(offset + len <= validity.len());
            let validity = validity.slice(offset..offset + len);
            write_nullable_selection(mut_sel_vec, codes, &validity, self.null_index);
        } else {
            write_selection(mut_sel_vec, codes);
        }

        vector.reuse_dictionary(&self.values, &sel_vec);

        Ok(())
    }
}

#[inline]
fn write_selection<I: IntegerPType + AsPrimitive<u32>>(selection: &mut [u32], codes: &[I]) {
    debug_assert_eq!(selection.len(), codes.len());
    for (dst, src) in selection.iter_mut().zip(codes) {
        *dst = src.as_();
    }
}

#[inline]
fn write_nullable_selection<I: IntegerPType + AsPrimitive<u32>>(
    selection: &mut [u32],
    codes: &[I],
    validity: &BitBuffer,
    null_index: u32,
) {
    debug_assert_eq!(selection.len(), codes.len());
    debug_assert_eq!(selection.len(), validity.len());

    for ((validity_word, selection), codes) in validity
        .chunks()
        .iter_padded()
        .zip(selection.chunks_mut(64))
        .zip(codes.chunks(64))
    {
        let all_valid = u64::MAX >> (64 - selection.len());
        if validity_word == all_valid {
            write_selection(selection, codes);
        } else if validity_word == 0 {
            selection.fill(null_index);
        } else {
            // Materialize the denser side, then patch only the sparse exceptions.
            let mut patch_word;
            if validity_word.count_ones() as usize <= selection.len() / 2 {
                selection.fill(null_index);
                patch_word = validity_word;
                while patch_word != 0 {
                    let index = patch_word.trailing_zeros() as usize;
                    selection[index] = codes[index].as_();
                    patch_word &= patch_word - 1;
                }
            } else {
                write_selection(selection, codes);
                patch_word = !validity_word & all_valid;
                while patch_word != 0 {
                    let index = patch_word.trailing_zeros() as usize;
                    selection[index] = null_index;
                    patch_word &= patch_word - 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::ConstantArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::BitBuffer;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::ColumnExporter;
    use crate::exporter::ConversionCache;
    use crate::exporter::dict::new_exporter_with_flatten;
    use crate::exporter::dict::write_nullable_selection;

    pub(crate) fn new_exporter(
        array: &DictArray,
        cache: &ConversionCache,
    ) -> VortexResult<Box<dyn ColumnExporter>> {
        new_exporter_with_flatten(array, cache, &mut SESSION.create_execution_ctx(), false)
    }

    #[test]
    fn test_constant_dict() -> VortexResult<()> {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None, Some(0u32)]).into_array(),
            ConstantArray::new(10, 1).into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr, &ConversionCache::default())?.export(
            0,
            2,
            chunk.get_vector_mut(0),
            &mut SESSION.create_execution_ctx(),
        )?;
        chunk.set_len(2);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 2 = [ NULL, 10]
"#
        );

        Ok(())
    }

    #[test]
    fn test_constant_dict_null() -> VortexResult<()> {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None::<u32>, None]).into_array(),
            ConstantArray::new(10, 1).into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        let mut ctx = VortexSession::default().create_execution_ctx();
        new_exporter_with_flatten(&arr, &ConversionCache::default(), &mut ctx, false)?.export(
            0,
            2,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(2);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- CONSTANT INTEGER: 2 = [ NULL]
"#
        );

        Ok(())
    }

    #[test]
    fn test_nullable_dict() -> VortexResult<()> {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None, Some(0u32), Some(1)]).into_array(),
            PrimitiveArray::from_option_iter([Some(10), None]).into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr, &ConversionCache::default())?.export(
            0,
            3,
            chunk.get_vector_mut(0),
            &mut SESSION.create_execution_ctx(),
        )?;
        chunk.set_len(3);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 3 = [ NULL, 10, NULL]
"#
        );

        let mut flat_chunk =
            DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);
        let mut ctx = SESSION.create_execution_ctx();

        new_exporter_with_flatten(&arr, &ConversionCache::default(), &mut ctx, true)?.export(
            0,
            3,
            flat_chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        flat_chunk.set_len(3);

        assert_eq!(
            String::try_from(&*flat_chunk)?,
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 3 = [ NULL, 10, NULL]
"#
        );

        Ok(())
    }

    #[test]
    fn test_nullable_string_dict_slice() -> VortexResult<()> {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([Some(0u32), None, Some(1), None, Some(0)])
                .into_array(),
            VarBinViewArray::from_iter_str(["ten", "twenty"]).into_array(),
        );
        let cache = ConversionCache::default();
        let mut chunk = DataChunk::new([LogicalType::varchar()]);

        new_exporter(&arr, &cache)?.export(
            1,
            3,
            chunk.get_vector_mut(0),
            &mut SESSION.create_execution_ctx(),
        )?;
        chunk.set_len(3);

        assert_eq!(cache.nullable_dict_cache.len(), 1);
        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- DICTIONARY VARCHAR: 3 = [ NULL, twenty, NULL]
"#
        );
        Ok(())
    }

    #[test]
    fn test_all_valid_codes_reuse_existing_nullable_dictionary() -> VortexResult<()> {
        let values = VarBinViewArray::from_iter_str(["ten", "twenty"]).into_array();
        let mixed = DictArray::new(
            PrimitiveArray::from_option_iter([Some(0u32), None]).into_array(),
            values.clone(),
        );
        let all_valid = DictArray::new(PrimitiveArray::from_iter([1u32, 0]).into_array(), values);
        let cache = ConversionCache::default();

        new_exporter(&mixed, &cache)?;
        new_exporter(&all_valid, &cache)?;

        assert_eq!(cache.nullable_dict_cache.len(), 1);
        assert!(cache.dict_cache.is_empty());
        Ok(())
    }

    #[test]
    fn test_nullable_dict_at_validity_word_boundary() -> VortexResult<()> {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None, Some(63u32)]).into_array(),
            PrimitiveArray::from_iter(0i32..64).into_array(),
        );
        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr, &ConversionCache::default())?.export(
            0,
            2,
            chunk.get_vector_mut(0),
            &mut SESSION.create_execution_ctx(),
        )?;
        chunk.set_len(2);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 2 = [ NULL, 63]
"#
        );
        Ok(())
    }

    #[test]
    fn test_nullable_selection_word_paths_with_unaligned_validity() {
        let validity = BitBuffer::from_iter(
            std::iter::once(false)
                .chain(std::iter::repeat_n(true, 64))
                .chain(std::iter::repeat_n(false, 64))
                .chain([true, false, true, false, true, false, true, false]),
        )
        .slice(1..137);
        let codes = (0..136u16).collect::<Vec<_>>();
        let mut selection = vec![u32::MAX; codes.len()];

        write_nullable_selection(&mut selection, &codes, &validity, 999);

        assert_eq!(&selection[..64], &(0..64u32).collect::<Vec<_>>());
        assert_eq!(&selection[64..128], &[999; 64]);
        assert_eq!(&selection[128..], &[128, 999, 130, 999, 132, 999, 134, 999]);
    }

    #[test]
    fn test_nullable_dict_with_all_null_values() -> VortexResult<()> {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([Some(0u32), None, Some(1)]).into_array(),
            PrimitiveArray::from_option_iter([None::<i32>, None]).into_array(),
        );
        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr, &ConversionCache::default())?.export(
            0,
            3,
            chunk.get_vector_mut(0),
            &mut SESSION.create_execution_ctx(),
        )?;
        chunk.set_len(3);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 3 = [ NULL, NULL, NULL]
"#
        );
        Ok(())
    }

    #[test]
    fn test_invalid_null_code_payload_is_ignored() -> VortexResult<()> {
        let arr = DictArray::new(
            PrimitiveArray::new(
                buffer![u64::MAX, 0],
                Validity::from(BitBuffer::from_iter([false, true])),
            )
            .into_array(),
            PrimitiveArray::from_iter([10i32]).into_array(),
        );
        let mut chunk = DataChunk::new([LogicalType::int32()]);

        new_exporter(&arr, &ConversionCache::default())?.export(
            0,
            2,
            chunk.get_vector_mut(0),
            &mut SESSION.create_execution_ctx(),
        )?;
        chunk.set_len(2);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 2 = [ NULL, 10]
"#
        );
        Ok(())
    }

    #[test]
    fn test_nullable_struct_dict_falls_back_to_flat() -> VortexResult<()> {
        let values =
            StructArray::from_fields(&[("a", PrimitiveArray::from_iter([10i32]).into_array())])?;
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([Some(0u32), None]).into_array(),
            values.into_array(),
        );
        let mut chunk = DataChunk::new([LogicalType::struct_type(
            vec![LogicalType::int32()],
            vec![c"a".to_owned()],
        )
        .vortex_expect("valid struct logical type")]);

        new_exporter(&arr, &ConversionCache::default())?.export(
            0,
            2,
            chunk.get_vector_mut(0),
            &mut SESSION.create_execution_ctx(),
        )?;
        chunk.set_len(2);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- FLAT STRUCT(a INTEGER): 2 = [ {'a': 10}, NULL]
"#
        );
        Ok(())
    }

    #[test]
    fn test_export_empty_dict() -> VortexResult<()> {
        let arr = DictArray::new(
            Buffer::<u32>::empty().into_array(),
            Buffer::<u32>::empty().into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_exporter(&arr, &ConversionCache::default())?.export(
            0,
            0,
            chunk.get_vector_mut(0),
            &mut SESSION.create_execution_ctx(),
        )?;
        chunk.set_len(0);

        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 0 = [ ]
"#
        );

        Ok(())
    }
}
