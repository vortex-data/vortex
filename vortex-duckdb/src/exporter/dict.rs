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
use vortex::dtype::IntegerPType;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::duckdb::ReusableDict;
use crate::duckdb::SelectionVector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::cache::ConversionCache;
use crate::exporter::cached_values_dict;
use crate::exporter::constant;
use crate::exporter::new_array_exporter;

struct DictExporter<I: IntegerPType> {
    // Store the dictionary values once and export the same dictionary with each codes chunk.
    values: ReusableDict,
    codes: PrimitiveArray,
    codes_type: PhantomData<I>,
}

pub(crate) fn new_exporter_with_flatten(
    array: &DictArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    // Whether to return a duckdb flat vector or not.
    mut flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Grab the cache dictionary values.
    let values = array.values();
    let codes = array.codes();
    let codes_len = codes.len();
    tracing::debug!(
        cache = ?std::ptr::from_ref(cache),
        key = values.addr(),
        rows = codes_len,
        values = values.len(),
        encoding = %values.encoding_id(),
        flatten,
        "Q28 dictionary input"
    );

    let validity = codes.validity()?;
    if let Some(constant) = values.as_opt::<Constant>() {
        return constant::new_exporter_with_validity(
            ConstantArray::new(constant.scalar().clone(), codes_len),
            validity,
            cache,
            ctx,
        );
    }

    match validity.execute_mask(codes_len, ctx)? {
        Mask::AllTrue(_) => {}
        Mask::AllFalse(_) => return Ok(all_invalid::new_exporter()),
        Mask::Values(_) => {
            // duckdb cannot have a dictionary with validity in the codes, so flatten the array and
            // apply the validity mask there.
            flatten = true;
        }
    }

    // Sparse string projections can gather compressed values before decoding. Exporting the
    // whole dictionary here would undo that saving, especially after a selective filter.
    if (values.dtype().is_utf8() || values.dtype().is_binary())
        && values.len() >= 1024
        && codes_len.saturating_mul(16) < values.len()
        && !cache.dict_cache.contains_key(&values.addr())
    {
        return new_array_exporter(
            array
                .clone()
                .into_array()
                .execute::<Canonical>(ctx)?
                .into_array(),
            cache,
            ctx,
        );
    }

    let values_key = values.addr();
    let codes = array.codes().clone().execute::<PrimitiveArray>(ctx)?;

    if flatten {
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

    let reusable_dict = cached_values_dict(values.clone(), cache, ctx)?;

    match_each_integer_ptype!(codes.ptype(), |I| {
        Ok(Box::new(DictExporter {
            values: reusable_dict,
            codes,
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
        for (dst, src) in mut_sel_vec.iter_mut().zip(
            self.codes.as_slice::<I>()[offset..offset + len]
                .iter()
                .map(|v| v.as_()),
        ) {
            *dst = src
        }

        vector.reuse_dictionary(&self.values, &sel_vec, len);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::iter::repeat_n;
    use std::panic::resume_unwind;
    use std::sync::Barrier;
    use std::thread;

    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::ConstantArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::buffer::Buffer;
    use vortex::encodings::fsst::fsst_compress;
    use vortex::encodings::fsst::fsst_train_compressor;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::duckdb::SelectionVector;
    use crate::exporter::ColumnExporter;
    use crate::exporter::ConversionCache;
    use crate::exporter::cached_values_dict;
    use crate::exporter::dict::new_exporter_with_flatten;
    use crate::exporter::new_array_exporter;

    pub(crate) fn new_exporter(
        array: &DictArray,
        cache: &ConversionCache,
    ) -> VortexResult<Box<dyn ColumnExporter>> {
        new_exporter_with_flatten(array, cache, &mut SESSION.create_execution_ctx(), false)
    }

    #[rstest]
    fn sparse_string_dictionary(
        #[values(false, true)] compressed: bool,
        #[values(false, true)] nullable_codes: bool,
    ) -> VortexResult<()> {
        let mut strings = vec![Some("unused dictionary value"); 1024];
        strings[0] = Some("first value");
        strings[2] = None;
        strings[1023] = Some("last value");
        let mut values = VarBinViewArray::from_iter_nullable_str(strings).into_array();
        let mut ctx = SESSION.create_execution_ctx();
        if compressed {
            let compressor = fsst_train_compressor(&values, &mut ctx)?;
            values = fsst_compress(&values, &compressor, &mut ctx)?.into_array();
        }
        let array = DictArray::new(
            PrimitiveArray::from_option_iter([
                Some(0u32),
                Some(2),
                (!nullable_codes).then_some(3),
                Some(1023),
            ])
            .into_array(),
            values,
        );
        let mut chunk = DataChunk::new([LogicalType::varchar()]);
        new_exporter_with_flatten(&array, &ConversionCache::default(), &mut ctx, false)?.export(
            0,
            4,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(4);
        let middle = if nullable_codes {
            "NULL"
        } else {
            "unused dictionary value"
        };
        assert_eq!(
            String::try_from(&*chunk)?,
            format!(
                "Chunk - [1 Columns]\n- FLAT VARCHAR: 4 = [ first value, NULL, {middle}, last value]\n"
            )
        );
        Ok(())
    }

    #[test]
    fn concurrent_dictionary_exports_share_values() -> VortexResult<()> {
        let len = 262_144;
        let values =
            VarBinViewArray::from_iter_str(repeat_n("shared dictionary value", len)).into_array();
        let cache = ConversionCache::default();
        let barrier = Barrier::new(8);
        let mut dictionaries = thread::scope(|scope| {
            let workers = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        let mut ctx = SESSION.create_execution_ctx();
                        barrier.wait();
                        cached_values_dict(values.clone(), &cache, &mut ctx)
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| match worker.join() {
                    Ok(result) => result,
                    Err(payload) => resume_unwind(payload),
                })
                .collect::<VortexResult<Vec<_>>>()
        })?;

        let expected_vector = dictionaries[0].vector().as_ptr();
        let mut selection = SelectionVector::with_capacity(2);
        // Both indices are within the dictionary and the selection has capacity for two rows.
        let last = u32::try_from(len - 1).vortex_expect("dictionary index fits in u32");
        unsafe { selection.as_slice_mut(2) }.copy_from_slice(&[0, last]);
        for dict in &mut dictionaries {
            assert_eq!(dict.vector().as_ptr(), expected_vector);
            let mut chunk = DataChunk::new([LogicalType::varchar()]);
            chunk
                .get_vector_mut(0)
                .reuse_dictionary(dict, &selection, 2);
            chunk.set_len(2);
            assert_eq!(
                String::try_from(&*chunk)?,
                "Chunk - [1 Columns]\n- DICTIONARY VARCHAR: 2 = [ shared dictionary value, shared dictionary value]\n"
            );
        }
        Ok(())
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

        let mut flat_chunk =
            DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);
        let mut ctx = SESSION.create_execution_ctx();

        new_array_exporter(arr.into_array(), &ConversionCache::default(), &mut ctx)?.export(
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
