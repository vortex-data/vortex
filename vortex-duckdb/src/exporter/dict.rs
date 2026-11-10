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
use vortex::error::vortex_ensure;
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
    values_len: usize,
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

    let values_len = values.len();
    let reusable_dict = cached_values_dict(values.clone(), cache, ctx)?;

    match_each_integer_ptype!(codes.ptype(), |I| {
        Ok(Box::new(DictExporter {
            values: reusable_dict,
            values_len,
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
        let codes = &self.codes.as_slice::<I>()[offset..offset + len];

        // A negative code becomes `usize::MAX`, so the bounds check below rejects it.
        let mut max_code = 0;
        for (dst, code) in mut_sel_vec.iter_mut().zip(codes) {
            max_code = max_code.max(code.to_usize().unwrap_or(usize::MAX));
            *dst = code.as_();
        }

        // DuckDB does not check the selection vector of a dictionary vector, so every code must
        // be in bounds of the values and fit in a u32.
        vortex_ensure!(
            len == 0 || (max_code < self.values_len && u32::try_from(max_code).is_ok()),
            "Dictionary code {max_code} out of bounds for {} values",
            self.values_len
        );

        vector.reuse_dictionary(&self.values, &sel_vec, len);

        Ok(())
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
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::ColumnExporter;
    use crate::exporter::ConversionCache;
    use crate::exporter::dict::new_exporter_with_flatten;
    use crate::exporter::new_array_exporter;

    pub(crate) fn new_exporter(
        array: &DictArray,
        cache: &ConversionCache,
    ) -> VortexResult<Box<dyn ColumnExporter>> {
        new_exporter_with_flatten(array, cache, &mut SESSION.create_execution_ctx(), false)
    }

    #[test]
    fn test_out_of_bounds_codes_fail() -> VortexResult<()> {
        // SAFETY: The code 5 is out of bounds on purpose, because the exporter must reject it.
        let arr = unsafe {
            DictArray::new_unchecked(
                buffer![0u32, 5].into_array(),
                buffer![10i32, 20].into_array(),
            )
        };

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);
        let result = new_exporter(&arr, &ConversionCache::default())?.export(
            0,
            2,
            chunk.get_vector_mut(0),
            &mut SESSION.create_execution_ctx(),
        );

        assert!(result.is_err_and(|err| err.to_string().contains("out of bounds for 2 values")));
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
