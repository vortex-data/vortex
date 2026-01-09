// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex::array::Array;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::ToCanonical;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::ConstantVTable;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::validity::Validity;
use vortex::array::vtable::ValidityHelper;
use vortex::compute;
use vortex::dtype::IntegerPType;
use vortex::dtype::match_each_integer_ptype;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::duckdb::LogicalType;
use crate::duckdb::ReusableDict;
use crate::duckdb::SelectionVector;
use crate::duckdb::Vector;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::cache::ConversionCache;
use crate::exporter::constant;
use crate::exporter::new_array_exporter;
use crate::exporter::new_operator_array_exporter;

struct DictExporter<I: IntegerPType> {
    // Store the dictionary values once and export the same dictionary with each codes chunk.
    values: ReusableDict,
    codes: PrimitiveArray,
    codes_type: PhantomData<I>,
}

pub(crate) fn new_exporter_with_flatten(
    array: &DictArray,
    cache: &ConversionCache,
    // Whether to return a duckdb flat vector or not.
    mut flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Grab the cache dictionary values.
    let values = array.values();
    let values_type: LogicalType = values.dtype().try_into()?;
    if let Some(constant) = values.as_opt::<ConstantVTable>() {
        return constant::new_exporter_with_mask(
            &ConstantArray::new(constant.scalar().clone(), array.codes().len()),
            array.codes().validity_mask(),
            cache,
        );
    }

    let codes_mask = array.codes().validity_mask();

    match codes_mask {
        Mask::AllTrue(_) => {}
        Mask::AllFalse(len) => return Ok(all_invalid::new_exporter(len, &values_type)),
        Mask::Values(_) => {
            // duckdb cannot have a dictionary with validity in the codes, so flatten the array and
            // apply the validity mask there.
            flatten = true;
        }
    }

    let values_key = Arc::as_ptr(values).addr();
    let codes = array.codes().to_primitive();

    let reusable_dict = if flatten {
        let canonical = cache
            .canonical_cache
            .get(&values_key)
            .map(|entry| entry.value().1.clone());
        let canonical = match canonical {
            Some(c) => c,
            None => {
                let canonical = values.to_canonical()?;
                cache
                    .canonical_cache
                    .insert(values_key, (values.clone(), canonical.clone()));
                canonical
            }
        };
        return new_array_exporter(&compute::take(canonical.as_ref(), codes.as_ref())?, cache);
    } else {
        // Check if we have a cached vector and extract it if we do.
        let reusable_dict = cache
            .dict_cache
            .get(&values_key)
            .map(|entry| entry.value().1.clone());

        match reusable_dict {
            Some(reusable_dict) => reusable_dict,
            None => {
                // Create a new reusable dictionary for the values.
                let reusable_dict = ReusableDict::new(values.dtype().try_into()?, values.len());
                let mut dict_vector = reusable_dict.vector();
                new_array_exporter(values, cache)?.export(0, values.len(), &mut dict_vector)?;

                cache
                    .dict_cache
                    .insert(values_key, (values.clone(), reusable_dict.clone()));

                reusable_dict
            }
        }
    };

    match_each_integer_ptype!(codes.ptype(), |I| {
        Ok(Box::new(DictExporter {
            values: reusable_dict,
            codes,
            codes_type: PhantomData::<I>,
        }))
    })
}

impl<I: IntegerPType + AsPrimitive<u32>> ColumnExporter for DictExporter<I> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
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

        vector.reuse_dictionary(&self.values, &sel_vec);

        Ok(())
    }
}

/// Operator-based exporter for dictionary arrays that uses ExecutionCtx.
struct DictOperatorExporter<I: IntegerPType> {
    // Store the dictionary values once and export the same dictionary with each codes chunk.
    values: ReusableDict,
    codes: PrimitiveArray,
    codes_type: PhantomData<I>,
}

pub(crate) fn new_operator_exporter_with_flatten(
    array: &DictArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    // Whether to return a duckdb flat vector or not.
    mut flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Grab the cache dictionary values.
    let values = array.values();
    let values_type: LogicalType = values.dtype().try_into()?;
    if let Some(constant) = values.as_opt::<ConstantVTable>() {
        return constant::new_exporter_with_mask(
            &ConstantArray::new(constant.scalar().clone(), array.codes().len()),
            array.codes().is_null()?.not()?.execute::<Mask>(ctx)?,
            cache,
        );
    }

    let codes = array
        .codes()
        .clone()
        .execute::<Canonical>(ctx)?
        .into_primitive();

    match codes.validity() {
        Validity::AllValid | Validity::NonNullable => {}
        Validity::AllInvalid => {
            return Ok(all_invalid::new_exporter(array.len(), &values_type));
        }
        Validity::Array(_) => {
            // duckdb cannot have a dictionary with validity in the codes, so flatten the array and
            // apply the validity mask there.
            flatten = true;
        }
    }

    let values_key = Arc::as_ptr(values).addr();

    let reusable_dict = if flatten {
        let canonical = cache
            .canonical_cache
            .get(&values_key)
            .map(|entry| entry.value().1.clone());
        let canonical = match canonical {
            Some(c) => c,
            None => {
                let canonical = values.to_canonical()?;
                cache
                    .canonical_cache
                    .insert(values_key, (values.clone(), canonical.clone()));
                canonical
            }
        };

        return new_operator_array_exporter(
            unsafe { DictArray::new_unchecked(codes.into_array(), canonical.into_array()) }
                .into_array()
                .execute::<Canonical>(ctx)?
                .into_array(),
            cache,
            ctx,
        );
    } else {
        // Check if we have a cached reusable dictionary and extract it if we do.
        let reusable_dict = cache
            .dict_cache
            .get(&values_key)
            .map(|entry| entry.value().1.clone());

        match reusable_dict {
            Some(reusable_dict) => reusable_dict,
            None => {
                // Create a new reusable dictionary for the values.
                let reusable_dict = ReusableDict::new(values.dtype().try_into()?, values.len());
                let mut dict_vector = reusable_dict.vector();
                new_operator_array_exporter(values.clone(), cache, ctx)?.export(
                    0,
                    values.len(),
                    &mut dict_vector,
                )?;

                cache
                    .dict_cache
                    .insert(values_key, (values.clone(), reusable_dict.clone()));

                reusable_dict
            }
        }
    };

    match_each_integer_ptype!(codes.ptype(), |I| {
        Ok(Box::new(DictOperatorExporter {
            values: reusable_dict,
            codes,
            codes_type: PhantomData::<I>,
        }))
    })
}

impl<I: IntegerPType + AsPrimitive<u32>> ColumnExporter for DictOperatorExporter<I> {
    fn export(&self, offset: usize, len: usize, vector: &mut Vector) -> VortexResult<()> {
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

        vector.reuse_dictionary(&self.values, &sel_vec);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::VortexSessionDefault;
    use vortex::array::ExecutionCtx;
    use vortex::array::IntoArray;
    use vortex::array::arrays::ConstantArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::buffer::Buffer;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::ColumnExporter;
    use crate::exporter::ConversionCache;
    use crate::exporter::dict::new_exporter_with_flatten;
    use crate::exporter::dict::new_operator_exporter_with_flatten;
    use crate::exporter::new_array_exporter;

    pub(crate) fn new_exporter(
        array: &DictArray,
        cache: &ConversionCache,
    ) -> VortexResult<Box<dyn ColumnExporter>> {
        new_exporter_with_flatten(array, cache, false)
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
            &mut chunk.get_vector(0),
        )?;
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&chunk)?),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 2 = [ NULL, 10]
"#
        );

        Ok(())
    }

    #[test]
    fn test_constant_dict_vector() -> VortexResult<()> {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None, Some(0u32)]).into_array(),
            ConstantArray::new(10, 1).into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        let mut ctx = ExecutionCtx::new(VortexSession::default());
        new_operator_exporter_with_flatten(&arr, &ConversionCache::default(), &mut ctx, false)?
            .export(0, 2, &mut chunk.get_vector(0))?;
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&chunk)?),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 2 = [ NULL, 10]
"#
        );

        Ok(())
    }

    #[test]
    fn test_constant_dict_vector_null() -> VortexResult<()> {
        let arr = DictArray::new(
            PrimitiveArray::from_option_iter([None::<u32>, None]).into_array(),
            ConstantArray::new(10, 1).into_array(),
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        let mut ctx = ExecutionCtx::new(VortexSession::default());
        new_operator_exporter_with_flatten(&arr, &ConversionCache::default(), &mut ctx, false)?
            .export(0, 2, &mut chunk.get_vector(0))?;
        chunk.set_len(2);

        assert_eq!(
            format!("{}", String::try_from(&chunk)?),
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
            &mut chunk.get_vector(0),
        )?;
        chunk.set_len(3);

        // some-invalid codes cannot be exported as a dictionary.
        assert_eq!(
            format!("{}", String::try_from(&chunk)?),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 3 = [ NULL, 10, NULL]
"#
        );

        let mut flat_chunk =
            DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_INTEGER)]);

        new_array_exporter(
            arr.to_canonical()
                .vortex_expect("to_canonical failed")
                .as_ref(),
            &ConversionCache::default(),
        )?
        .export(0, 3, &mut flat_chunk.get_vector(0))?;
        flat_chunk.set_len(3);

        assert_eq!(
            format!("{}", String::try_from(&flat_chunk)?),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 3 = [ NULL, 10, NULL]
"#
        );

        Ok(())
    }

    #[ignore = "TODO(connor)[4809]: Exporters do not correctly handle empty vectors"]
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
            &mut chunk.get_vector(0),
        )?;
        chunk.set_len(0);

        assert_eq!(
            format!("{}", String::try_from(&chunk)?),
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 0 = [ ]
"#
        );

        Ok(())
    }
}
