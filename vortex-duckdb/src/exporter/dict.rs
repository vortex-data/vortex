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

    // A dictionary vector needs every value converted, even when only a handful of codes
    // are selected. Let encoded take kernels gather these rows before canonicalization.
    if cache.sparse_dictionaries
        && codes_len > 0
        && values.len() >= 1024
        && codes_len.saturating_mul(16) < values.len()
        && !cache.dict_cache.contains_key(&values_key)
        && !cache.canonical_cache.contains_key(&values_key)
    {
        let selected = DictArray::new(codes.into_array(), values.clone())
            .into_array()
            .execute::<Canonical>(ctx)?;
        tracing::debug!(
            dictionary_values = values.len(),
            selected_rows = codes_len,
            "v2 sparse dictionary export"
        );
        return new_array_exporter(selected.into_array(), cache, ctx);
    }

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
mod tests;
