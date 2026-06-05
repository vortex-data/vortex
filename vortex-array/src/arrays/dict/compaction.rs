// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compaction support for [`Dict`] arrays.
//!
//! When a dictionary is the child of a [`Compaction`](crate::arrays::Compaction) array, we can
//! often avoid a full decode. A dictionary that has accumulated unreferenced ("dead") values
//! after slicing/taking can either be:
//!
//! - left alone, if every value is still referenced and the dictionary still compresses;
//! - garbage collected in place, dropping dead values and remapping codes, keeping the data
//!   dictionary-encoded; or
//! - decoded to a flat canonical array, when the dictionary no longer earns its indirection.
//!
//! We pick the cheapest option with a simple heuristic and let the fallback path handle the flat
//! decode.

use num_traits::FromPrimitive;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::compaction::CompactKernel;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::match_each_integer_ptype;

impl CompactKernel for Dict {
    fn compact(
        array: ArrayView<'_, Dict>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let values = array.values();
        let codes = array.codes();
        let num_values = values.len();
        let num_codes = codes.len();

        // Which values are referenced by at least one (valid) code.
        let referenced = array.compute_referenced_values_mask(true)?;
        let num_referenced = referenced.iter().filter(|r| *r).count();

        // If every code maps to a distinct value the dictionary provides no compression, so a flat
        // decode is at least as good. Decline and let the fallback decode to canonical.
        if num_referenced >= num_codes {
            return Ok(None);
        }

        // The dictionary still compresses. If there are no dead values it is already compact, so
        // keep it as-is (and record that all values are referenced for downstream kernels).
        if num_referenced == num_values {
            // SAFETY: we just verified that every value is referenced.
            let dict = unsafe {
                array
                    .array()
                    .clone()
                    .downcast::<Dict>()
                    .set_all_values_referenced(true)
            };
            return Ok(Some(dict.into_array()));
        }

        // Otherwise garbage collect: drop dead values and remap the codes.
        let codes_ptype = codes.dtype().as_ptype();
        let remap = match_each_integer_ptype!(codes_ptype, |P| {
            let mut new_index: usize = 0;
            let mut remap = Vec::<P>::with_capacity(num_values);
            for is_referenced in referenced.iter() {
                // Unreferenced slots are never indexed by a code, so their value is irrelevant.
                remap.push(
                    <P as FromPrimitive>::from_usize(new_index)
                        .vortex_expect("compacted dictionary index does not fit in code type"),
                );
                if is_referenced {
                    new_index += 1;
                }
            }
            PrimitiveArray::from_iter(remap).into_array()
        });

        // `take` preserves the nullability of `codes`, so null codes stay null.
        let new_codes = remap.take(codes.clone())?;
        let new_values = values.filter(Mask::from(referenced))?;

        // SAFETY: `new_codes` index into `new_values` by construction, and we have dropped exactly
        // the unreferenced values, so all remaining values are referenced.
        let dict = unsafe {
            DictArray::new_unchecked(new_codes, new_values).set_all_values_referenced(true)
        };
        Ok(Some(dict.into_array()))
    }
}
