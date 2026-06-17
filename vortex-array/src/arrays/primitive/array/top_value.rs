// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use rustc_hash::FxBuildHasher;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::NativeValue;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar::PValue;

impl PrimitiveArray {
    /// Compute most common present value of this array
    pub fn top_value(&self) -> VortexResult<Option<(PValue, usize)>> {
        if self.is_empty() {
            return Ok(None);
        }

        if self.validity()?.definitely_all_null() {
            return Ok(None);
        }

        match_each_native_ptype!(self.ptype(), |P| {
            let (top, count) = typed_top_value(
                self.as_slice::<P>(),
                self.as_ref().validity()?.execute_mask(
                    self.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx(),
                )?,
            );
            Ok(Some((top.into(), count)))
        })
    }
}

fn typed_top_value<T>(values: &[T], mask: Mask) -> (T, usize)
where
    T: NativePType,
    NativeValue<T>: Eq + Hash,
{
    let mut distinct_values: HashMap<NativeValue<T>, usize, FxBuildHasher> =
        HashMap::with_hasher(FxBuildHasher);
    match mask.indices() {
        AllOr::All => {
            for value in values.iter().copied() {
                *distinct_values.entry(NativeValue(value)).or_insert(0) += 1;
            }
        }
        AllOr::None => unreachable!("All invalid arrays should be handled earlier"),
        AllOr::Some(idxs) => {
            for &i in idxs {
                *distinct_values
                    .entry(NativeValue(unsafe { *values.get_unchecked(i) }))
                    .or_insert(0) += 1
            }
        }
    }

    let (&top_value, &top_count) = distinct_values
        .iter()
        .max_by_key(|&(_, &count)| count)
        .vortex_expect("non-empty");
    (top_value.0, top_count)
}
