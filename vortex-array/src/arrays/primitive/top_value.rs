use std::hash::Hash;

use rustc_hash::FxBuildHasher;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{AllOr, Mask};
use vortex_scalar::PValue;

use crate::Array;
use crate::aliases::hash_map::HashMap;
use crate::arrays::{NativeValue, PrimitiveArray};
use crate::variants::PrimitiveArrayTrait;

impl PrimitiveArray {
    /// Compute most common present value of this array
    pub fn top_value(&self) -> VortexResult<Option<(PValue, usize)>> {
        if self.is_empty() {
            return Ok(None);
        }

        if self.all_invalid()? {
            return Ok(None);
        }

        match_each_native_ptype!(self.ptype(), |$P| {
            let (top, count) = typed_top_value(self.as_slice::<$P>(), self.validity_mask()?);
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
