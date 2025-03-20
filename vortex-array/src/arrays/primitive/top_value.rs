use std::hash::Hash;

use rustc_hash::FxBuildHasher;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_mask::{AllOr, Mask};
use vortex_scalar::PValue;

use crate::Array;
use crate::aliases::hash_map::HashMap;
use crate::arrays::{NativeValue, PrimitiveArray};
use crate::variants::PrimitiveArrayTrait;

impl PrimitiveArray {
    pub fn top_value(&self) -> VortexResult<(PValue, u32)> {
        if self.is_empty() {
            vortex_bail!("Can't compute top value for empty array")
        }

        if self.all_invalid()? {
            vortex_bail!("Can't compute top value for all null array")
        }

        match_each_native_ptype!(self.ptype(), |$P| {
            typed_top_value(self.as_slice::<$P>(), self.validity_mask()?).map(|(v, c)|  (v.into(), c))
        })
    }
}

fn typed_top_value<T>(values: &[T], mask: Mask) -> VortexResult<(T, u32)>
where
    T: NativePType,
    NativeValue<T>: Eq + Hash,
{
    let mut distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher> =
        HashMap::with_hasher(FxBuildHasher);
    match mask.boolean_buffer() {
        AllOr::All => {
            for value in values.iter().copied() {
                *distinct_values.entry(NativeValue(value)).or_insert(0) += 1;
            }
        }
        AllOr::None => unreachable!("All invalid arrays should be handled earlier"),
        AllOr::Some(b) => {
            for (idx, value) in values.iter().copied().enumerate() {
                if b.value(idx) {
                    *distinct_values.entry(NativeValue(value)).or_insert(0) += 1;
                }
            }
        }
    }

    let (&top_value, &top_count) = distinct_values
        .iter()
        .max_by_key(|&(_, &count)| count)
        .vortex_expect("non-empty");
    Ok((top_value.0, top_count))
}
