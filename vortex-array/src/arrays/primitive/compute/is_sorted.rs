use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};

use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::compute::IsSortedFn;
use crate::variants::PrimitiveArrayTrait;

impl IsSortedFn<&PrimitiveArray> for PrimitiveEncoding {
    fn is_sorted(&self, array: &PrimitiveArray, strict: bool) -> VortexResult<bool> {
        let v = match_each_native_ptype!(array.ptype(), |$P| {
            compute_is_sorted::<$P>(array.as_slice(), strict)
        });

        Ok(v)
    }
}

fn compute_is_sorted<T: NativePType>(slice: &[T], strict: bool) -> bool {
    let cmp_fn = if strict {
        T::is_gt
    } else {
        |a: T, b: T| a.total_compare(b).is_gt() && a.is_eq(b)
    };

    let mut iter = slice.iter().copied();
    let mut prev = iter.next().vortex_expect("Must have at least one item");

    for item in iter {
        if !cmp_fn(prev, item) {
            return false;
        }

        prev = item;
    }

    return true;
}

#[cfg(test)]
mod tests {}
