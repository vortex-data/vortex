// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::collect_bits;
use crate::scalar_fn::fns::list_contains::ListContainsElementKernel;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::ListContainsSet;

/// Single-pass membership of primitive needles in a constant set.
///
/// The set is sorted with the total order and probed by binary search, so a float needle is a
/// member exactly when the compare kernel would call it equal to an element: `total_compare` is
/// `Equal` precisely when the bit patterns match, distinguishing `-0.0` from `0.0` and one NaN
/// payload from another, as `is_eq` does.
impl ListContainsElementKernel for Primitive {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        options: &ListContainsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(set) = ListContainsSet::try_new(list, element.dtype(), options) else {
            return Ok(None);
        };

        let bits = match_each_native_ptype!(element.ptype(), |T| {
            let sorted = sorted_set::<T>(set.elements());
            collect_bits(
                element.as_slice::<T>(),
                |value| {
                    sorted
                        .binary_search_by(|element| element.total_compare(value))
                        .is_ok()
                },
                ctx.allocator(),
            )
        });

        set.finish(bits, element.validity()?).map(Some)
    }
}

/// The non-null elements as a sorted, de-duplicated `Vec<T>` for binary search.
fn sorted_set<T: NativePType>(elements: &[Scalar]) -> Vec<T> {
    let mut set: Vec<T> = elements
        .iter()
        .filter_map(|element| element.as_primitive().typed_value::<T>())
        .collect();
    set.sort_unstable_by(|a, b| a.total_compare(*b));
    set.dedup_by(|a, b| a.is_eq(*b));
    set
}
