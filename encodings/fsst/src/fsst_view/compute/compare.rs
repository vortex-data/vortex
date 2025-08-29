// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::{BoolArray, BooleanBuffer, BooleanBufferBuilder, ConstantArray};
use vortex_array::compute::{CompareKernel, CompareKernelAdapter, Operator};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::fsst_view::MAX_INLINE_STR;
use crate::{FSSTViewArray, FSSTViewVTable, View};

enum MatchType {
    /// The operator evaluates to false for the predicate
    False,
    /// The operator evaluates to true for the predicate
    True,
    /// The operation cannot be exactly determined just from examining the view.
    Maybe,
}

// Compare inline, when we know that the needle fits in an inlined `View`.
// This is a fast path where we can do straight-line, fixed-width comparisons without touching
// or decoding the string buffer.
fn compare_inline<F>(needle_view: View, haystack: &FSSTViewArray, cmp: F) -> BooleanBuffer
where
    F: Fn(View, View) -> bool,
{
    // Create one for every boolean buffer instead.
    let mut result = BooleanBufferBuilder::new(haystack.len());

    for &view in haystack.views.iter() {
        result.append(cmp(needle_view, view));
    }

    result.finish()
}

fn compare_outlined<V, F>(
    needle: &[u8],
    haystack: &FSSTViewArray,
    cmp_view: V,
    cmp_full: F,
) -> BooleanBuffer
where
    V: Fn(View, View) -> MatchType,
    F: Fn(&[u8], &[u8]) -> bool,
{
    let mut result = BooleanBufferBuilder::new(haystack.len());

    // use dummy buffer index of zero, comparison functions will chop off the `index` component
    let needle_view = View::new_outlined(needle, 0);

    for (index, &view) in haystack.views().iter().enumerate() {
        match cmp_view(needle_view, view) {
            MatchType::False => result.append(false),
            MatchType::True => result.append(true),
            MatchType::Maybe => {
                if haystack.is_valid(index) {
                    let full = haystack.bytes_at(index);
                    result.append(cmp_full(needle, full.as_ref()))
                } else {
                    // Null value, doesn't matter anyway
                    result.append(false);
                }
            }
        }
    }

    result.finish()
}

impl CompareKernel for FSSTViewVTable {
    fn compare(
        &self,
        lhs: &FSSTViewArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(constant) = rhs.as_constant() {
            // Compare to NULL returns all-NULL
            if constant.is_null() {
                return Ok(Some(
                    ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), lhs.len())
                        .into_array(),
                ));
            }

            let needle = if let Some(n) = constant.as_utf8_opt() {
                let buffer = n.value().vortex_expect("constant checked to be non-null");
                buffer.into_inner()
            } else {
                constant
                    .as_binary()
                    .value()
                    .vortex_expect("constant checked to be non-null")
            };

            let result_nullability = lhs.dtype.nullability() | constant.dtype().nullability();
            let validity = lhs.validity.clone().cast_nullability(result_nullability)?;

            let needle = needle.as_ref();
            let comparison = if needle.len() <= MAX_INLINE_STR {
                let needle_view = View::new_inlined(needle);

                let buffer = match operator {
                    Operator::Eq => {
                        compare_inline(needle_view, lhs, |n, v| n.to_u128() == v.to_u128())
                    }
                    Operator::NotEq => {
                        compare_inline(needle_view, lhs, |n, v| n.to_u128() != v.to_u128())
                    }
                    // TODO(aduffy): support <, >, etc.
                    _ => return Ok(None),
                };

                BoolArray::new(buffer, validity).into_array()
            } else {
                // Perform a full compare using the operator.
                let buffer = match operator {
                    Operator::Eq => compare_outlined(needle, lhs, eq, |n, v| n == v),
                    Operator::NotEq => compare_outlined(needle, lhs, not_eq, |n, v| n != v),
                    // TODO(aduffy): support <, >, etc.
                    _ => return Ok(None),
                };

                BoolArray::new(buffer, validity).into_array()
            };

            Ok(Some(comparison))
        } else {
            Ok(None)
        }
    }
}

#[inline(always)]
fn eq(needle: View, view: View) -> MatchType {
    // Shift off the last 4 bytes which are the buffer index. Keep only the len (4B) and prefix (8B)
    let shifted_needle = needle.to_u128() >> 32;
    let shifted_view = view.to_u128() >> 32;

    if shifted_needle == shifted_view {
        MatchType::Maybe
    } else {
        MatchType::False
    }
}

#[inline(always)]
fn not_eq(needle: View, view: View) -> MatchType {
    // Shift off the last 4 bytes which are the buffer index. Keep only the len (4B) and prefix (8B)
    let shifted_needle = needle.to_u128() >> 32;
    let shifted_view = view.to_u128() >> 32;

    if shifted_needle == shifted_view {
        // If the views match, it's possible that the full values do not match
        MatchType::Maybe
    } else {
        MatchType::True
    }
}

register_kernel!(CompareKernelAdapter(FSSTViewVTable).lift());
