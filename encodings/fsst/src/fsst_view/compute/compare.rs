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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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
#[inline]
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

#[inline]
fn compare_outlined<V, B>(
    needle: &[u8],
    haystack: &FSSTViewArray,
    cmp_views: V,
    cmp_bytes: B,
) -> BooleanBuffer
where
    V: Fn(View, View) -> MatchType,
    B: Fn(&[u8], &[u8]) -> bool,
{
    let mut result = BooleanBufferBuilder::new(haystack.len());

    // use dummy buffer index of zero, comparison functions will chop off the `index` component
    let needle_view = View::new_outlined(needle, 0);

    for (index, &view) in haystack.views().iter().enumerate() {
        match cmp_views(needle_view, view) {
            MatchType::False => result.append(false),
            MatchType::True => result.append(true),
            MatchType::Maybe => {
                if haystack.is_valid(index) {
                    let full = haystack.bytes_at(index);
                    result.append(cmp_bytes(needle, full.as_ref()))
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
    let shifted_needle = needle.to_u128() << 32;
    let shifted_view = view.to_u128() << 32;

    if shifted_needle == shifted_view {
        // Short-circuit full match if needle and view are both inlined
        if needle.is_inlined() && view.is_inlined() {
            MatchType::True
        } else {
            MatchType::Maybe
        }
    } else {
        MatchType::False
    }
}

#[inline(always)]
fn not_eq(needle: View, view: View) -> MatchType {
    // Shift off the top 4 bytes which are the buffer index. Keep only the len (4B) and prefix (8B)
    let shifted_needle = needle.to_u128() << 32;
    let shifted_view = view.to_u128() << 32;

    if shifted_needle == shifted_view {
        // If the views match, it's possible that the full values do not match
        if needle.is_inlined() && view.is_inlined() {
            MatchType::False
        } else {
            MatchType::Maybe
        }
    } else {
        MatchType::True
    }
}

register_kernel!(CompareKernelAdapter(FSSTViewVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{MatchType, eq, not_eq};
    use crate::View;

    // Check that
    #[rstest]
    #[case::outline_eq1(
        View::new_outlined(b"hello world 1234", 1),
        View::new_outlined(b"hello world 1235", 10),
        MatchType::Maybe
    )]
    #[case::outline_eq2(
        View::new_outlined(b"hello world 123", 10),
        View::new_outlined(b"hello world 1234", 10),
        MatchType::False
    )]
    #[case::inline_eq1(
        View::new_inlined(b"hello world"),
        View::new_outlined(b"hello world     ", 10),
        MatchType::False
    )]
    #[case::inline_eq2(
        View::new_inlined(b"hello world"),
        View::new_inlined(b"hello world"),
        MatchType::True
    )]
    fn test_eq_kernel(#[case] needle: View, #[case] view: View, #[case] match_type: MatchType) {
        assert_eq!(eq(needle, view), match_type);
    }

    #[rstest]
    #[case::outline_neq1(
        View::new_outlined(b"hello world 12345", 1),
        View::new_outlined(b"hello world 1", 1),
        MatchType::True
    )]
    #[case::outline_neq2(
        View::new_outlined(b"hello world 12345", 1),
        View::new_outlined(b"hello world 12346", 1),
        MatchType::Maybe
    )]
    #[case::outline_neq3(
        View::new_outlined(b"hello world 12345", 1),
        View::new_outlined(b"hello world 12345", 1),
        MatchType::Maybe
    )]
    #[case::inline_neq1(
        View::new_inlined(b"hello world"),
        View::new_inlined(b"HELLO world"),
        MatchType::True
    )]
    #[case::inline_neq1(
        View::new_inlined(b"hello world"),
        View::new_inlined(b"hello world"),
        MatchType::False
    )]
    fn test_not_eq_kernel(#[case] needle: View, #[case] view: View, #[case] match_type: MatchType) {
        assert_eq!(not_eq(needle, view), match_type);
    }
}
