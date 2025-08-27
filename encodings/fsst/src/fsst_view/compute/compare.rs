// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::{BoolArray, BooleanBufferBuilder, ConstantArray};
use vortex_array::compute::{CompareKernel, CompareKernelAdapter, Operator};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::{VortexExpect, VortexResult};

use crate::{FSSTViewArray, FSSTViewVTable, View};

enum MatchType {
    /// The operator evaluates to false for the predicate
    False,
    /// The operator evaluates to true for the predicate
    True,
    /// The operation cannot be exactly determined just from examining the view.
    Maybe,
}

trait Predicate {
    /// Perform an inexact match against a view
    fn eval_view(needle: &[u8], view: View) -> MatchType;
    /// Perform a complete match against an entire string
    fn eval_full(v1: &[u8], v2: &[u8]) -> bool;
}

// Compare function that will prune operands that do not match at all.
// Then for any of the ones that MAY match, we perform a direct comparison in encoded space.
fn compare_scalar<P: Predicate>(needle: &[u8], haystack: &FSSTViewArray) -> VortexResult<ArrayRef> {
    let mut result = BooleanBufferBuilder::new(haystack.len());
    for (index, &view) in haystack.views().iter().enumerate() {
        match P::eval_view(needle, view) {
            MatchType::False => result.append(false),
            MatchType::True => result.append(true),
            MatchType::Maybe => {
                if haystack.is_valid(index)? {
                    let full = haystack.bytes_at(index);
                    result.append(P::eval_full(needle, full.as_ref()))
                } else {
                    // Null value, doesn't matter anyway
                    result.append(false);
                }
            }
        }
    }

    Ok(BoolArray::new(result.finish(), haystack.validity.clone()).into_array())
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
                return Ok(Some(ConstantArray::new(constant, lhs.len()).into_array()));
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

            let needle = needle.as_ref();

            return match operator {
                Operator::Eq => compare_scalar::<Eq>(needle, lhs).map(Some),
                Operator::NotEq => compare_scalar::<NotEq>(needle, lhs).map(Some),
                Operator::Gt | Operator::Gte | Operator::Lt | Operator::Lte => {
                    // Other operators not supported
                    // TODO(aduffy): support pushdown of the other operators onto views
                    Ok(None)
                }
            };
        }

        Ok(None)
    }
}

// Equality
struct Eq;

impl Predicate for Eq {
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn eval_view(needle: &[u8], view: View) -> MatchType {
        let needle_len = needle.len() as u32;
        if needle_len != view.len() {
            return MatchType::False;
        }

        // Check prefix
        if view.is_inlined() {
            // Perform exact match against the string
            let whole = unsafe { view.inline }.bytes;
            if whole == needle {
                MatchType::True
            } else {
                MatchType::False
            }
        } else {
            // Check if the prefix matches

            // SAFETY: we check !is_inlined
            let outlined = unsafe { view.outline };
            let needle_prefix = &needle[..8];
            if needle_prefix == outlined.prefix {
                // If prefix matches, then MAYBE these are the same string.
                // We need to perform a complete comparison to see.
                MatchType::Maybe
            } else {
                MatchType::False
            }
        }
    }

    #[inline(always)]
    fn eval_full(v1: &[u8], v2: &[u8]) -> bool {
        v1 == v2
    }
}

// Inequality
struct NotEq;

impl Predicate for NotEq {
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn eval_view(needle: &[u8], view: View) -> MatchType {
        let needle_len = needle.len() as u32;
        // If lengths don't match, strings won't match
        if needle_len != view.len() {
            return MatchType::True;
        }

        if view.is_inlined() {
            let whole = unsafe { view.inline }.bytes;
            if whole == needle {
                MatchType::False
            } else {
                MatchType::True
            }
        } else {
            let needle_prefix = &needle[..8];
            let prefix = unsafe { view.outline }.prefix;

            if prefix != needle_prefix {
                MatchType::True
            } else {
                // If prefixes match, it's still possible full string won't match
                MatchType::Maybe
            }
        }
    }

    #[inline(always)]
    fn eval_full(v1: &[u8], v2: &[u8]) -> bool {
        v1 != v2
    }
}

register_kernel!(CompareKernelAdapter(FSSTViewVTable).lift());
