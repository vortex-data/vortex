//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{BitAnd, Range};

use async_trait::async_trait;
use vortex_array::arrays::BinaryView;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;

use crate::PruningEvaluation;
use crate::layouts::view::reader::SharedBinaryViewFuture;

/// Pruning evaluator for a ViewLayout, which is able to pushdown certain string expressions
/// which operate over binary views.
#[allow(unused)]
pub(crate) struct ViewPruning {
    views: SharedBinaryViewFuture,
    pushdown_expr: StringPushdownExpr,
    row_range: Range<usize>,
}

impl ViewPruning {
    pub fn new(
        views: SharedBinaryViewFuture,
        pushdown_expr: StringPushdownExpr,
        row_range: Range<usize>,
    ) -> Self {
        Self {
            views,
            pushdown_expr,
            row_range,
        }
    }
}

// An expression over StringView data that is able to pushdown and evaluate over the views alone.
#[allow(unused)]
pub(crate) enum StringPushdownExpr {
    /// STRING1 LIKE "STRING2%"
    StartsWith(StartsWithPredicate),
    /// STRING1 <> STRING2
    Equals(EqualsPredicate),
}

#[async_trait]
impl PruningEvaluation for ViewPruning {
    async fn invoke(&self, mask: Mask) -> VortexResult<Mask> {
        // Wait for the views.
        let mut views = self.views.clone().await?;

        // Slice the views buffer to the specified row range
        if self.row_range.start > 0 || self.row_range.end < views.len() {
            views = views.slice(self.row_range.start..self.row_range.end);
        }

        // Apply the pushdown predicate to the sliced views
        let prune_result = match self.pushdown_expr {
            StringPushdownExpr::StartsWith(ref pred) => build_mask(pred, &views),
            StringPushdownExpr::Equals(ref pred) => build_mask(pred, &views),
        };
        Ok(prune_result.bitand(&mask))
    }
}

#[allow(unused)]
trait StringViewPredicate {
    /// Check if the given binary view *may* match the predicate, or if it *definitely* cannot
    /// match the predicate.
    ///
    /// Certain predicates can be pushed down over the string views, which can be substantially
    /// smaller than the full strings.
    fn matches(&self, view: BinaryView) -> bool;
}

/// Build a mask
#[allow(unused)]
fn build_mask<Pred>(predicate: Pred, views: &[BinaryView]) -> Mask
where
    Pred: StringViewPredicate,
{
    Mask::from_iter(views.iter().map(|&view| predicate.matches(view)))
}

/// Predicate for prefix query pushdown, e.g. `LIKE 'ABC%'`
#[allow(unused)]
pub(crate) struct StartsWithPredicate {
    /// Length of `prefix` in bits. Must be between 1..=32 and a multiple of 8,
    /// but we pre-compute the bits to save a mul op on the hot path.
    prefix_bits: u32,
    /// The prefix, holding up to 4 characters of string data.
    prefix: u32,
}

impl From<&str> for StartsWithPredicate {
    #[allow(clippy::cast_possible_truncation)]
    fn from(value: &str) -> Self {
        let mut prefix = [0u8; 4];
        let bytes = value.as_bytes();
        // cap the prefix len at 4 bytes
        let len = bytes.len().min(4);
        assert!(len >= 1, "Prefix must be at least one byte");

        prefix[..len].copy_from_slice(&bytes[..len]);

        Self {
            // this cast should never truncate, because `len` will always be 1..=4
            prefix_bits: u8::BITS * len as u32,
            prefix: u32::from_le_bytes(prefix),
        }
    }
}

impl StringViewPredicate for StartsWithPredicate {
    fn matches(&self, view: BinaryView) -> bool {
        // First check: the string is at least as long as the prefix
        view.len() > self.prefix_bits &&
        // Second check: the search prefix bytes must match the front of the view.
        // Otherwise, it certainly does not match.
        (self.prefix << (32 - self.prefix_bits)) == (view.prefix() << (32 - self.prefix_bits))
    }
}

impl StringViewPredicate for &StartsWithPredicate {
    fn matches(&self, view: BinaryView) -> bool {
        StringViewPredicate::matches(*self, view)
    }
}

/// Predicate for equality matching.
///
/// This predicate evaluates to true IFF the length and prefix both match the BinaryView.
pub(crate) struct EqualsPredicate {
    length: u32,
    prefix: u32,
}

impl From<&str> for EqualsPredicate {
    fn from(value: &str) -> Self {
        let mut prefix = [0u8; 4];
        let bytes = value.as_bytes();
        let len = bytes.len().min(4);
        prefix[..len].copy_from_slice(&bytes[..len]);

        Self {
            length: value
                .len()
                .try_into()
                .vortex_expect("string length cannot exceed 4GiB"),
            prefix: u32::from_le_bytes(prefix),
        }
    }
}

impl StringViewPredicate for EqualsPredicate {
    fn matches(&self, view: BinaryView) -> bool {
        // If the length and prefix match, the strings MAY be equal.
        // If neither matches, then certainly the strings are not equal.
        self.length == view.len() && view.prefix() == self.prefix
    }
}

impl StringViewPredicate for &EqualsPredicate {
    fn matches(&self, view: BinaryView) -> bool {
        StringViewPredicate::matches(*self, view)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::VarBinViewArray;

    use super::{StartsWithPredicate, StringViewPredicate};

    #[test]
    fn test_stringview_predicates() {
        let array = VarBinViewArray::from_iter_nullable_str([
            Some("AquaTeen Hunger Force"),
            Some("Samurai Jack"),
            Some("Batman Beyond"),
            Some("Johnny Bravo"),
        ]);

        // An inexact match
        let starts_with = StartsWithPredicate::from("Bat");

        for &view in array.views().iter() {
            println!(
                "{} LIKE {} ? {}",
                str::from_utf8(starts_with.prefix.to_le_bytes().as_slice()).unwrap(),
                str::from_utf8(view.prefix().to_le_bytes().as_slice()).unwrap(),
                starts_with.matches(view)
            );
        }
    }
}
