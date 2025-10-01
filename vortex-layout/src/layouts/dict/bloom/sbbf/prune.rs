// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_expr::{
    BinaryExpr, BinaryVTable, LikeExpr, LikeVTable, LiteralVTable, Operator, RootVTable, VortexExpr,
};

use crate::layouts::dict::bloom::{BloomFilter, LikePattern, parse_like_pattern};

pub trait BloomPruner {
    /// Returns true if the data referenced by this expression can be pruned using the provided
    /// filter.
    ///
    /// Returns false if the expression fails to prune, or the expression does not support pruning
    /// with bloom filters.
    fn can_prune(&self, _filter: &BloomFilter) -> bool {
        false
    }
}

impl BloomPruner for dyn VortexExpr {
    fn can_prune(&self, filter: &BloomFilter) -> bool {
        if let Some(binary) = self.as_opt::<BinaryVTable>() {
            binary.can_prune(filter)
        } else if let Some(like) = self.as_opt::<LikeVTable>() {
            like.can_prune(filter)
        } else {
            // fall through to fail pruning
            false
        }
    }
}

impl BloomPruner for BinaryExpr {
    fn can_prune(&self, filter: &BloomFilter) -> bool {
        match self.op() {
            Operator::Eq => {
                // LHS must be the root "$" expression that is covered by the bloom filter.
                let Some(_) = self.lhs().as_opt::<RootVTable>() else {
                    return false;
                };

                // RHS must be a literal value
                let Some(literal) = self.rhs().as_opt::<LiteralVTable>() else {
                    return false;
                };

                // The literal cannot be null
                let Some(value) = literal.value().as_utf8().value() else {
                    return false;
                };
                let value_str = value.as_str();

                !filter.check(value_str)
            }
            Operator::And => {
                let lhs = self.lhs();
                let rhs = self.rhs();

                // AND can be pruned if EITHER the left or right hand operands can prune
                BloomPruner::can_prune(lhs.as_ref(), filter)
                    || BloomPruner::can_prune(rhs.as_ref(), filter)
            }
            Operator::Or => {
                let lhs = self.lhs();
                let rhs = self.rhs();

                // OR can only be pruned if BOTH the left/right operands are pruned
                BloomPruner::can_prune(lhs.as_ref(), filter)
                    && BloomPruner::can_prune(rhs.as_ref(), filter)
            }
            _ => false,
        }
    }
}

impl BloomPruner for LikeExpr {
    fn can_prune(&self, filter: &BloomFilter) -> bool {
        // We can't support negated filters or case-insensitive with our normal tokenizer.
        if self.negated() || self.case_insensitive() {
            return false;
        }

        // Child/LHS must be the root node
        if !self.child().is::<RootVTable>() {
            return false;
        }

        // Pattern must be a literal
        let Some(pattern) = self.pattern().as_opt::<LiteralVTable>() else {
            return false;
        };

        // Parse the literal as a UTF-8 string instead
        let Some(pattern_str) = pattern.value().as_utf8_opt().and_then(|s| s.value()) else {
            return false;
        };

        // Use the bloom filter to try and prune the different LIKE patterns.
        // We prune when the filter check returns false, indicating that the column chunk
        // *certainly* does not contain the target data.
        match parse_like_pattern(pattern_str.as_str()) {
            LikePattern::Exact(exact) => !filter.check(exact),
            LikePattern::Suffix(suffix) => !filter.check_suffix(suffix),
            LikePattern::Prefix(prefix) => !filter.check_prefix(prefix),
            LikePattern::Contains(substr) => !filter.check_contains(substr),
            LikePattern::Other(_) => false,
        }
    }
}
