// SPDX-License-Identifier: Apache-2.0
//! Predicate AST.
//!
//! A minimal SQL-string-predicate AST. We keep this hand-written
//! rather than pulling in a SQL parser because we only need ~15
//! variants for skip-index dispatch, and avoiding the dep keeps this
//! crate embeddable.

use serde::{Deserialize, Serialize};

/// SQL-ish string predicates that the skip index can prune against.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Pred {
    /// `col = 'x'`
    Eq(Vec<u8>),
    /// `col < 'x'`
    Lt(Vec<u8>),
    /// `col > 'x'`
    Gt(Vec<u8>),
    /// `col <= 'x'`
    Le(Vec<u8>),
    /// `col >= 'x'`
    Ge(Vec<u8>),
    /// `col BETWEEN lo AND hi` (inclusive both ends)
    Between(Vec<u8>, Vec<u8>),
    /// `col LIKE 'prefix%'`
    Prefix(Vec<u8>),
    /// `col LIKE '%suffix'`
    Suffix(Vec<u8>),
    /// `col LIKE '%needle%'`
    Contains(Vec<u8>),
    /// `col LIKE 'p%s'` (anchored prefix + anchored suffix)
    PrefixSuffix(Vec<u8>, Vec<u8>),
    /// `col LIKE '%p_s%'` (single-char wildcard between two anchors)
    SingleWildcard(Vec<u8>, Vec<u8>),
    /// `col LIKE '%a%b%c%'` (multiple fragments in order)
    MultiFragment(Vec<Vec<u8>>),
    /// `LENGTH(col) > k`
    LengthGt(usize),
    /// `LENGTH(col) BETWEEN lo AND hi`
    LengthBetween(usize, usize),
    /// `col IS NULL`
    IsNull,
    /// `col IS NOT NULL`
    IsNotNull,
    /// `col IN (...)` — set membership
    InSet(Vec<Vec<u8>>),
}

impl Pred {
    /// Evaluate this predicate against a single row's bytes.
    pub fn matches_one(&self, row: &[u8]) -> bool {
        match self {
            Pred::Eq(x) => row == x.as_slice(),
            Pred::Lt(x) => row < x.as_slice(),
            Pred::Gt(x) => row > x.as_slice(),
            Pred::Le(x) => row <= x.as_slice(),
            Pred::Ge(x) => row >= x.as_slice(),
            Pred::Between(a, b) => row >= a.as_slice() && row <= b.as_slice(),
            Pred::Prefix(p) => row.starts_with(p.as_slice()),
            Pred::Suffix(s) => row.ends_with(s.as_slice()),
            Pred::Contains(s) => memchr::memmem::find(row, s).is_some(),
            Pred::PrefixSuffix(p, s) => row.starts_with(p.as_slice()) && row.ends_with(s.as_slice()),
            Pred::SingleWildcard(p, s) => {
                let need = p.len() + 1 + s.len();
                if row.len() < need {
                    return false;
                }
                for i in 0..=row.len() - need {
                    if &row[i..i + p.len()] == p.as_slice()
                        && &row[i + p.len() + 1..i + p.len() + 1 + s.len()] == s.as_slice()
                    {
                        return true;
                    }
                }
                false
            }
            Pred::MultiFragment(frags) => {
                let mut pos = 0;
                for f in frags {
                    match memchr::memmem::find(&row[pos..], f.as_slice()) {
                        Some(off) => pos = pos + off + f.len(),
                        None => return false,
                    }
                }
                true
            }
            Pred::LengthGt(k) => row.len() > *k,
            Pred::LengthBetween(lo, hi) => row.len() >= *lo && row.len() <= *hi,
            Pred::IsNull => false, // bytes are non-null by construction
            Pred::IsNotNull => true,
            Pred::InSet(xs) => xs.iter().any(|x| row == x.as_slice()),
        }
    }

    /// True iff this predicate matches ANY row in the given slice.
    /// This is the ground-truth check used for soundness verification.
    pub fn matches_any<S: AsRef<[u8]>>(&self, rows: &[S]) -> bool {
        rows.iter().any(|r| self.matches_one(r.as_ref()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_matches_exactly() {
        let p = Pred::Eq(b"hello".to_vec());
        assert!(p.matches_one(b"hello"));
        assert!(!p.matches_one(b"helloo"));
        assert!(!p.matches_one(b"hell"));
    }

    #[test]
    fn between_inclusive() {
        let p = Pred::Between(b"apple".to_vec(), b"cherry".to_vec());
        assert!(p.matches_one(b"apple"));
        assert!(p.matches_one(b"banana"));
        assert!(p.matches_one(b"cherry"));
        assert!(!p.matches_one(b"date"));
        assert!(!p.matches_one(b"alpha"));
    }

    #[test]
    fn prefix_suffix() {
        let p = Pred::PrefixSuffix(b"http://".to_vec(), b".com".to_vec());
        assert!(p.matches_one(b"http://example.com"));
        assert!(!p.matches_one(b"https://example.com"));
        assert!(!p.matches_one(b"http://example.org"));
    }

    #[test]
    fn single_wildcard() {
        let p = Pred::SingleWildcard(b"hel".to_vec(), b"o".to_vec());
        // Looking for "hel?o" as a substring
        assert!(p.matches_one(b"hello"));    // hel + l + o
        assert!(p.matches_one(b"yhelxo abc")); // hel + x + o at offset 1
        assert!(p.matches_one(b"helloo"));   // hel + l + o at offset 0 still matches
        // Negative: pattern doesn't appear at all
        assert!(!p.matches_one(b"hxllo"));   // no "hel" prefix
        assert!(!p.matches_one(b"helxx"));   // "hel" then wildcard then NOT 'o'
        assert!(!p.matches_one(b"hi"));      // too short
    }

    #[test]
    fn multi_fragment_in_order() {
        let p = Pred::MultiFragment(vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
        assert!(p.matches_one(b"axbyc"));
        assert!(p.matches_one(b"abc"));
        assert!(!p.matches_one(b"cba")); // wrong order
    }

    #[test]
    fn length_predicates() {
        assert!(Pred::LengthGt(5).matches_one(b"abcdef"));
        assert!(!Pred::LengthGt(5).matches_one(b"abcde"));
        assert!(Pred::LengthBetween(2, 4).matches_one(b"abc"));
        assert!(!Pred::LengthBetween(2, 4).matches_one(b"a"));
    }
}
