// SPDX-License-Identifier: Apache-2.0
//! Per-chunk byte/length/null statistics.
//!
//! These are the "free" pruning stats every columnar engine should
//! emit. On sorted data they give **exact** pruning for equality,
//! range, prefix, length, and null predicates.

use serde::Deserialize;
use serde::Serialize;

/// Per-chunk statistics. Built once at write time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkStats {
    /// Minimum byte value (lex) across rows.
    pub min: Vec<u8>,
    /// Maximum byte value (lex) across rows.
    pub max: Vec<u8>,
    /// Minimum row length in bytes.
    pub min_len: usize,
    /// Maximum row length in bytes.
    pub max_len: usize,
    /// Number of NULL rows.
    pub null_count: usize,
    /// Total raw bytes in this chunk (sum of row lengths).
    pub raw_bytes: usize,
    /// Number of non-null rows.
    pub n_rows: usize,
}

impl ChunkStats {
    /// Build stats from a slice of byte strings (no nulls).
    pub fn from_rows<S: AsRef<[u8]>>(rows: &[S]) -> Self {
        if rows.is_empty() {
            return Self {
                min: Vec::new(),
                max: Vec::new(),
                min_len: 0,
                max_len: 0,
                null_count: 0,
                raw_bytes: 0,
                n_rows: 0,
            };
        }
        let mut min = rows[0].as_ref().to_vec();
        let mut max = rows[0].as_ref().to_vec();
        let mut min_len = rows[0].as_ref().len();
        let mut max_len = min_len;
        let mut raw_bytes = 0usize;
        for r in rows {
            let b = r.as_ref();
            if b < min.as_slice() {
                min = b.to_vec();
            }
            if b > max.as_slice() {
                max = b.to_vec();
            }
            min_len = min_len.min(b.len());
            max_len = max_len.max(b.len());
            raw_bytes += b.len();
        }
        Self {
            min,
            max,
            min_len,
            max_len,
            null_count: 0,
            raw_bytes,
            n_rows: rows.len(),
        }
    }

    /// Build stats from a slice of optional byte strings (with nulls).
    pub fn from_optional_rows<S: AsRef<[u8]>>(rows: &[Option<S>]) -> Self {
        let mut min: Option<Vec<u8>> = None;
        let mut max: Option<Vec<u8>> = None;
        let mut min_len = usize::MAX;
        let mut max_len = 0usize;
        let mut raw_bytes = 0usize;
        let mut null_count = 0usize;
        let mut n_rows = 0usize;
        for r in rows {
            match r {
                Some(b) => {
                    let b = b.as_ref();
                    n_rows += 1;
                    min_len = min_len.min(b.len());
                    max_len = max_len.max(b.len());
                    raw_bytes += b.len();
                    if min.as_ref().is_none_or(|m| b < m.as_slice()) {
                        min = Some(b.to_vec());
                    }
                    if max.as_ref().is_none_or(|m| b > m.as_slice()) {
                        max = Some(b.to_vec());
                    }
                }
                None => null_count += 1,
            }
        }
        Self {
            min: min.unwrap_or_default(),
            max: max.unwrap_or_default(),
            min_len: if min_len == usize::MAX { 0 } else { min_len },
            max_len,
            null_count,
            raw_bytes,
            n_rows,
        }
    }

    /// Byte size of the stats payload when serialized (approximate).
    pub fn byte_size(&self) -> usize {
        self.min.len() + self.max.len() + 8 * 4 // min/max bytes + four usize stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_rows_min_max() {
        let rows = vec![b"banana".as_slice(), b"apple", b"cherry"];
        let s = ChunkStats::from_rows(&rows);
        assert_eq!(s.min, b"apple");
        assert_eq!(s.max, b"cherry");
        assert_eq!(s.min_len, 5);
        assert_eq!(s.max_len, 6);
        assert_eq!(s.n_rows, 3);
        assert_eq!(s.null_count, 0);
    }

    #[test]
    fn from_optional_handles_nulls() {
        let rows: Vec<Option<&[u8]>> = vec![Some(b"a"), None, Some(b"z"), None];
        let s = ChunkStats::from_optional_rows(&rows);
        assert_eq!(s.min, b"a");
        assert_eq!(s.max, b"z");
        assert_eq!(s.null_count, 2);
        assert_eq!(s.n_rows, 2);
    }

    #[test]
    fn empty_rows() {
        let rows: Vec<&[u8]> = vec![];
        let s = ChunkStats::from_rows(&rows);
        assert_eq!(s.n_rows, 0);
        assert!(s.min.is_empty());
    }
}
