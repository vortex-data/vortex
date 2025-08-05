//! An implementation of the FSST+ extension, original at https://github.com/cwida/fsst_plus/.
//!
//! FSST+ augments FSST with the addition.

#![allow(unused)]

#[derive(Debug)]
pub struct SimilarityBlock {
    start_idx: usize,
    prefix_len: usize,
}

/// Maximum shared prefix length.
pub const MAX_PREFIX: usize = 128;

/// Find the longest-common-prefix between adjacent encoded texts.
pub fn longest_common_prefix(codes: &[&[u8]]) -> Vec<usize> {
    // LCP for each pair of successive strings.
    // LCP[i] is the length of longest common prefix between the i and i+1 string.
    // For example, lcp(["abc", "abcd", "ab"]) -> [3, 2]
    let mut longest_common_prefix = Vec::new();

    // Calculate the LCP of consecutive strings in the input.
    for w in codes.windows(2) {
        let s1 = w[0];
        let s2 = w[1];

        // Consecutive strings to evaluate LCP
        let mut lcp = 0;
        for (&a, &b) in s1.iter().zip(s2.iter()) {
            if a == b {
                lcp += 1;
            } else {
                break;
            }
        }

        longest_common_prefix.push(lcp);
    }

    longest_common_prefix
}

/// Input: a vector of FSST-encoded strings.
/// Output: a vector of "similarity blocks".
///
/// We first calculate the LCP between adjacent strings, then once that has been completed, we find
/// the optimal split points to create "similarity" blocks that all share a maximal prefix.
#[allow(clippy::needless_range_loop)]
pub fn chunk_by_similarity(strings: &[&[u8]]) -> Vec<SimilarityBlock> {
    // Calculate LCP between all items first
    let lcp = longest_common_prefix(strings);

    // min_lcp[i][j] = min(LCP[i], LCP[i+1], ..., LCP[j])
    let mut min_lcp = vec![vec![0; strings.len()]; strings.len()];

    // ... the diagonals are just the LCP[i]'s
    for i in 0..strings.len() {
        // up to 128 is longest prefix we allow.
        min_lcp[i][i] = strings[i].len().min(MAX_PREFIX);
        for j in (i + 1)..lcp.len() {
            min_lcp[i][j] = min_lcp[i][j - 1].min(lcp[j - 1]);
        }
    }


    // Cost is the total cost of the block split.
    let mut cost = vec![usize::MAX; 1+strings.len()];
    cost[0] = 0;

    let mut block = vec![0; 1+strings.len()];
    let mut prefix = vec![0; 1+strings.len()];

    // Estimate the cost for all strings instead here.
    let estimate_cost = |i: u32, j: u32| min_lcp[i as usize][j as usize];

    // cum_len[k] = sum(strings[i].len() for i in 0..k)
    let mut length_prefix_sum =  Vec::new();
    length_prefix_sum.push(0);
    for string in strings {
        length_prefix_sum.push(length_prefix_sum.last().unwrap() + string.len());
    }

    // cost[j] holds the cost up to j.
    // block[j] holds the start of the block that j is a part of
    // prefix[j] is the prefix length for the j-th string
    for end in 1..=strings.len() {
        for start in 0..end {
            dbg!((start, end));
            let min_prefix = min_lcp[start][end - 1];
            dbg!(min_prefix);
            for prefix_len in [0, min_prefix] {
                dbg!(prefix_len);
                let n = end - start;
                dbg!(n);
                let per_string_overhead = if prefix_len == 0 { 1 } else { 3 };
                dbg!(per_string_overhead);
                let overhead = n * per_string_overhead;
                dbg!(overhead);
                let sum_len = length_prefix_sum[end] - length_prefix_sum[start];
                dbg!(sum_len);
                let total_cost = cost[start] + overhead + sum_len - ((n - 1) * prefix_len as usize);
                dbg!(total_cost);

                if total_cost < cost[end] {
                    cost[end] = total_cost;
                    block[end] = start;
                    prefix[end] = prefix_len;
                }
            }
        }
    }

    // Traverse the blocks from end to start.
    let mut blocks = Vec::new();

    let mut idx = strings.len();
    while idx > 0 {
        let start_idx = block[idx];
        let prefix_len = prefix[idx];

        blocks.push(SimilarityBlock {
            start_idx, prefix_len,
        });

        // Advance backward.
        idx = start_idx;
    }

    // Reverse the list of blocks so they're in order now.
    blocks.reverse();

    blocks
}

#[cfg(test)]
mod tests {
    use fsst::Compressor;

    use crate::prefix::{chunk_by_similarity, longest_common_prefix};

    #[test]
    fn test_urls() {
        let strings = vec![
            "reddit.com".as_bytes(),
            "reddit.com/a".as_bytes(),
            "reddit.com/a/b".as_bytes(),
            "reddit.com/c".as_bytes(),
            "reddit.com/c/d/d".as_bytes(),
            "reddit.com/c/e".as_bytes(),
            "google.com".as_bytes(),
            "google.com/search?q=beans".as_bytes(),
            "google.com/search?q=black+beans".as_bytes(),
            "google.com/search?q=lima+beans".as_bytes(),
            "google.com/search?q=taylor+swift".as_bytes(),
        ];

        let compressor = Compressor::train(&strings);
        let result = compressor.compress_bulk(&strings);

        let lcps = longest_common_prefix(strings.as_slice());
        assert_eq!(lcps, vec![10, 12, 11, 12, 13, 0, 10, 21, 20, 20]);

        let chunks = chunk_by_similarity(&strings);

        dbg!(&chunks);

        // Once we have calculate the adjacent_lcp, expand into a new LCP with a bunch of strings
        // in a single block.
    }
}
