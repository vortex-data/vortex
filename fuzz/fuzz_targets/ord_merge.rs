// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! libfuzz target for the `OrdIter` n-way merge driver.
//!
//! Generates random sorted columns across mixed encodings (Primitive,
//! Constant, RunEnd) and asserts the merge driver:
//!   - emits exactly the total number of input rows
//!   - never panics or aliases memory
//!   - handles edge cases (empty sides, chunk_size=1, mixed encodings)
//!
//! Run with `cargo fuzz run ord_merge` from the `fuzz/` directory.

#![no_main]

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use libfuzzer_sys::{Corpus, fuzz_target};
use vortex_array::ord_iter::{
    ConstantIter, OrdIter, PrimIter, RunEndIter, merge_n_way,
};

#[derive(Debug)]
enum SideSpec {
    Primitive(Vec<i64>),
    Constant { value: i64, len: u8 },
    RunEnd(Vec<(u8, i64)>),
}

impl<'a> Arbitrary<'a> for SideSpec {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        match u.int_in_range(0u8..=2)? {
            0 => Ok(SideSpec::Primitive(u.arbitrary()?)),
            1 => Ok(SideSpec::Constant {
                value: u.arbitrary()?,
                len: u.arbitrary()?,
            }),
            _ => Ok(SideSpec::RunEnd(u.arbitrary()?)),
        }
    }
}

#[derive(Debug)]
struct FuzzInput {
    sides: Vec<SideSpec>,
    chunk_size_log2: u8,
}

impl<'a> Arbitrary<'a> for FuzzInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Cap the number of sides at 8 to bound work.
        let n_sides = (u.int_in_range(0u8..=8)?) as usize;
        let mut sides = Vec::with_capacity(n_sides);
        for _ in 0..n_sides {
            sides.push(u.arbitrary()?);
        }
        Ok(Self {
            sides,
            chunk_size_log2: u.arbitrary()?,
        })
    }
}

enum OwnedSide {
    Prim(Vec<i64>),
    Const(i64, usize),
    Run { ends: Vec<u32>, vals: Vec<i64>, total: usize },
}

fuzz_target!(|input: FuzzInput| -> Corpus {
    if input.sides.is_empty() {
        return Corpus::Reject;
    }
    let est_total: usize = input
        .sides
        .iter()
        .map(|s| match s {
            SideSpec::Primitive(v) => v.len(),
            SideSpec::Constant { len, .. } => *len as usize,
            SideSpec::RunEnd(runs) => runs.iter().map(|(l, _)| *l as usize).sum(),
        })
        .sum();
    if est_total > 100_000 {
        return Corpus::Reject;
    }

    let chunk_size = (1usize << (input.chunk_size_log2 as u32 % 13)).max(1);

    let mut owned: Vec<OwnedSide> = Vec::with_capacity(input.sides.len());
    let mut expected_count = 0usize;
    for spec in &input.sides {
        match spec {
            SideSpec::Primitive(values) => {
                let mut v = values.clone();
                v.sort();
                expected_count += v.len();
                owned.push(OwnedSide::Prim(v));
            }
            SideSpec::Constant { value, len } => {
                let l = *len as usize;
                expected_count += l;
                owned.push(OwnedSide::Const(*value, l));
            }
            SideSpec::RunEnd(runs) => {
                let mut pairs: Vec<(u8, i64)> = runs.clone();
                pairs.sort_by_key(|(_, v)| *v);
                let mut ends = Vec::new();
                let mut vals = Vec::new();
                let mut cursor: u32 = 0;
                for (l, v) in &pairs {
                    let len = u32::from(*l);
                    if len == 0 {
                        continue;
                    }
                    cursor = cursor.saturating_add(len);
                    ends.push(cursor);
                    vals.push(*v);
                }
                let total_re = cursor as usize;
                expected_count += total_re;
                owned.push(OwnedSide::Run { ends, vals, total: total_re });
            }
        }
    }

    let mut iters: Vec<Box<dyn OrdIter + '_>> = Vec::with_capacity(owned.len());
    for s in &owned {
        match s {
            OwnedSide::Prim(v) => iters.push(Box::new(PrimIter::new(v))),
            OwnedSide::Const(v, l) => iters.push(Box::new(ConstantIter::new(*v, *l))),
            OwnedSide::Run { ends, vals, total } => {
                iters.push(Box::new(RunEndIter::new(ends, vals, *total)))
            }
        }
    }

    let count = merge_n_way(&mut iters, chunk_size);
    assert_eq!(
        count, expected_count,
        "merge_n_way count {count} != expected {expected_count} \
         chunk={chunk_size} n_sides={}",
        owned.len()
    );

    Corpus::Keep
});
