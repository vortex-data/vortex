// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression scheme implementations.

pub mod binary;
pub mod float;
pub mod integer;
pub mod string;

pub mod decimal;
pub mod temporal;

pub(crate) mod patches;

use std::ops::Range;

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_native_ptype;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::SchemeExt;
use vortex_error::VortexResult;

use crate::normalize_null_values;
use crate::schemes::integer::SparseScheme;

const SAMPLE_BLOCK_LEN: usize = 64;
const MIN_SAMPLE_BLOCKS: usize = 16;
const SAMPLE_BLOCK_MULTIPLE: usize = 16;

fn sample_primitive_one_percent(
    primitive: ArrayView<'_, Primitive>,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let sample_blocks = (primitive.len() / 100 / SAMPLE_BLOCK_LEN)
        .next_multiple_of(SAMPLE_BLOCK_MULTIPLE)
        .max(MIN_SAMPLE_BLOCKS);
    let sample_len = sample_blocks * SAMPLE_BLOCK_LEN;
    if primitive.len() <= sample_len {
        return normalize_null_values(primitive, exec_ctx);
    }

    let ranges = one_percent_sample_ranges(primitive.len(), sample_blocks);
    let validity = primitive.validity()?;
    if validity.definitely_no_nulls() {
        return Ok(match_each_native_ptype!(primitive.ptype(), |T| {
            let values = primitive.as_slice::<T>();
            let mut sample = Vec::with_capacity(sample_len);
            for range in ranges {
                sample.extend_from_slice(&values[range]);
            }
            PrimitiveArray::from_iter(sample)
        }));
    }

    let validity = validity.execute_mask(primitive.len(), exec_ctx)?;
    Ok(match_each_native_ptype!(primitive.ptype(), |T| {
        let values = primitive.as_slice::<T>();
        PrimitiveArray::from_option_iter(
            ranges
                .into_iter()
                .flatten()
                .map(|index| validity.value(index).then_some(values[index])),
        )
    }))
}

fn one_percent_sample_ranges(len: usize, sample_blocks: usize) -> Vec<Range<usize>> {
    let partition_len = len / sample_blocks;
    let long_partitions = len % sample_blocks;
    let mut partition_start = 0;
    (0..sample_blocks)
        .map(|partition_index| {
            let current_partition_len =
                partition_len + usize::from(partition_index < long_partitions);
            let start = partition_start + (current_partition_len - SAMPLE_BLOCK_LEN) / 2;
            partition_start += current_partition_len;
            start..start + SAMPLE_BLOCK_LEN
        })
        .collect()
}

fn sample_primitive_blocks<T: NativePType>(
    values: &[T],
    all_valid: bool,
    is_valid: impl Fn(usize) -> bool,
    full_blocks: usize,
    sample_blocks: usize,
    block_len: usize,
) -> PrimitiveArray {
    if all_valid {
        let mut sample = Vec::with_capacity(sample_blocks * block_len);
        for sample_index in 0..sample_blocks {
            let block_index = sample_index * full_blocks / sample_blocks;
            let start = block_index * block_len;
            sample.extend_from_slice(&values[start..start + block_len]);
        }
        PrimitiveArray::from_iter(sample)
    } else {
        let mut sample = Vec::with_capacity(sample_blocks * block_len);
        for sample_index in 0..sample_blocks {
            let block_index = sample_index * full_blocks / sample_blocks;
            let start = block_index * block_len;
            sample.extend(
                (start..start + block_len).map(|index| is_valid(index).then_some(values[index])),
            );
        }
        PrimitiveArray::from_option_iter(sample)
    }
}

/// Shared descendant exclusion rules for RLE schemes.
///
/// RLE indices (child 1) and offsets (child 2) are monotonically increasing positions with all
/// unique values. Dict and Sparse are pointless on such data. Self-exclusion already prevents
/// RLE on RLE children.
fn rle_descendant_exclusions() -> Vec<DescendantExclusion> {
    vec![
        DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::Many(&[1, 2]),
        },
        // TODO(connor): This is wrong for some reason?
        // DescendantExclusion {
        //     excluded: RunEndScheme.id(),
        //     children: ChildSelection::Many(&[1, 2]),
        // },
        DescendantExclusion {
            excluded: SparseScheme.id(),
            children: ChildSelection::Many(&[1, 2]),
        },
    ]
}

/// Shared ancestor exclusion rules for RLE schemes.
///
/// Dict values (child 0) are all unique by definition, so RLE is pointless on them.
fn rle_ancestor_exclusions() -> Vec<AncestorExclusion> {
    vec![
        AncestorExclusion {
            ancestor: IntDictScheme.id(),
            children: ChildSelection::One(0),
        },
        AncestorExclusion {
            ancestor: FloatDictScheme.id(),
            children: ChildSelection::One(0),
        },
        AncestorExclusion {
            ancestor: StringDictScheme.id(),
            children: ChildSelection::One(0),
        },
        AncestorExclusion {
            ancestor: BinaryDictScheme.id(),
            children: ChildSelection::One(0),
        },
    ]
}
