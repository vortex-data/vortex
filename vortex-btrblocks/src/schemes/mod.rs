// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression scheme implementations.

pub mod binary;
pub mod boolean;
pub mod float;
pub mod integer;
pub mod string;

pub mod decimal;
pub mod fixed_size_list;
pub mod temporal;

pub(crate) mod patches;

use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::SchemeExt;

use crate::schemes::integer::RunEndScheme;
use crate::schemes::integer::SparseScheme;

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
        // RLE indices count up within each chunk, so Delta packs them into a few bits. RunEnd
        // over them would run-length encode already run-length encoded data, which every RLE
        // decode and filter then has to expand twice.
        DescendantExclusion {
            excluded: RunEndScheme.id(),
            children: ChildSelection::One(1),
        },
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
