// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression schemes owned by this encoding package.

pub mod bitpacking;
pub mod delta;
pub mod float_rle;
pub mod for_;
pub mod integer_rle;

const RUN_LENGTH_THRESHOLD: u32 = 4;

use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::SchemeExt;
use vortex_compressor::scheme::SchemeId;
use vortex_compressor::session::CompressionSessionExt;

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
            excluded: SchemeId::new("vortex.int.sparse"),
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

/// Register the encoding plugins and their compression schemes.
pub fn initialize(session: &vortex_session::VortexSession) {
    crate::initialize(session);
    session.register_scheme(&for_::FoRScheme);
    session.register_scheme(&bitpacking::BitPackingScheme);
    session.register_scheme(&integer_rle::IntRLEScheme);
    session.register_scheme(&float_rle::FloatRLEScheme);
}
