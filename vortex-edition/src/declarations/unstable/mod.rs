// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `unstable` edition family: core-maintained components undergoing focused testing.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionFamily;
use crate::EditionId;
use crate::EditionMember;

/// The shared family for core-maintained components before promotion to preview.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "unstable",
    origin: "vortex",
    doc: "Opt-in components maintained as part of Vortex whose serialized contracts are ready \
for focused testing. The family has one undated edition that never freezes, so components join \
and leave it in place; a reader-visible correction still takes a new wire ID. Tested contracts \
move into preview for broad opt-in use, then into core for use by default. The unstable edition \
carries no read-forever guarantee.",
};

/// The single, undated edition of the `unstable` family.
pub const UNSTABLE: EditionId = EditionId::undated("unstable");

/// Built-in serialized components awaiting stabilization in [`UNSTABLE`].
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE,
        min_library_version: None,
    },
    added: &[
        EditionMember::array(&"fastlanes.bitpacked.v2"),
        EditionMember::array(&"fastlanes.delta"),
        EditionMember::array(&"fastlanes.for.v2"),
        EditionMember::array(&"vortex.decimal_byte_parts.v2"),
        EditionMember::array(&"vortex.patched"),
        EditionMember::array(&"vortex.union"),
        EditionMember::layout(&"vortex.list"),
        EditionMember::aggregate(&"vortex.bloom_filter.sbbf"),
        EditionMember::aggregate(&"vortex.sum_v2"),
    ],
};
