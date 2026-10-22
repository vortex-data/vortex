// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `unstable` edition family: experimental components with mutable wire contracts.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionFamily;
use crate::EditionId;
use crate::EditionMember;

/// The shared family for experimental components, including third-party plugins.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "unstable",
    origin: "vortex",
    doc: "Opt-in experimental formats in a draft edition that never freezes, so they have no \
compatibility guarantee. Membership and wire contracts may change under existing IDs. Plugins \
can register additional members using ordinary edition inclusions. Settled formats move into \
preview or a plugin edition before freezing; earlier experimental files acquire no compatibility \
guarantee.",
};

/// The shared draft edition. It never freezes, so it has no compatibility guarantee. Its date
/// records when it was added.
pub const UNSTABLE: EditionId = EditionId::new("unstable", 2026, 10, 0);

/// Built-in serialized components awaiting stabilization in [`UNSTABLE`].
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE,
        freezes: false,
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
