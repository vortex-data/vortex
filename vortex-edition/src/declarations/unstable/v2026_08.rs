// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 `unstable` edition.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The August 2026 draft edition of the `unstable` family.
pub const UNSTABLE_2026_08_0: EditionId = EditionId::new("unstable", 2026, 8, 0);

/// Built-in serialized components awaiting stabilization in [`UNSTABLE_2026_08_0`].
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE_2026_08_0,
        min_library_version: None,
    },
    added: &[
        EditionMember::array(&"fastlanes.delta"),
        EditionMember::array(&"vortex.patched"),
        EditionMember::array(&"vortex.union"),
        EditionMember::layout(&"vortex.list"),
        EditionMember::aggregate(&"vortex.bloom_filter.sbbf"),
        EditionMember::aggregate(&"vortex.sum_v2"),
    ],
    removed: &[],
};
