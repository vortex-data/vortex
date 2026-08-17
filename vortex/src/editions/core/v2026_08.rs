// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 core edition adding canonical Map arrays and zoned layouts.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The August 2026 core edition containing canonical Map arrays and zoned layouts.
pub const CORE_2026_08: EditionId = EditionId::new("core", 2026, 8, 0);

/// The declaration of [`CORE_2026_08`] and the components that join the family at it.
///
/// The aggregates are the set the default writer records in zone maps. A strategy asking for an
/// aggregate outside this set fails the write instead of producing zone maps an older reader would
/// have to skip.
///
/// `vortex.sum` is deliberately not a member: zone maps prune, a zone sum does not, and its
/// null-on-empty semantics were changed and reverted within a single week. The writer no longer
/// records it, so the two stay consistent.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_08,
        min_vortex_version: Some("0.84.0"),
    },
    added: &[
        EditionMember::array(&"vortex.map"),
        EditionMember::layout(&"vortex.zoned"),
        EditionMember::aggregate(&"vortex.bounded_max"),
        EditionMember::aggregate(&"vortex.bounded_min"),
        EditionMember::aggregate(&"vortex.max"),
        EditionMember::aggregate(&"vortex.min"),
        EditionMember::aggregate(&"vortex.nan_count"),
        EditionMember::aggregate(&"vortex.null_count"),
    ],
};
