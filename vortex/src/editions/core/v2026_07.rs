// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `core` edition adding stable components released through July 2026.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The July 2026 edition of the `core` family.
pub const CORE_2026_07_0: EditionId = EditionId::new("core", 2026, 7, 0);

/// The declaration of [`CORE_2026_07_0`] and the components that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_07_0,
        min_vortex_version: Some("0.65.0"),
    },
    added: &[
        EditionMember::array(&"vortex.variant"),
        EditionMember::dtype(&"vortex.uuid"),
    ],
};
