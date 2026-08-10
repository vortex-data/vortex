// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 core edition adding the canonical Map encoding.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The August 2026 core edition containing canonical Map arrays.
pub const CORE_2026_08: EditionId = EditionId::new("core", 2026, 8, 0);

/// The declaration of [`CORE_2026_08`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_08,
        min_vortex_version: Some("0.84.0"),
    },
    added: &[EditionMember::array(&"vortex.map")],
};
