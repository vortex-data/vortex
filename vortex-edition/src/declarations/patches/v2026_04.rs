// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The April 2026 `patches` edition.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The April 2026 draft edition of the `patches` family.
pub const PATCHES_2026_04_0: EditionId = EditionId::new("patches", 2026, 4, 0);

/// The declaration of [`PATCHES_2026_04_0`].
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PATCHES_2026_04_0,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"vortex.patched")],
};
