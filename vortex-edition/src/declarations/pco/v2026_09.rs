// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The September 2026 `pco` edition.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The September 2026 draft edition of the `pco` family.
pub const PCO_2026_09_0: EditionId = EditionId::new("pco", 2026, 9, 0);

/// The declaration of [`PCO_2026_09_0`].
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PCO_2026_09_0,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"vortex.pco.v2")],
};
