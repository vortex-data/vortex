// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The first April 2026 `preview` edition.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The April 2026 draft edition adding Patched arrays.
pub const PREVIEW_2026_04_0: EditionId = EditionId::new("preview", 2026, 4, 0);

/// The declaration of [`PREVIEW_2026_04_0`] and the component that joins the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2026_04_0,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"vortex.patched")],
};
