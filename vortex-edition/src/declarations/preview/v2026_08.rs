// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 `preview` edition.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;

/// The August 2026 draft edition of the `preview` family.
pub const PREVIEW_2026_08_0: EditionId = EditionId::new("preview", 2026, 8, 0);

/// The empty declaration of [`PREVIEW_2026_08_0`].
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2026_08_0,
        min_library_version: None,
    },
    added: &[],
};
