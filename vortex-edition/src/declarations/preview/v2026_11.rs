// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The November 2026 `preview` edition.
//!
//! Writers can opt in to the DecimalByteParts v2 and Widen wire formats before they enter `core`.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The November 2026 draft edition of the `preview` family.
pub const PREVIEW_2026_11_0: EditionId = EditionId::new("preview", 2026, 11, 0);

/// The components that [`PREVIEW_2026_11_0`] adds to the preview family.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2026_11_0,
        immutable: true,
        min_library_version: None,
    },
    added: &[
        EditionMember::array(&"vortex.decimal_byte_parts.v2"),
        EditionMember::array(&"vortex.widen"),
    ],
};
