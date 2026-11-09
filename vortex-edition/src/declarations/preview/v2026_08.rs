// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 `preview` edition.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The August 2026 draft edition of the `preview` family.
pub const PREVIEW_2026_08_0: EditionId = EditionId::new("preview", 2026, 8, 0);

/// The declaration of [`PREVIEW_2026_08_0`] and the components that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2026_08_0,
        immutable: true,
        min_library_version: None,
    },
    added: &[EditionMember::aggregate(
        &"vortex.uncompressed_size_in_bytes",
    )],
};
