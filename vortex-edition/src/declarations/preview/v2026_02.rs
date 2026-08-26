// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The February 2026 `preview` encoding cohort.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The February 2026 draft edition of the `preview` family.
pub const PREVIEW_2026_02_0: EditionId = EditionId::new("preview", 2026, 2, 0);

/// The declaration of [`PREVIEW_2026_02_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2026_02_0,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"vortex.zstd_buffers")],
};
