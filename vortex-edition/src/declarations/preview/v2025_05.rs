// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The May 2025 `preview` encoding cohort.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The May 2025 draft edition of the `preview` family.
pub const PREVIEW_2025_05_0: EditionId = EditionId::new("preview", 2025, 5, 0);

/// The declaration of [`PREVIEW_2025_05_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2025_05_0,
        min_vortex_version: None,
    },
    added: &[EditionMember::array(&"fastlanes.delta")],
};
