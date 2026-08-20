// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The February 2026 `unstable` encoding cohort.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The February 2026 draft edition of the `unstable` family.
pub const UNSTABLE_2026_02_0: EditionId = EditionId::new("unstable", 2026, 2, 0);

/// The declaration of [`UNSTABLE_2026_02_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE_2026_02_0,
        min_vortex_version: None,
    },
    added: &[EditionMember::array(&"vortex.zstd_buffers")],
};
