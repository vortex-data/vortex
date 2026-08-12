// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The June 2026 `unstable` encoding cohort.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;

/// The June 2026 draft edition of the `unstable` family.
pub const UNSTABLE_2026_06_0: EditionId = EditionId::new("unstable", 2026, 6, 0);

/// The declaration of [`UNSTABLE_2026_06_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE_2026_06_0,
        min_vortex_version: None,
    },
    added: &[&"vortex.onpair"],
};
