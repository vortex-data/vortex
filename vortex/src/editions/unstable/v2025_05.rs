// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The May 2025 `unstable` encoding cohort.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

/// The May 2025 draft edition of the `unstable` family.
pub const UNSTABLE_2025_05_0: EditionId = EditionId::new("unstable", 2025, 5, 0);

/// The declaration of [`UNSTABLE_2025_05_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE_2025_05_0,
        min_vortex_version: None,
    },
    added: &[&"fastlanes.delta"],
};
