// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `core` edition adding stable encodings released through June 2025.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

/// The June 2025 edition of the `core` family.
pub const CORE_2025_06_0: EditionId = EditionId::new("core", 2025, 6, 0);

/// The declaration of [`CORE_2025_06_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2025_06_0,
        min_vortex_version: Some("0.40.0"),
    },
    added: &[&"vortex.pco", &"vortex.sequence", &"vortex.zstd"],
};
