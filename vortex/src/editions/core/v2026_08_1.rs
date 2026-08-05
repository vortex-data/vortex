// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 core edition revision adding the canonical Map encoding.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

/// The August 2026 core edition revision containing canonical Map arrays.
pub const CORE_2026_08_1: EditionId = EditionId::new("core", 2026, 8, 1);

/// The declaration of [`CORE_2026_08_1`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_08_1,
        min_vortex_version: Some("0.84.0"),
    },
    added: &[&"vortex.map"],
};
