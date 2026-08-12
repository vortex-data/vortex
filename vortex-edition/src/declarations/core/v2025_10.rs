// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `core` edition adding stable encodings released through October 2025.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;

/// The October 2025 edition of the `core` family.
pub const CORE_2025_10_0: EditionId = EditionId::new("core", 2025, 10, 0);

/// The declaration of [`CORE_2025_10_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2025_10_0,
        min_vortex_version: Some("0.55.0"),
    },
    added: &[
        &"fastlanes.rle",
        &"vortex.fixed_size_list",
        &"vortex.listview",
        &"vortex.masked",
    ],
};
