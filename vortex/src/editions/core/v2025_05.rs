// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The baseline `core` edition: stable encodings writable by Vortex 0.36.0.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

/// The first edition of the `core` family, matching the first stable Vortex file release.
pub const CORE_2025_05_0: EditionId = EditionId::new("core", 2025, 5, 0);

/// The declaration of [`CORE_2025_05_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2025_05_0,
        min_vortex_version: Some("0.36.0"),
    },
    added: &[
        &"fastlanes.bitpacked",
        &"fastlanes.for",
        &"vortex.alp",
        &"vortex.alprd",
        &"vortex.bool",
        &"vortex.bytebool",
        &"vortex.chunked",
        &"vortex.constant",
        &"vortex.datetimeparts",
        &"vortex.decimal",
        &"vortex.decimal_byte_parts",
        &"vortex.dict",
        &"vortex.ext",
        &"vortex.fsst",
        &"vortex.list",
        &"vortex.null",
        &"vortex.primitive",
        &"vortex.runend",
        &"vortex.sparse",
        &"vortex.struct",
        &"vortex.varbin",
        &"vortex.varbinview",
        &"vortex.zigzag",
    ],
};
