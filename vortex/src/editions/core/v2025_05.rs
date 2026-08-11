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
        &("fastlanes.bitpacked", "0.36.0"),
        &("fastlanes.for", "0.36.0"),
        &("vortex.alp", "0.36.0"),
        &("vortex.alprd", "0.36.0"),
        &("vortex.bool", "0.36.0"),
        &("vortex.bytebool", "0.36.0"),
        &("vortex.chunked", "0.36.0"),
        &("vortex.constant", "0.36.0"),
        &("vortex.datetimeparts", "0.36.0"),
        &("vortex.decimal", "0.36.0"),
        &("vortex.decimal_byte_parts", "0.36.0"),
        &("vortex.dict", "0.36.0"),
        &("vortex.ext", "0.36.0"),
        &("vortex.fsst", "0.36.0"),
        &("vortex.list", "0.36.0"),
        &("vortex.null", "0.36.0"),
        &("vortex.primitive", "0.36.0"),
        &("vortex.runend", "0.36.0"),
        &("vortex.sparse", "0.36.0"),
        &("vortex.struct", "0.36.0"),
        &("vortex.varbin", "0.36.0"),
        &("vortex.varbinview", "0.36.0"),
        &("vortex.zigzag", "0.36.0"),
    ],
};
