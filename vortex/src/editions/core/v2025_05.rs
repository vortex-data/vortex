// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The baseline `core` edition: stable encodings writable by Vortex 0.36.0.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The first edition of the `core` family, matching the first stable Vortex file release.
pub const CORE_2025_05_0: EditionId = EditionId::new("core", 2025, 5, 0);

/// The declaration of [`CORE_2025_05_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2025_05_0,
        min_vortex_version: Some("0.36.0"),
    },
    added: &[
        EditionMember::array(&"fastlanes.bitpacked"),
        EditionMember::array(&"fastlanes.for"),
        EditionMember::array(&"vortex.alp"),
        EditionMember::array(&"vortex.alprd"),
        EditionMember::array(&"vortex.bool"),
        EditionMember::array(&"vortex.bytebool"),
        EditionMember::array(&"vortex.chunked"),
        EditionMember::array(&"vortex.constant"),
        EditionMember::array(&"vortex.datetimeparts"),
        EditionMember::array(&"vortex.decimal"),
        EditionMember::array(&"vortex.decimal_byte_parts"),
        EditionMember::array(&"vortex.dict"),
        EditionMember::array(&"vortex.ext"),
        EditionMember::array(&"vortex.fsst"),
        EditionMember::array(&"vortex.list"),
        EditionMember::array(&"vortex.null"),
        EditionMember::array(&"vortex.primitive"),
        EditionMember::array(&"vortex.runend"),
        EditionMember::array(&"vortex.sparse"),
        EditionMember::array(&"vortex.struct"),
        EditionMember::array(&"vortex.varbin"),
        EditionMember::array(&"vortex.varbinview"),
        EditionMember::array(&"vortex.zigzag"),
    ],
};
