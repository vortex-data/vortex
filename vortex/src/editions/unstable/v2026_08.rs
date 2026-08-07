// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 `unstable` encoding cohort.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

/// The August 2026 draft edition of the `unstable` family.
pub const UNSTABLE_2026_08_0: EditionId = EditionId::new("unstable", 2026, 8, 0);

/// The declaration of [`UNSTABLE_2026_08_0`] and the encodings that join the family at it.
///
/// `vortex.decimal_byte_parts_v2` is a serialized format of the `DecimalByteParts`
/// encoding: byte-parts arrays carrying lower parts serialize under this id because the
/// frozen `vortex.decimal_byte_parts` format promises a single child.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE_2026_08_0,
        min_vortex_version: None,
    },
    added: &[&"vortex.decimal_byte_parts_v2"],
};
