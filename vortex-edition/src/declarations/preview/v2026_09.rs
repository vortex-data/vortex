// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The September 2026 `preview` edition.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The September 2026 draft edition of the `preview` family, adding byte-parts decimals wider
/// than one signed part.
pub const PREVIEW_2026_09_0: EditionId = EditionId::new("preview", 2026, 9, 0);

/// The declaration of [`PREVIEW_2026_09_0`] and the components that join the family at it.
///
/// `vortex.decimal_byte_parts_v2` is the second serialized format of the `DecimalByteParts`
/// encoding: an array carrying 64-bit lower parts serializes under it because the frozen
/// `vortex.decimal_byte_parts` format promises a single child.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2026_09_0,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"vortex.decimal_byte_parts_v2")],
};
