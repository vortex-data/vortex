// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The first `core` edition: the canonical encodings.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

/// The first edition of the `core` family: the canonical encodings — the uncompressed
/// representations every logical type decodes to. A draft until the release shipping it is
/// published and its version is recorded.
pub const CORE_2026_01_0: EditionId = EditionId::new("core", 2026, 1, 0);

/// The declaration of [`CORE_2026_01_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_01_0,
        // TODO(editions): freeze by setting this to the first release shipping this
        //  edition, once it is published.
        min_vortex_version: None,
    },
    added: &[
        &"vortex.null",
        &"vortex.bool",
        &"vortex.primitive",
        &"vortex.decimal",
        &"vortex.varbin",
        &"vortex.varbinview",
        &"vortex.list",
        &"vortex.listview",
        &"vortex.fixed_size_list",
        &"vortex.struct",
        &"vortex.variant",
        &"vortex.ext",
    ],
};

#[cfg(test)]
mod tests {
    use vortex_edition::EditionError;
    use vortex_edition::test_harness::validate_edition;

    use super::CORE_2026_01_0;
    use crate::editions::edition_session;

    #[test]
    fn edition_is_valid() -> Result<(), EditionError> {
        validate_edition(&edition_session(), &CORE_2026_01_0)
    }
}
