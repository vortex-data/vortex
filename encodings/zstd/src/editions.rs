// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `zstd` edition family.
//!
//! Zstd buffer wrapping is opt-in. This module declares its persisted array encoding.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionFamily;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The `zstd` family: optional Zstd-backed serialized array representations.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "zstd",
    origin: "vortex-zstd",
    doc: "Optional Zstd-backed serialized array representations. A reader built without \
`vortex-zstd` cannot resolve these members, so they are versioned independently of `core` and \
enabled only when the crate is initialized with the corresponding feature.",
};

/// The February 2026 draft edition of the `zstd` family.
pub const ZSTD_2026_02: EditionId = EditionId::new("zstd", 2026, 2, 0);

/// The declaration of [`ZSTD_2026_02`] and the components that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: ZSTD_2026_02,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"vortex.zstd_buffers")],
};

#[cfg(test)]
#[cfg(feature = "unstable_encodings")]
mod tests {
    use vortex_edition::EditionError;
    use vortex_edition::EditionSessionExt;
    use vortex_edition::test_harness::validate_edition;

    use super::*;

    #[test]
    fn zstd_edition_is_valid() -> Result<(), EditionError> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        validate_edition(&session.editions(), &ZSTD_2026_02)
    }
}
