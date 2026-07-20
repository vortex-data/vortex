// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The second `core` edition: the container and compressed encodings.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;

/// The second `core` edition: inherits everything in
/// [`CORE_2026_01_0`](super::v2026_01::CORE_2026_01_0) and adds the container and
/// compressed encodings the default file writer emits. A draft until the release shipping
/// it is published and its version is recorded.
pub const CORE_2026_07_0: EditionId = EditionId::new("core", 2026, 7, 0);

/// The declaration of [`CORE_2026_07_0`] and the encodings that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_07_0,
        min_vortex_version: None,
    },
    added: &[
        &"vortex.chunked",
        &"vortex.constant",
        &"vortex.dict",
        &"vortex.masked",
        &"vortex.sparse",
        &"vortex.alp",
        &"vortex.alprd",
        &"vortex.bytebool",
        &"vortex.datetimeparts",
        &"vortex.decimal_byte_parts",
        &"vortex.fsst",
        &"vortex.pco",
        &"vortex.runend",
        &"vortex.sequence",
        &"vortex.zigzag",
        &"vortex.zstd",
        &"fastlanes.bitpacked",
        &"fastlanes.delta",
        &"fastlanes.for",
        &"fastlanes.rle",
    ],
};

#[cfg(test)]
mod tests {
    use vortex_edition::EditionError;
    use vortex_edition::test_harness::validate_edition;

    use super::CORE_2026_07_0;
    use crate::editions::edition_session;

    #[test]
    fn edition_is_valid() -> Result<(), EditionError> {
        validate_edition(&edition_session(), &CORE_2026_07_0)
    }
}
