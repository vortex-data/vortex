// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `parquet-variant` edition family.
//!
//! [`crate::initialize`] registers and enables the edition together with the layout.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionFamily;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The `parquet-variant` family: the Parquet Variant storage layout.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "parquet-variant",
    origin: "vortex-parquet-variant",
    doc: "The layout that stores Parquet Variant columns as their decomposed storage struct. A \
reader built without `vortex-parquet-variant` cannot resolve `vortex.parquet_variant`, so the \
layout is versioned independently of `core` and a session enables this family only by \
initializing the crate.",
};

/// The August 2026 draft edition of the `parquet-variant` family.
pub const PARQUET_VARIANT_2026_08: EditionId = EditionId::new("parquet-variant", 2026, 8, 0);

/// The declaration of [`PARQUET_VARIANT_2026_08`] and the components that join the family at it.
///
/// A draft: no Vortex release yet guarantees this member forever.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PARQUET_VARIANT_2026_08,
        min_library_version: None,
    },
    added: &[EditionMember::layout(&"vortex.parquet_variant")],
};

#[cfg(test)]
mod tests {
    use vortex_edition::test_harness::validate_edition;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn parquet_variant_edition_is_valid() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        validate_edition(
            &vortex_edition::EditionSessionExt::editions(&session),
            &PARQUET_VARIANT_2026_08,
        )
    }
}
