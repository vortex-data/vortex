// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The experimental `binned` edition family.
//!
//! `vortex.binned` is a prototype whose format may still change, so it is opt-in: writers only
//! produce it after enabling [`BINNED_2026_10`] on their session.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionFamily;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The `binned` family: the experimental binned tANS numeric encoding.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "binned",
    origin: "vortex-binned",
    doc: "The experimental binned tANS numeric encoding. Its format may still change, so it is \
versioned independently of `core` and must be enabled explicitly by the caller.",
};

/// The October 2026 draft edition of the `binned` family.
pub const BINNED_2026_10: EditionId = EditionId::new("binned", 2026, 10, 0);

/// The declaration of [`BINNED_2026_10`].
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: BINNED_2026_10,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"vortex.binned")],
};

#[cfg(test)]
mod tests {
    use vortex_edition::EditionSessionExt;
    use vortex_edition::test_harness::validate_edition;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn binned_edition_is_valid() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        validate_edition(&session.editions(), &BINNED_2026_10)
    }

    #[test]
    fn initialize_does_not_enable_binned_edition() {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        assert!(
            !session
                .enabled_editions()
                .editions()
                .contains(&BINNED_2026_10)
        );
    }
}
