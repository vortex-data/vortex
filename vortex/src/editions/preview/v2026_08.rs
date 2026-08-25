// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 `preview` component cohort.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The August 2026 draft edition of the `preview` family.
pub const PREVIEW_2026_08_0: EditionId = EditionId::new("preview", 2026, 8, 0);

/// The declaration of [`PREVIEW_2026_08_0`] and the components that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2026_08_0,
        min_vortex_version: None,
    },
    added: &[EditionMember::array(&"fastlanes.blockedfor")],
};
