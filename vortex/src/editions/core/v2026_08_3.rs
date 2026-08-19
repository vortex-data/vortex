// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 draft core edition adding OnPair arrays.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The fourth August 2026 edition of the `core` family.
pub const CORE_2026_08_3: EditionId = EditionId::new("core", 2026, 8, 3);

/// The declaration of [`CORE_2026_08_3`] and the components that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_08_3,
        min_vortex_version: None,
    },
    added: &[EditionMember::array(&"vortex.onpair")],
};
