// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 core edition adding Variant arrays and UUID extension dtypes.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The fourth August 2026 edition of the `core` family.
pub const CORE_2026_08_3: EditionId = EditionId::new("core", 2026, 8, 3);

/// The declaration of [`CORE_2026_08_3`] and the components that join the family at it.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_08_3,
        min_library_version: Some("0.85.0"),
    },
    added: &[
        EditionMember::array(&"vortex.parquet.variant"),
        EditionMember::array(&"vortex.variant"),
        EditionMember::dtype(&"vortex.uuid"),
    ],
};
