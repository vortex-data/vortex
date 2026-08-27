// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 draft core edition adding block-wise frame of reference arrays.

use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;

/// The fifth August 2026 edition of the `core` family.
pub const CORE_2026_08_4: EditionId = EditionId::new("core", 2026, 8, 4);

/// The declaration of [`CORE_2026_08_4`] and the components that join the family at it.
///
/// A draft: [`crate::editions::DEFAULT_CORE_EDITION`] does not point here, so the default file
/// writer still refuses `fastlanes.blockedfor`. Callers evaluating the encoding — `vortex-bench`
/// today — enable this edition explicitly.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_08_4,
        min_vortex_version: None,
    },
    added: &[EditionMember::array(&"fastlanes.blockedfor")],
};
