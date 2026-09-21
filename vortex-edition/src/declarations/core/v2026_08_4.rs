// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The August 2026 draft core edition adding block-wise frame of reference arrays.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The fifth August 2026 edition of the `core` family.
pub const CORE_2026_08_4: EditionId = EditionId::new("core", 2026, 8, 4);

/// The declaration of [`CORE_2026_08_4`] and the components that join the family at it.
///
/// A draft: the default core edition does not point here, so the default file writer still
/// refuses `fastlanes.blockedfor`. Callers evaluating the encoding — `vortex-bench` today —
/// enable this edition explicitly.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_08_4,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"fastlanes.blockedfor")],
};
