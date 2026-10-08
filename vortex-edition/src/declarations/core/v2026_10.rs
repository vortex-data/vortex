// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The October 2026 core edition adding the byte-bounded extremum aggregates.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The October 2026 edition of the `core` family.
pub const CORE_2026_10_0: EditionId = EditionId::new("core", 2026, 10, 0);

/// The declaration of [`CORE_2026_10_0`] and the components that join the family at it.
///
/// `vortex.max_bound` and `vortex.min_bound` succeed `vortex.bounded_max` and
/// `vortex.bounded_min` as the zone-map extrema of `Utf8`/`Binary` columns. Their partials record
/// whether the stored value is the exact extremum or only a bound, and keep an empty zone apart
/// from one whose maximum has no representable upper bound. The superseded aggregates remain
/// members so that older files stay readable.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: CORE_2026_10_0,
        min_library_version: None,
    },
    added: &[
        EditionMember::aggregate(&"vortex.max_bound"),
        EditionMember::aggregate(&"vortex.min_bound"),
    ],
};
