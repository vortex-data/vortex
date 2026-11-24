// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `unstable` edition family: experimental components with mutable wire contracts.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionFamily;
use crate::EditionId;
use crate::EditionMember;

/// The shared family for experimental components, including third-party plugins.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "unstable",
    origin: "vortex",
    doc: "Opt-in experimental formats in a draft edition that never freezes, so they have no \
compatibility guarantee. Membership and wire contracts may change under existing IDs. Plugins \
can register additional members using ordinary edition inclusions. Settled formats move into \
preview or a plugin edition before freezing; earlier experimental files acquire no compatibility \
guarantee.",
};

/// The shared draft edition. It never freezes, so it has no compatibility guarantee. Its date
/// records when it was added.
pub const UNSTABLE: EditionId = EditionId::new("unstable", 2026, 10, 0);

/// The edition declaration that lets plugins add experimental formats to [`UNSTABLE`].
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: UNSTABLE,
        immutable: false,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"vortex.widen.v2")],
};
