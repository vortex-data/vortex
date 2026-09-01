// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The June 2026 `list` edition.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The June 2026 draft edition of the `list` family.
pub const LIST_2026_06_0: EditionId = EditionId::new("list", 2026, 6, 0);

/// The declaration of [`LIST_2026_06_0`].
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: LIST_2026_06_0,
        min_library_version: None,
    },
    added: &[EditionMember::layout(&"vortex.list")],
};
