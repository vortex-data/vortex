// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The October 2026 preview edition adding Narrow integer arrays.
//!
//! This declaration permits Narrow on the wire for sessions that enable it. Earlier preview
//! and core editions retain their existing membership.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The October 2026 draft edition of the `preview` family.
pub const PREVIEW_2026_10_0: EditionId = EditionId::new("preview", 2026, 10, 0);

/// The declaration of [`PREVIEW_2026_10_0`] and its additional components.
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: PREVIEW_2026_10_0,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"vortex.narrow")],
};
