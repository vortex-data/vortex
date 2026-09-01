// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The May 2025 `delta` edition.

use crate::Edition;
use crate::EditionDeclaration;
use crate::EditionId;
use crate::EditionMember;

/// The May 2025 draft edition of the `delta` family.
pub const DELTA_2025_05_0: EditionId = EditionId::new("delta", 2025, 5, 0);

/// The declaration of [`DELTA_2025_05_0`].
pub static DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: DELTA_2025_05_0,
        min_library_version: None,
    },
    added: &[EditionMember::array(&"fastlanes.delta")],
};
