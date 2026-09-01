// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `list` edition family.

use crate::EditionFamily;

/// The `list` family: list layouts.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "list",
    doc: "Layouts that store list elements and offsets in separate child layouts.",
};

pub mod v2026_06;

pub use v2026_06::LIST_2026_06_0;
