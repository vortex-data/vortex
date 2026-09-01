// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `patches` edition family.

use crate::EditionFamily;

/// The `patches` family: patched arrays.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "patches",
    origin: "vortex",
    doc: "Patched arrays that store sparse exceptions separately from a child array.",
};

pub mod v2026_04;

pub use v2026_04::PATCHES_2026_04_0;
