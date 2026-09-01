// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `delta` edition family.

use crate::EditionFamily;

/// The `delta` family: delta-compressed integer arrays.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "delta",
    origin: "vortex",
    doc: "Delta-compressed integer arrays.",
};

pub mod v2025_05;

pub use v2025_05::DELTA_2025_05_0;
