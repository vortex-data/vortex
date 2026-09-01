// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `pco` edition family: revisions to Pco's serialized representation.

use crate::EditionFamily;

/// The `pco` family: Pco wire-format revisions undergoing focused testing.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "pco",
    origin: "vortex",
    doc: "Pco wire-format revisions undergoing focused testing before they are ready for the \
shared preview family.",
};

pub mod v2026_09;

pub use v2026_09::PCO_2026_09_0;
