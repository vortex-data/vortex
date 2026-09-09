// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `unstable` edition family: core-maintained components undergoing focused testing.

use crate::EditionFamily;

/// The shared family for core-maintained components before promotion to preview.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "unstable",
    origin: "vortex",
    doc: "Opt-in components maintained as part of Vortex whose serialized contracts are ready \
for focused testing. Draft editions may add and remove components together, including replacing \
a wire ID after a reader-visible correction. Tested contracts move into preview for broad opt-in \
use, then into core for use by default. Unstable editions carry no read-forever guarantee.",
};

pub mod v2026_08;

pub use v2026_08::UNSTABLE_2026_08_0;
