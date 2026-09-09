// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `preview` edition family: opt-in components awaiting adoption into `core`.

use crate::EditionFamily;

/// The `preview` family: tested opt-in components not yet available by default.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "preview",
    origin: "vortex",
    doc: "Opt-in components maintained as part of Vortex but not yet adopted by the default \
core writer. Components move from unstable into preview once their serialized contracts are \
ready for broad testing, then join core with the same IDs and wire contracts. Draft preview \
editions may add and remove components together; frozen members cannot be removed.",
};

pub mod v2026_08;

pub use v2026_08::PREVIEW_2026_08_0;
