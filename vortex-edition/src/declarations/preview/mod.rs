// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `preview` edition family: opt-in components awaiting adoption into `core`.

use crate::EditionFamily;

/// The `preview` family: tested opt-in components not yet available by default.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "preview",
    origin: "vortex",
    doc: "Opt-in components maintained as part of Vortex but not yet adopted by the default \
core writer. Components leave UNSTABLE once their serialized contracts \
are settled, then enter preview for broad testing. They join core with the same IDs and wire \
contracts. Preview edition membership is additive; changing contracts remain in UNSTABLE.",
};

pub mod v2026_08;

pub use v2026_08::PREVIEW_2026_08_0;
