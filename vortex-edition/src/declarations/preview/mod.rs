// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `preview` edition family: additive, opt-in components awaiting adoption into `core`.

use crate::EditionFamily;

/// The `preview` family: stable opt-in components not yet available by default.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "preview",
    origin: "vortex",
    doc: "Additive, opt-in components maintained as part of Vortex but not yet adopted by the \
default core writer. Components enter preview only once their serialized contracts are ready for \
broad testing; independently evolving work remains in its own family until then.",
};

pub mod v2026_08;

pub use v2026_08::PREVIEW_2026_08_0;
