// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `preview` edition family: stabilized core components and serialized array representations
//! awaiting explicit adoption.
//!
//! One module per draft edition, each declaring the members that join the family at it.
//! Members of earlier editions are inherited and never restated.

use crate::EditionFamily;

/// The `preview` family: stabilized, opt-in core functionality.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "preview",
    doc: "Stabilized, opt-in components and serialized array representations maintained as part \
of core but not yet adopted by the default core writer. Preview behavior is expected to remain \
compatible and should change only to fix a defect serious enough to block promotion into core. \
A new wire representation has a new array ID, even when it serializes and deserializes the same \
in-memory array as an older ID. Experimental work advances through new draft editions; optional \
plugins instead use standalone families such as spatial and json.",
};

pub mod v2025_05;
pub mod v2026_02;
pub mod v2026_04;
pub mod v2026_06;

pub use v2025_05::PREVIEW_2025_05_0;
pub use v2026_02::PREVIEW_2026_02_0;
pub use v2026_04::PREVIEW_2026_04_0;
pub use v2026_06::PREVIEW_2026_06_0;
