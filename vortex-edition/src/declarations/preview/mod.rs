// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `preview` edition family: stabilized core components and array writer-version upgrades
//! awaiting explicit adoption.
//!
//! One module per draft edition, each declaring the members that join the family at it.
//! Members of earlier editions are inherited and never restated.

use crate::EditionFamily;

/// The `preview` family: stabilized, opt-in core functionality.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "preview",
    doc: "Stabilized, opt-in components and array writer-version upgrades maintained as \
part of core but not yet adopted by the default core writer. Preview behavior is expected to \
remain compatible and should change only to fix a defect serious enough to block promotion into \
core. A writer-version upgrade lets compression schemes produce new optional fields or \
properties; it never selects a reader. Users keep the earlier serialized form until they opt \
into that edition. Experimental work \
advances through new draft editions; optional plugins instead use standalone families such as \
spatial and json.",
};

pub mod v2025_05;
pub mod v2026_02;
pub mod v2026_04;
pub mod v2026_06;

pub use v2025_05::PREVIEW_2025_05_0;
pub use v2026_02::PREVIEW_2026_02_0;
pub use v2026_04::PREVIEW_2026_04_0;
pub use v2026_06::PREVIEW_2026_06_0;
