// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `preview` edition family: stabilized core components and serialized array representations
//! awaiting explicit adoption.
//!
//! One module per draft edition, each declaring the members that join the family at it.
//! Members of earlier editions are inherited and never restated.

use crate::EditionFamily;

/// The `preview` family: test-ready, opt-in core functionality.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "preview",
    doc: "Core-maintained objects that are ready for real-world testing but are not yet adopted \
by the default core writer. Each new object enters through its own new preview edition with a \
wire format believed complete. That format changes only when absolutely necessary to resolve an \
issue found during testing; a reader-visible correction normally gets a new ID and another \
preview edition. After successful testing, the same ID and wire contract move into a new core \
edition. Optional plugins instead use standalone families such as spatial and json.",
};

pub mod v2025_05;
pub mod v2026_02;
pub mod v2026_04;
pub mod v2026_06;

pub use v2025_05::PREVIEW_2025_05_0;
pub use v2026_02::PREVIEW_2026_02_0;
pub use v2026_04::PREVIEW_2026_04_0;
pub use v2026_06::PREVIEW_2026_06_0;
