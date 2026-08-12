// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `unstable` edition family: opt-in encodings without a frozen compatibility guarantee.
//!
//! One module per draft edition, each declaring the encodings that join the family at it.
//! Members of earlier editions are inherited and never restated.

use crate::EditionFamily;

/// The `unstable` family: opt-in encodings with no compatibility guarantee.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "unstable",
    doc: "Opt-in encodings that are still being evaluated. Every unstable edition stays a \
draft, so the family never freezes and carries no compatibility guarantee: a file written \
with these encodings is readable only by a build that knows them, and a later release may \
stop supporting one. The writer emits them only when the `unstable_encodings` feature is \
selected. An encoding graduates by joining a core edition.",
};

pub mod v2025_05;
pub mod v2026_02;
pub mod v2026_04;
pub mod v2026_06;

pub use v2025_05::UNSTABLE_2025_05_0;
pub use v2026_02::UNSTABLE_2026_02_0;
pub use v2026_04::UNSTABLE_2026_04_0;
pub use v2026_06::UNSTABLE_2026_06_0;
