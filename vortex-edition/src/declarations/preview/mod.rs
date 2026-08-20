// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `preview` edition family: opt-in components without a frozen compatibility guarantee.
//!
//! One module per draft edition, each declaring the components that join the family at it.
//! Members of earlier editions are inherited and never restated.

use crate::EditionFamily;

/// The `preview` family: opt-in components with no compatibility guarantee.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "preview",
    doc: "Opt-in components that are still being evaluated. Every preview edition stays a \
draft, so the family never freezes and carries no compatibility guarantee: a file written \
with these components is readable only by a build that knows them, and a later release may \
stop supporting one. The writer emits them only when the `unstable_encodings` feature is \
selected. A component graduates by joining a core edition.",
};

pub mod v2025_05;
pub mod v2026_02;
pub mod v2026_04;
pub mod v2026_06;

pub use v2025_05::PREVIEW_2025_05_0;
pub use v2026_02::PREVIEW_2026_02_0;
pub use v2026_04::PREVIEW_2026_04_0;
pub use v2026_06::PREVIEW_2026_06_0;
