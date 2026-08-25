// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `preview` edition family: opt-in components without a frozen compatibility guarantee.
//!
//! One module per draft edition, each declaring the components that join the family at it.
//! Members of earlier editions are inherited and never restated.

pub mod v2025_05;
pub mod v2026_02;
pub mod v2026_04;
pub mod v2026_06;

pub use v2025_05::PREVIEW_2025_05_0;
pub use v2026_02::PREVIEW_2026_02_0;
pub use v2026_04::PREVIEW_2026_04_0;
pub use v2026_06::PREVIEW_2026_06_0;
