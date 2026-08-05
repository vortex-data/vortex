// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `core` edition family: the encodings the default file writer emits.
//!
//! One module per edition, each declaring the edition and the encodings that join the
//! family at it; members of earlier editions are inherited and never restated.

pub mod v2025_05;
pub mod v2025_06;
pub mod v2025_10;
pub mod v2026_07;
pub mod v2026_08_1;

pub use v2025_05::CORE_2025_05_0;
pub use v2025_06::CORE_2025_06_0;
pub use v2025_10::CORE_2025_10_0;
pub use v2026_07::CORE_2026_07_0;
pub use v2026_08_1::CORE_2026_08_1;
