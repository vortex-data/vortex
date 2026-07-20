// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `core` edition family: the encodings the default file writer emits.
//!
//! One module per edition, each declaring the edition and the encodings that join the
//! family at it; members of earlier editions are inherited and never restated.

pub mod v2026_01;
pub mod v2026_07;

pub use v2026_01::CORE_2026_01_0;
pub use v2026_07::CORE_2026_07_0;
