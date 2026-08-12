// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `core` edition family: the serialized components the default file writer emits.
//!
//! One module per edition, each declaring the edition and the components that join the
//! family at it; members of earlier editions are inherited and never restated.

use crate::EditionFamily;

/// The `core` family: what the default writer may emit.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "core",
    doc: "The encodings the default file writer emits. Every core edition freezes, and a \
frozen edition carries a read-forever guarantee: a file written with it stays readable by \
every later Vortex release. New encodings join by being declared in a new edition; an \
edition that has frozen never changes again.",
};

pub mod v2025_05;
pub mod v2025_06;
pub mod v2025_10;
pub mod v2026_08;
pub mod v2026_08_2;
pub mod v2026_08_3;

pub use v2025_05::CORE_2025_05_0;
pub use v2025_06::CORE_2025_06_0;
pub use v2025_10::CORE_2025_10_0;
pub use v2026_08::CORE_2026_08_0;
pub use v2026_08::CORE_2026_08_1;
pub use v2026_08_2::CORE_2026_08_2;
pub use v2026_08_3::CORE_2026_08_3;
