// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `core` edition family: serialized components available to the default file writer.
//!
//! One module per edition, each declaring the edition and the serialized components that join the
//! family at it; members of earlier editions are inherited and never restated.

use crate::EditionFamily;

/// The `core` family: serialized components available by default.
pub static FAMILY: EditionFamily = EditionFamily {
    name: "core",
    origin: "vortex",
    doc: "The serialized components available to the default file writer. Each array ID names a \
wire representation that old readers either recognize or reject; several IDs may deserialize \
into one current in-memory array. Every core edition freezes, and a frozen edition carries a \
read-forever guarantee: a file written with it stays readable by every later Vortex release. \
New core objects first undergo testing in independently versioned families, then join preview for \
broad opt-in use, and finally join core with the same IDs and wire contracts. An edition may \
freeze in the release that cuts it; after that release version is known, the declaration is \
backfilled with it as the minimum. A frozen edition never changes.",
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
