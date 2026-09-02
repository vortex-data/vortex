// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The first-party Vortex edition declarations, one module per edition.
//!
//! These are plain constants naming components by id, so they depend on nothing but the types
//! in this crate. That keeps them cheap to read: tooling that only needs to know what an
//! edition contains — `cargo run -p xtask -- generate-editions`, for one — can depend on
//! this crate alone rather than on the whole of `vortex`.
//!
//! The `vortex` facade re-exports everything here and owns the session wiring: registering
//! the declarations and selecting which of them the default writer may emit.

pub mod core;
pub mod preview;

use crate::EditionDeclaration;
use crate::EditionFamily;

/// The first-party edition families. Every family must be declared before its editions.
pub static EDITION_FAMILIES: &[&EditionFamily] = &[&core::FAMILY, &preview::FAMILY];

/// The first-party Vortex edition declarations.
pub static EDITION_DECLARATIONS: &[&EditionDeclaration] = &[
    &core::v2025_05::DECLARATION,
    &core::v2025_06::DECLARATION,
    &core::v2025_10::DECLARATION,
    &core::v2026_08::DECLARATION_0,
    &core::v2026_08::DECLARATION_1,
    &core::v2026_08_2::DECLARATION,
    &core::v2026_08_3::DECLARATION,
    &preview::v2026_08::DECLARATION,
];
