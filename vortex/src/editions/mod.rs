// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The Vortex edition declarations.
//!
//! [`vortex_edition`] provides the types, the [`EditionSession`] session variable, and the
//! test harness; the actual declarations live here, one module per edition
//! (`editions::<family>::<date>`), and are seeded into the default session by
//! [`register_default_editions`].
//!
//! Each edition module declares the edition together with the encodings that join the
//! family at it; members of earlier editions are inherited and never restated. Correctness
//! is enforced by unit tests: every edition module calls
//! [`vortex_edition::test_harness::validate_edition`] once from its `#[cfg(test)]` module,
//! and the computed set of a frozen edition is pinned by a golden test, so any change to
//! these declarations that alters a frozen set fails CI. New encodings are staged into the
//! newest draft edition.
//!
//! Note this is currently unused but a future PR will make this public and gate the writer behind
//! editions.

pub mod core;
pub mod unstable;

use vortex_edition::EditionDeclaration;
pub use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

pub use self::core::CORE_2025_05_0;
pub use self::core::CORE_2025_06_0;
pub use self::core::CORE_2025_10_0;
pub use self::core::CORE_2026_07_0;
pub use self::unstable::UNSTABLE_2025_05_0;
pub use self::unstable::UNSTABLE_2026_02_0;
pub use self::unstable::UNSTABLE_2026_04_0;
pub use self::unstable::UNSTABLE_2026_06_0;

/// The Vortex editions, each declared together with the encodings it adds.
pub static EDITION_DECLARATIONS: &[&EditionDeclaration] = &[
    &core::v2025_05::DECLARATION,
    &core::v2025_06::DECLARATION,
    &core::v2025_10::DECLARATION,
    &core::v2026_07::DECLARATION,
    &unstable::v2025_05::DECLARATION,
    &unstable::v2026_02::DECLARATION,
    &unstable::v2026_04::DECLARATION,
    &unstable::v2026_06::DECLARATION,
];

/// Register the Vortex edition declarations with the session's [`EditionSession`].
pub fn register_default_editions(session: &VortexSession) {
    let editions = session.editions();
    for declaration in EDITION_DECLARATIONS {
        editions
            .declare(declaration)
            .map_err(|e| vortex_err!("{e}"))
            .vortex_expect("edition declarations are valid");
    }
}
