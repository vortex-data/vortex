// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The Vortex edition declarations.
//!
//! [`vortex_edition`] provides the types, session variables, and test harness. The actual
//! first-party declarations live here, one module per edition. The default session first
//! registers them with [`crate::editions::register_default_editions`] and then selects its write
//! policy with [`crate::editions::enable_default_editions`].
//!
//! Members carry a [`crate::editions::ComponentKind`]: arrays a written array may use, extension
//! dtypes its schema may contain, and the aggregates zone maps record. Every kind is restricted to
//! its declared members, so an empty set permits no components of that kind.
//!
//! The default file writer resolves the session's enabled editions at write time. The
//! facade enables the newest frozen `core` edition, [`crate::editions::CORE_2026_08`], and
//! additionally enables the latest unstable edition when the `unstable_encodings` feature is
//! selected.

pub mod core;
#[cfg(test)]
mod tests;
pub mod unstable;

pub use vortex_edition::ComponentKind;
pub use vortex_edition::Edition;
pub use vortex_edition::EditionDeclaration;
pub use vortex_edition::EditionId;
pub use vortex_edition::EditionInclusion;
pub use vortex_edition::EditionMember;
pub use vortex_edition::EditionSession;
pub use vortex_edition::EditionSessionExt;
pub use vortex_edition::EnabledEditions;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

pub use self::core::CORE_2025_05_0;
pub use self::core::CORE_2025_06_0;
pub use self::core::CORE_2025_10_0;
pub use self::core::CORE_2026_07_0;
pub use self::core::CORE_2026_08;
pub use self::unstable::UNSTABLE_2025_05_0;
pub use self::unstable::UNSTABLE_2026_02_0;
pub use self::unstable::UNSTABLE_2026_04_0;
pub use self::unstable::UNSTABLE_2026_06_0;

/// The `core` edition enabled for writing by the default Vortex session.
pub const DEFAULT_CORE_EDITION: EditionId = CORE_2026_08;

/// The `unstable` edition enabled for writing by the default Vortex session when the
/// `unstable_encodings` feature is selected.
pub const DEFAULT_UNSTABLE_EDITION: EditionId = UNSTABLE_2026_06_0;

/// The first-party Vortex edition declarations.
pub static EDITION_DECLARATIONS: &[&EditionDeclaration] = &[
    &core::v2025_05::DECLARATION,
    &core::v2025_06::DECLARATION,
    &core::v2025_10::DECLARATION,
    &core::v2026_07::DECLARATION,
    &core::v2026_08::DECLARATION,
    &unstable::v2025_05::DECLARATION,
    &unstable::v2026_02::DECLARATION,
    &unstable::v2026_04::DECLARATION,
    &unstable::v2026_06::DECLARATION,
];

/// Register the Vortex edition declarations with the session's [`EditionSession`].
pub fn register_default_editions(session: &VortexSession) {
    for declaration in EDITION_DECLARATIONS {
        session
            .register_edition(declaration)
            .map_err(|e| vortex_err!("{e}"))
            .vortex_expect("edition declarations are valid");
    }
}

/// Enable the default Vortex editions for writing.
///
/// This selects the newest frozen `core` edition and, when configured, the newest unstable
/// edition. All declarations must have been registered first with
/// [`register_default_editions`].
pub fn enable_default_editions(session: &VortexSession) {
    session
        .enable_edition(DEFAULT_CORE_EDITION)
        .map_err(|e| vortex_err!("{e}"))
        .vortex_expect("default core edition is registered");

    #[cfg(feature = "unstable_encodings")]
    session
        .enable_edition(DEFAULT_UNSTABLE_EDITION)
        .map_err(|e| vortex_err!("{e}"))
        .vortex_expect("default unstable edition is registered");
}
