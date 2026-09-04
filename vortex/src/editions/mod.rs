// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The Vortex edition declarations.
//!
//! [`vortex_edition`] provides the types, session variables, test harness, and the
//! first-party declarations themselves. This module re-exports them and owns the session
//! wiring: the default session first registers them with
//! [`crate::editions::register_default_editions`] and then selects its write policy with
//! [`crate::editions::enable_default_editions`].
//!
//! Members carry a [`crate::editions::ComponentKind`]: serialized array IDs a writer may emit,
//! extension dtypes its schema may contain, and aggregates zone maps record. Array serializers
//! choose a wire representation independently of the enabled editions; the serialization context
//! rejects an ID that is not permitted unless the writer explicitly disables edition enforcement.
//!
//! The default file writer resolves the session's enabled editions at write time. The
//! facade enables [`crate::editions::CORE_2026_08_3`] and
//! additionally enables the `preview` edition when the
//! `unstable_encodings` feature is selected.

#[cfg(test)]
mod tests;

pub use vortex_edition::ComponentKind;
pub use vortex_edition::EDITION_DECLARATIONS;
pub use vortex_edition::EDITION_FAMILIES;
pub use vortex_edition::Edition;
pub use vortex_edition::EditionDeclaration;
pub use vortex_edition::EditionFamily;
pub use vortex_edition::EditionId;
pub use vortex_edition::EditionInclusion;
pub use vortex_edition::EditionMember;
pub use vortex_edition::EditionSession;
pub use vortex_edition::EditionSessionExt;
pub use vortex_edition::EnabledEditions;
pub use vortex_edition::declarations::core;
pub use vortex_edition::declarations::core::CORE_2025_05_0;
pub use vortex_edition::declarations::core::CORE_2025_06_0;
pub use vortex_edition::declarations::core::CORE_2025_10_0;
pub use vortex_edition::declarations::core::CORE_2026_08_0;
pub use vortex_edition::declarations::core::CORE_2026_08_1;
pub use vortex_edition::declarations::core::CORE_2026_08_2;
pub use vortex_edition::declarations::core::CORE_2026_08_3;
pub use vortex_edition::declarations::preview;
pub use vortex_edition::declarations::preview::PREVIEW_2026_08_0;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

/// The `core` edition enabled for writing by the default Vortex session.
pub const DEFAULT_CORE_EDITION: EditionId = CORE_2026_08_3;

/// The `preview` edition enabled for writing by the default Vortex session when the
/// `unstable_encodings` feature is selected.
pub const DEFAULT_PREVIEW_EDITION: EditionId = PREVIEW_2026_08_0;

/// Register the Vortex edition families and declarations with the session's
/// [`EditionSession`].
pub fn register_default_editions(session: &VortexSession) {
    for family in EDITION_FAMILIES {
        session
            .editions()
            .declare_family(family)
            .map_err(|e| vortex_err!("{e}"))
            .vortex_expect("edition families are valid");
    }
    for declaration in EDITION_DECLARATIONS {
        session
            .register_edition(declaration)
            .map_err(|e| vortex_err!("{e}"))
            .vortex_expect("edition declarations are valid");
    }
}

/// Enable the default Vortex editions for writing.
///
/// This selects the default `core` edition and, when configured, the `preview` edition. All
/// declarations must have been registered first with
/// [`register_default_editions`].
pub fn enable_default_editions(session: &VortexSession) {
    session
        .enable_edition(DEFAULT_CORE_EDITION)
        .map_err(|e| vortex_err!("{e}"))
        .vortex_expect("default core edition is registered");

    #[cfg(feature = "unstable_encodings")]
    session
        .enable_edition(DEFAULT_PREVIEW_EDITION)
        .map_err(|e| vortex_err!("{e}"))
        .vortex_expect("feature edition is registered");
}
