// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

//! Extension type and related functionality for a JSON extension type for Vortex.

mod arrow;
mod dtype;
mod json_to_variant;
mod text;

pub mod editions;
pub mod proto;
pub use dtype::Json;
pub use json_to_variant::JsonToVariant;
pub use json_to_variant::JsonToVariantOptions;
pub use json_to_variant::ShreddingSpec;
pub use json_to_variant::json_to_variant;
pub use text::format_json;
pub use text::parse_json;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_arrow::ArrowSessionExt;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

/// Register JSON extension support with a session.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Json);
    session.arrow().register_extension(Json);
    session.scalar_fns().register(JsonToVariant);
    session.scalar_fns().register(text::JsonText);

    // JSON is an opt-in durable dtype, so it belongs to an independently enabled edition family.
    // `initialize` is idempotent, hence the guard around declaration registration.
    if session.editions().find(&editions::JSON_2026_08).is_none() {
        session
            .editions()
            .declare_family(&editions::FAMILY)
            .vortex_expect("JSON edition family is valid");
        session
            .register_edition(&editions::DECLARATION)
            .vortex_expect("JSON edition declaration is valid");
    }
    session
        .enable_edition(editions::JSON_2026_08)
        .vortex_expect("JSON edition is registered");
}

/// A session with JSON support registered, shared by the unit tests.
#[cfg(test)]
static SESSION: std::sync::LazyLock<VortexSession> = std::sync::LazyLock::new(|| {
    let session = VortexSession::empty();
    initialize(&session);
    session
});

#[cfg(test)]
mod arrow_properties;
