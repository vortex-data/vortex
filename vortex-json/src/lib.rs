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

use std::sync::Arc;

pub mod editions;
pub use dtype::Json;
pub use json_to_variant::JsonToVariant;
pub use json_to_variant::JsonToVariantOptions;
pub use json_to_variant::ShreddingSpec;
pub use json_to_variant::json_to_variant;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_arrow::ArrowSessionExt;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

/// Register JSON extension support with a session.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Json);
    session.arrow().register_exporter(Arc::new(Json));
    session.arrow().register_importer(Arc::new(Json));
    session.scalar_fns().register(JsonToVariant);

    // JSON is an opt-in durable dtype, so it belongs to an independently enabled edition family.
    // `initialize` is idempotent, hence the guard around declaration registration.
    if session.editions().find(&editions::JSON_2026_08).is_none() {
        session
            .register_edition(&editions::DECLARATION)
            .map_err(|error| vortex_err!("{error}"))
            .vortex_expect("JSON edition declaration is valid");
    }
    session
        .enable_edition(editions::JSON_2026_08)
        .map_err(|error| vortex_err!("{error}"))
        .vortex_expect("JSON edition is registered");
}
