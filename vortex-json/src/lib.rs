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

use std::sync::Arc;

pub use dtype::Json;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_session::VortexSession;

/// Register JSON extension support with a session.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Json);
    session.arrow().register_exporter(Arc::new(Json));
    session.arrow().register_importer(Arc::new(Json));
}
