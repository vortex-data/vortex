// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

//! UUID extension type for Vortex.
//!
//! Provides a UUID extension type backed by `FixedSizeList(Primitive(U8), 16)` storage. Each UUID
//! is stored as 16 bytes in big-endian (network) byte order, matching [RFC 4122] and Arrow's
//! [canonical UUID extension].
//!
//! [`initialize`] registers the extension dtype and the Arrow import/export plugins that map it
//! to Arrow's `arrow.uuid` extension type over `FixedSizeBinary[16]` storage.
//!
//! [RFC 4122]: https://www.rfc-editor.org/rfc/rfc4122
//! [canonical UUID extension]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#uuid

mod arrow;
mod metadata;
mod vtable;

use std::sync::Arc;

pub use metadata::UuidMetadata;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_arrow::ArrowSessionExt;
use vortex_session::VortexSession;

/// The VTable for the UUID extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Uuid;

/// Register UUID extension support with a session.
///
/// This registers the `vortex.uuid` extension dtype along with the Arrow exporter and importer
/// for Arrow's canonical `arrow.uuid` extension type.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(Uuid);
    session.arrow().register_exporter(Arc::new(Uuid));
    session.arrow().register_importer(Arc::new(Uuid));
}
