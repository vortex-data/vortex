// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]

//! File reference extension type for Vortex.
//!
//! A file reference points at a range of bytes, stored inline in the value or in an external
//! file, such as an image or audio clip in object storage. Its fields and their meanings follow
//! the [Parquet FILE logical type], and it maps to the canonical Arrow `arrow.parquet.file`
//! extension type.
//!
//! The storage is a struct of nullable fields, of which a column defines at least one and only
//! those it uses, in any order:
//!
//! | Field          | Storage dtype |
//! |----------------|---------------|
//! | `uri`          | `utf8?`       |
//! | `offset`       | `i64?`        |
//! | `size`         | `i64?`        |
//! | `content_type` | `utf8?`       |
//! | `checksum`     | `utf8?`       |
//! | `inline`       | `binary?`     |
//!
//! The defined fields and their order come from the storage dtype, so the serialized metadata
//! is empty.
//!
//! The Arrow canonical extension type is not yet ratified, so [`initialize`] adds the
//! `vortex.file_ref` dtype to the [`UNSTABLE`](vortex_edition::UNSTABLE) edition, which has no
//! compatibility guarantee. A session writes the dtype only after enabling that edition.
//!
//! [Parquet FILE logical type]: https://github.com/apache/parquet-format/blob/master/LogicalTypes.md

mod arrow;
mod dtype;
mod metadata;

pub use arrow::ARROW_FILE_EXTENSION_NAME;
pub use dtype::FileRefValue;
pub use metadata::FileRefField;
pub use metadata::FileRefMetadata;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_arrow::ArrowSessionExt;
use vortex_edition::ComponentKind;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSessionExt;
use vortex_edition::UNSTABLE;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

/// The VTable for the file reference extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FileRef;

/// Register file reference support with a session.
///
/// This registers the `vortex.file_ref` extension dtype, its Arrow import and export as the
/// canonical `arrow.parquet.file` extension type, and its membership of the
/// [`UNSTABLE`](vortex_edition::UNSTABLE) edition. It does not enable that edition.
pub fn initialize(session: &VortexSession) {
    session.dtypes().register(FileRef);
    session.arrow().register_extension(FileRef);

    // `initialize` is idempotent, but an inclusion may only be declared once.
    let id = FileRef.id();
    let declared = session
        .editions()
        .components_in(&UNSTABLE, ComponentKind::DType)
        .iter()
        .any(|inclusion| inclusion.component_id == id);
    if !declared {
        session
            .editions()
            .declare_inclusion(EditionInclusion::new(ComponentKind::DType, &id, UNSTABLE))
            .vortex_expect("the file reference dtype joins the unstable edition once");
    }
}

#[cfg(test)]
mod tests;
