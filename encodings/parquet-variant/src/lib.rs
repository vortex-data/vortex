// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex support for Arrow's canonical `arrow.parquet.variant` extension type.
//!
//! This encoding provides a lossless representation of semi-structured data stored as
//! [Parquet Variant values] inside Arrow columns. It also supports [shredded variant values],
//! allowing systems to pass variant-encoded data around without special handling unless they
//! need to inspect the encoded contents directly.
//!
//! The storage type is a `Struct` that follows the Arrow canonical extension contract:
//! - `metadata` (required): a non-nullable binary child containing variant metadata
//! - `value` (optional): a binary child containing unshredded variant bytes
//! - `typed_value` (optional): a shredded child with a primitive, list, or struct layout
//!
//! At least one of `value` or `typed_value` must be present. Nested shredded values recurse
//! through the same `value` and `typed_value` structure described by the canonical extension
//! type documentation.
//!
//! See the Arrow canonical extension docs for the storage rules, and the Parquet format
//! specification for the binary representation.
//!
//! [Parquet Variant values]: https://github.com/apache/parquet-format/blob/master/VariantEncoding.md
//! [shredded variant values]: https://github.com/apache/parquet-format/blob/master/VariantShredding.md
//! [Arrow canonical extension type]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#parquet-variant

mod array;
mod arrow;
#[cfg(test)]
mod json_to_variant_tests;
mod kernel;
mod operations;
mod validity;
mod vtable;

use std::sync::Arc;

pub use array::ParquetVariantArrayExt;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::session::ArraySessionExt;
pub use vortex_json::JsonToVariant;
pub use vortex_json::JsonToVariantOptions;
pub use vortex_json::ShreddingSpec;
pub use vortex_json::json_to_variant;
use vortex_session::VortexSession;
pub use vtable::ParquetVariant;
pub use vtable::ParquetVariantArray;

/// Register Parquet Variant array, Arrow extension, and scalar function support with a
/// session.
///
/// This also initializes [`vortex_json`], registering the `Json` extension dtype and the
/// `json_to_variant` scalar function whose execution this crate provides.
pub fn initialize(session: &VortexSession) {
    vortex_json::initialize(session);
    session.arrays().register(ParquetVariant);
    kernel::initialize(session);
    session.arrow().register_exporter(Arc::new(ParquetVariant));
    session.arrow().register_importer(Arc::new(ParquetVariant));
}
