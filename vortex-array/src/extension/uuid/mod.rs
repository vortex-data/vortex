// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UUID extension type for Vortex.
//!
//! Provides a UUID extension type backed by `FixedSizeList(Primitive(U8), 16)` storage. Each UUID
//! is stored as 16 bytes in big-endian (network) byte order, matching [RFC 4122] and Arrow's
//! [canonical UUID extension].
//!
//! [RFC 4122]: https://www.rfc-editor.org/rfc/rfc4122
//! [canonical UUID extension]: https://arrow.apache.org/docs/format/CanonicalExtensions.html#uuid

mod metadata;
pub use metadata::UuidMetadata;

pub(crate) mod vtable;

use std::sync::Arc;

use vortex_error::VortexExpect;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;

/// The VTable for the UUID extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Uuid;

#[expect(
    clippy::cast_possible_truncation,
    reason = "UUID_BYTE_LEN always fits u32"
)]
#[allow(clippy::same_name_method)]
impl Uuid {
    /// Returns the canonical UUID storage dtype: `FixedSizeList(Primitive(U8, NonNullable), 16)`.
    pub fn storage_dtype(nullability: Nullability) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            vtable::UUID_BYTE_LEN as u32,
            nullability,
        )
    }

    /// Creates a new UUID extension dtype with the given metadata and nullability.
    pub fn new(metadata: UuidMetadata, nullability: Nullability) -> ExtDType<Self> {
        ExtDType::try_new(metadata, Self::storage_dtype(nullability))
            .vortex_expect("valid UUID storage dtype")
    }

    /// Creates a new UUID extension dtype with default metadata.
    pub fn default(nullability: Nullability) -> ExtDType<Self> {
        Self::new(UuidMetadata::default(), nullability)
    }
}
