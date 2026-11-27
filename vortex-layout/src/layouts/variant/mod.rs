// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stores a shredded Variant in a struct layout, as the struct of its core storage and its
//! shredded child.
//!
//! The struct is written like any other, so the shredded child, usually a struct itself, becomes
//! a column per shredded field. When a chunk's shredded values are complete, a typed extraction
//! can read the selected column directly. Nullable Variants also need the core's SQL validity.
//!
//! **Note**: this is an unstable and experimental format that is expected to change. Using it may
//! lead to unreadable files in the future.

pub mod writer;

use prost::Message;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::proto::dtype as pb;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_expr::get_item;
use vortex_expr::root;
use vortex_functions::expr::variant_pack;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
pub use writer::VariantStrategy;

use crate::LayoutId;

/// Name of the field of the stored struct that holds the core storage.
pub(crate) const CORE: &str = "core";

/// Name of the field of the stored struct that holds the shredded child.
pub(crate) const SHREDDED: &str = "shredded";

/// The wire ID of a struct layout that stores a shredded Variant.
///
/// A struct layout of a struct is written as `vortex.struct`, whose contract is frozen. A struct
/// layout of a Variant is written under this ID, which is in the unstable edition.
pub fn layout_id() -> LayoutId {
    static ID: CachedId = CachedId::new("vortex.variant");
    *ID
}

fn storage_dtype(dtype: &DType, shredded: DType) -> DType {
    DType::struct_(
        [(CORE, dtype.clone()), (SHREDDED, shredded)],
        Nullability::NonNullable,
    )
}

/// How a struct layout stores a shredded Variant.
#[derive(Clone, Debug)]
pub struct VariantStorage {
    /// The dtype of the stored struct.
    storage_dtype: DType,
    metadata: StructLayoutMetadata,
}

impl VariantStorage {
    /// Stores a Variant of `dtype`, shredded as `shredded`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dtype` is not a Variant.
    pub(crate) fn try_new(dtype: &DType, shredded: DType, complete: bool) -> VortexResult<Self> {
        vortex_ensure!(
            dtype.is_variant(),
            "a struct layout stores a shredded Variant only for a Variant dtype, found {dtype}"
        );
        let metadata = StructLayoutMetadata {
            shredded_dtype: pb::DType::try_from(&shredded)?.encode_to_vec(),
            complete,
        };
        Ok(Self {
            storage_dtype: storage_dtype(dtype, shredded),
            metadata,
        })
    }

    pub(crate) fn deserialize(
        dtype: &DType,
        metadata: &StructLayoutMetadata,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let shredded = pb::DType::decode(metadata.shredded_dtype.as_slice())
            .map_err(|e| vortex_err!("invalid shredded dtype in struct layout: {e}"))?;
        let shredded = DType::from_proto(&shredded, session)?;
        Ok(Self {
            storage_dtype: storage_dtype(dtype, shredded),
            metadata: metadata.clone(),
        })
    }

    pub(crate) fn metadata(&self) -> StructLayoutMetadata {
        self.metadata.clone()
    }

    /// The dtype of the stored struct.
    pub(crate) fn storage_dtype(&self) -> &DType {
        &self.storage_dtype
    }

    /// The Variant, as an expression over the stored struct.
    pub(crate) fn variant(&self) -> VortexResult<BoundExpression> {
        variant_pack(
            get_item(CORE, root()),
            get_item(SHREDDED, root()),
            self.metadata.complete,
        )
        .bind(&self.storage_dtype)
    }
}

/// Serialized metadata of a struct layout.
///
/// A struct layout of a struct has no metadata, so this message is empty for it. A struct layout
/// of a Variant records how the Variant is shredded.
#[derive(Clone, prost::Message)]
pub struct StructLayoutMetadata {
    /// The dtype of the shredded child, as a serialized `vortex.dtype.DType`.
    #[prost(bytes, tag = "1")]
    shredded_dtype: Vec<u8>,
    /// Whether the shredded child holds every value it could hold.
    #[prost(bool, tag = "2")]
    complete: bool,
}

impl StructLayoutMetadata {
    /// Whether the metadata records nothing, as for a struct layout of a struct.
    pub(crate) fn is_empty(&self) -> bool {
        self.shredded_dtype.is_empty() && !self.complete
    }
}
