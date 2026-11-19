// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stores a shredded Variant in a struct layout, as the struct of its core storage and its
//! shredded child.
//!
//! The struct is written like any other, so the shredded child, usually a struct itself, becomes
//! a column per shredded field. The struct layout reads the Variant as the pack of its columns,
//! so a scan that extracts a shredded path reads the column that holds it, and leaves the core
//! storage, which holds the bulk of the bytes, unread.
//!
//! **Note**: this is an unstable and experimental format that is expected to change. Using it may
//! lead to unreadable files in the future.

pub mod writer;

use prost::Message;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::get_item;
use vortex_array::expr::root;
use vortex_array::expr::variant_pack;
use vortex_array::expr::variant_pack_with_validity;
use vortex_array::proto::dtype as pb;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
pub use writer::FixedShredding;
pub use writer::NoShredding;
pub use writer::StrictShredding;
pub use writer::VariantShreddingPolicy;
pub use writer::VariantStrategy;

use crate::LayoutId;

/// Name of the field of the stored struct that holds the core storage.
pub(crate) const CORE: &str = "core";

/// Name of the field of the stored struct that holds the shredded child.
pub(crate) const SHREDDED: &str = "shredded";

/// Name of the field of the stored struct that holds the validity of a shredded struct.
pub(crate) const VALIDITY: &str = "validity";

/// The wire ID of a struct layout that stores a shredded Variant.
///
/// A struct layout of a struct is written as `vortex.struct`, whose contract is frozen. A struct
/// layout of a Variant is written under this ID, which is in the unstable edition.
pub fn layout_id() -> LayoutId {
    static ID: CachedId = CachedId::new("vortex.variant");
    *ID
}

/// Whether a shredded child of `shredded` dtype is stored without its validity, which is then a
/// column of its own.
///
/// A scan cannot push a field read through a nullable struct, so a shredded struct, which is
/// null in the rows that are not objects, is stored as a struct that is not nullable.
pub(crate) fn splits_validity(shredded: &DType) -> bool {
    shredded.is_struct() && shredded.is_nullable()
}

/// The dtype of the struct that stores a Variant of `dtype` shredded as `shredded`.
pub(crate) fn storage_dtype(dtype: &DType, shredded: DType) -> DType {
    let mut fields = vec![(FieldName::from(CORE), dtype.clone())];
    if splits_validity(&shredded) {
        fields.push((FieldName::from(SHREDDED), shredded.as_nonnullable()));
        fields.push((
            FieldName::from(VALIDITY),
            DType::Bool(Nullability::NonNullable),
        ));
    } else {
        fields.push((FieldName::from(SHREDDED), shredded));
    }
    DType::Struct(StructFields::from_iter(fields), Nullability::NonNullable)
}

/// How a struct layout stores a shredded Variant.
#[derive(Clone, Debug)]
pub struct VariantStorage {
    /// The dtype of the stored struct.
    storage_dtype: DType,
    /// The dtype of the shredded child.
    shredded_dtype: DType,
    /// Whether the shredded child holds every value it could hold; see
    /// [`VariantPackOptions::complete`](vortex_array::scalar_fn::fns::variant_pack::VariantPackOptions::complete).
    complete: bool,
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
        Ok(Self {
            storage_dtype: storage_dtype(dtype, shredded.clone()),
            shredded_dtype: shredded,
            complete,
        })
    }

    /// Reads how a Variant of `dtype` is stored from the `metadata` of its struct layout.
    pub(crate) fn deserialize(
        dtype: &DType,
        metadata: &StructLayoutMetadata,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let shredded = pb::DType::decode(metadata.shredded_dtype.as_slice())
            .map_err(|e| vortex_err!("invalid shredded dtype in struct layout: {e}"))?;
        let shredded = DType::from_proto(&shredded, session)?;
        Self::try_new(dtype, shredded, metadata.complete)
    }

    /// The metadata of the struct layout that stores the Variant.
    pub(crate) fn metadata(&self) -> StructLayoutMetadata {
        let shredded = pb::DType::try_from(&self.shredded_dtype)
            .map(|shredded| shredded.encode_to_vec())
            .unwrap_or_default();
        StructLayoutMetadata {
            shredded_dtype: shredded,
            complete: self.complete,
        }
    }

    /// The dtype of the stored struct.
    pub(crate) fn storage_dtype(&self) -> &DType {
        &self.storage_dtype
    }

    /// The Variant, as an expression over the stored struct.
    pub(crate) fn variant(&self) -> VortexResult<BoundExpression> {
        let variant = if splits_validity(&self.shredded_dtype) {
            variant_pack_with_validity(
                get_item(CORE, root()),
                get_item(SHREDDED, root()),
                get_item(VALIDITY, root()),
                self.complete,
            )
        } else {
            variant_pack(
                get_item(CORE, root()),
                get_item(SHREDDED, root()),
                self.complete,
            )
        };
        variant.bind(&self.storage_dtype)
    }
}

/// Serialized metadata of a struct layout.
///
/// A struct layout of a struct has no metadata, so this message is empty for it. A struct layout
/// of a Variant records how the Variant is shredded.
#[derive(prost::Message)]
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
