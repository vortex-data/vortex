// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A Variant layout stores two row-aligned children: core Variant storage and shredded values.
//!
//! The struct writer splits a shredded struct into a column per field. When a chunk's shredded
//! values are complete, a typed extraction can read the selected column directly. Nullable
//! Variants also need the core's SQL validity.
//!
//! **Note**: this is an unstable and experimental format that is expected to change. Using it may
//! lead to unreadable files in the future.

#[cfg(test)]
mod tests;
pub mod writer;

use std::sync::Arc;

use prost::Message;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::proto::dtype as pb;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_expr::get_item;
use vortex_expr::root;
use vortex_functions::expr::variant_pack;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
pub use writer::VariantStrategy;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::layouts::columnar::ColumnarReader;
use crate::layouts::columnar::Columns;
use crate::layouts::columnar::validate_child_rows;
use crate::layouts::struct_::Struct;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::PlanRef;
use crate::segments::SegmentSource;

/// Name of the field of the stored struct that holds the core storage.
pub(crate) const CORE: &str = "core";

/// Name of the field of the stored struct that holds the shredded child.
pub(crate) const SHREDDED: &str = "shredded";

fn storage_dtype(dtype: &DType, shredded: DType) -> DType {
    DType::struct_(
        [(CORE, dtype.clone()), (SHREDDED, shredded)],
        Nullability::NonNullable,
    )
}

/// Column schema and completeness for a shredded Variant chunk.
#[derive(Clone, Debug)]
pub struct VariantData {
    /// The dtype of the stored struct.
    storage_dtype: DType,
    metadata: VariantLayoutMetadata,
}

impl VariantData {
    /// Stores a Variant of `dtype`, shredded as `shredded`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dtype` is not a Variant.
    fn try_new(dtype: &DType, shredded: DType, complete: bool) -> VortexResult<Self> {
        vortex_ensure!(
            dtype.is_variant(),
            "Variant layout requires a Variant dtype, found {dtype}"
        );
        let metadata = VariantLayoutMetadata {
            shredded_dtype: pb::DType::try_from(&shredded)?.encode_to_vec(),
            complete,
        };
        Ok(Self {
            storage_dtype: storage_dtype(dtype, shredded),
            metadata,
        })
    }

    fn deserialize(
        dtype: &DType,
        metadata: &VariantLayoutMetadata,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let shredded = pb::DType::decode(metadata.shredded_dtype.as_slice())
            .map_err(|e| vortex_err!("invalid shredded dtype in Variant layout: {e}"))?;
        let shredded = DType::from_proto(&shredded, session)?;
        Ok(Self {
            storage_dtype: storage_dtype(dtype, shredded),
            metadata: metadata.clone(),
        })
    }

    /// The Variant, as an expression over the stored struct.
    fn projection(&self) -> VortexResult<BoundExpression> {
        variant_pack(
            get_item(CORE, root()),
            get_item(SHREDDED, root()),
            self.metadata.complete,
        )
        .bind(&self.storage_dtype)
    }
}

/// Serialized metadata of `vortex.variant`. Its two required children are core, then shredded.
#[derive(Clone, prost::Message)]
pub struct VariantLayoutMetadata {
    /// The dtype of the shredded child, as a serialized `vortex.dtype.DType`.
    #[prost(bytes, tag = "1")]
    shredded_dtype: Vec<u8>,
    /// Whether the shredded child holds every value it could hold.
    #[prost(bool, tag = "2")]
    complete: bool,
}

/// Layout vtable for a shredded Variant.
#[derive(Clone, Debug)]
pub struct Variant;

/// A Variant reconstructed from its core and shredded columns.
pub type VariantLayout = Layout<Variant>;

impl VTable for Variant {
    type LayoutData = VariantData;
    type Metadata = ProstMetadata<VariantLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.variant");
        *ID
    }

    fn metadata(layout: &VariantLayout) -> Self::Metadata {
        ProstMetadata(layout.data().metadata.clone())
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &VariantLayoutMetadata,
    ) -> VortexResult<VariantData> {
        vortex_ensure!(
            args.dtype.is_variant(),
            "Variant layout requires a Variant dtype, found {}",
            args.dtype
        );
        vortex_ensure!(
            args.children.nchildren() >= 2,
            "Variant layout needs core and shredded children"
        );
        validate_child_rows(args, 2)?;
        VariantData::deserialize(args.dtype, metadata, args.session)
    }

    fn nslots(layout: &VariantLayout) -> usize {
        layout.columns().nslots()
    }

    fn slot_to_child(layout: &VariantLayout, slot: usize) -> Option<usize> {
        layout.columns().slot_to_child(slot)
    }

    fn child_dtype(layout: &VariantLayout, slot: usize) -> VortexResult<DType> {
        layout.columns().child_dtype(slot)
    }

    fn child_type(layout: &VariantLayout, slot: usize) -> LayoutChildType {
        layout.columns().child_type(slot)
    }

    fn new_reader(
        layout: &VariantLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ColumnarReader::try_new(
            layout,
            &layout.data().storage_dtype,
            Some(layout.data().projection()?),
            name,
            segment_source,
            session,
            ctx,
        )?))
    }

    fn new_plan(layout: &VariantLayout) -> VortexResult<PlanRef> {
        let columns = layout.columns().pack_plan(&layout.to_layout());
        Ok(EvalPlan::try_new(layout.data().projection()?, columns)?.into_plan())
    }
}

impl VariantLayout {
    /// Reuse the two child layouts produced by the ordinary struct writer.
    fn from_storage(
        dtype: DType,
        shredded: DType,
        storage: &LayoutRef,
        complete: bool,
    ) -> VortexResult<Self> {
        let data = VariantData::try_new(&dtype, shredded, complete)?;
        let storage = storage
            .as_opt::<Struct>()
            .ok_or_else(|| vortex_err!("Variant storage writer must produce a struct layout"))?;
        vortex_ensure_eq!(
            storage.dtype(),
            &data.storage_dtype,
            "Variant storage columns have unexpected names or dtypes"
        );
        Ok(LayoutParts::new(
            Variant,
            dtype,
            storage.row_count(),
            Vec::new(),
            Arc::clone(storage.children()),
            data,
        )
        .into_typed())
    }

    fn columns(&self) -> Columns<'_> {
        Columns::of(&self.data().storage_dtype)
    }
}
