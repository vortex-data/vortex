// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A layout that stores a shredded Variant as its core storage and its shredded child.

mod reader;
pub mod writer;

use std::sync::Arc;

use prost::Message;
use reader::VariantReader;
use vortex_array::ProstMetadata;
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
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
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
use crate::children::OwnedLayoutChildren;
use crate::plan::physical::EvalPlan;
use crate::plan::physical::PlanRef;
use crate::segments::SegmentSource;

/// Name of the field of the stored struct that holds the core storage.
pub(crate) const CORE: &str = "core";

/// Name of the field of the stored struct that holds the shredded child.
pub(crate) const SHREDDED: &str = "shredded";

/// Variant layout vtable.
///
/// **Note**: this is an unstable and experimental layout that is expected to change. Using it may
/// lead to unreadable files in the future.
#[derive(Clone, Debug)]
pub struct Variant;

/// A layout that stores a shredded Variant column as a struct of two columns: the `core` storage
/// of its values and their `shredded` child.
///
/// The struct is written like any other, so the shredded child, usually a struct itself, becomes
/// a column per shredded field. A scan that extracts a shredded path then reads that column and
/// leaves the core storage, which holds the bulk of the bytes, unread.
pub type VariantLayout = Layout<Variant>;

/// What a [`VariantLayout`] knows besides its child.
#[derive(Clone, Debug)]
pub struct VariantLayoutData {
    /// The dtype of the stored struct.
    storage_dtype: DType,
    /// The dtype of the shredded child.
    shredded_dtype: DType,
    /// Whether the shredded child holds every value it could hold; see
    /// [`VariantPackOptions::complete`](vortex_array::scalar_fn::fns::variant_pack::VariantPackOptions::complete).
    complete: bool,
}

/// Serialized form of [`VariantLayoutData`].
#[derive(prost::Message)]
pub struct VariantLayoutMetadata {
    /// The dtype of the shredded child, as a serialized `vortex.dtype.DType`.
    #[prost(bytes, tag = "1")]
    shredded_dtype: Vec<u8>,
    #[prost(bool, tag = "2")]
    complete: bool,
}

/// Name of the field of the stored struct that holds the validity of a shredded struct.
pub(crate) const VALIDITY: &str = "validity";

/// Whether a shredded child of `shredded` dtype is stored without its validity, which is then a
/// column of its own.
///
/// A scan cannot push a field read through a nullable struct, so a shredded struct, which is
/// null in the rows that are not objects, is stored as a struct that is not nullable.
pub(crate) fn splits_validity(shredded: &DType) -> bool {
    shredded.is_struct() && shredded.is_nullable()
}

/// The dtype of the struct that stores a Variant of `dtype` shredded as `shredded`.
fn storage_dtype(dtype: &DType, shredded: DType) -> DType {
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

impl VTable for Variant {
    type LayoutData = VariantLayoutData;
    type Metadata = ProstMetadata<VariantLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.variant");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        let shredded = pb::DType::try_from(&layout.data().shredded_dtype)
            .map(|shredded| shredded.encode_to_vec())
            .unwrap_or_default();
        ProstMetadata(VariantLayoutMetadata {
            shredded_dtype: shredded,
            complete: layout.data().complete,
        })
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &VariantLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        vortex_ensure!(
            args.dtype.is_variant(),
            "Variant layout must have a Variant dtype, found {}",
            args.dtype
        );
        vortex_ensure_eq!(
            args.children.nchildren(),
            1,
            "Variant layout must have exactly one child"
        );
        vortex_ensure_eq!(
            args.children.child_row_count(0),
            args.row_count,
            "Variant layout child row count does not match parent"
        );
        let shredded = pb::DType::decode(metadata.shredded_dtype.as_slice())
            .map_err(|e| vortex_err!("invalid shredded dtype in Variant layout: {e}"))?;
        let shredded = DType::from_proto(&shredded, args.session)?;
        Ok(VariantLayoutData {
            storage_dtype: storage_dtype(args.dtype, shredded.clone()),
            shredded_dtype: shredded,
            complete: metadata.complete,
        })
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        match slot {
            0 => Ok(layout.data().storage_dtype.clone()),
            _ => vortex_bail!("Variant layout has no slot {slot}"),
        }
    }

    fn child_type(_layout: &Layout<Self>, _slot: usize) -> LayoutChildType {
        LayoutChildType::Transparent("storage".into())
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        let storage =
            layout
                .storage()?
                .new_reader(Arc::clone(&name), segment_source, session, ctx)?;
        Ok(Arc::new(VariantReader::new(
            name,
            layout.dtype().clone(),
            storage,
            layout.variant()?,
        )))
    }

    /// The Variant is the stored columns packed back together. An extraction over this plan
    /// folds into the pack, and from there into a read of the shredded column it names.
    fn new_plan(layout: &Layout<Self>) -> VortexResult<PlanRef> {
        let storage = layout.storage()?.new_plan()?;
        Ok(EvalPlan::try_new(layout.variant()?, storage)?.into_plan())
    }
}

impl VariantLayout {
    /// Creates a Variant layout of `dtype`, shredded as `shredded`, over the layout of its
    /// stored struct, `storage`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dtype` is not a Variant, or `storage` is not the struct that stores
    /// a Variant of `dtype` shredded as `shredded`.
    pub fn try_new(
        dtype: DType,
        shredded: DType,
        storage: LayoutRef,
        complete: bool,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            dtype.is_variant(),
            "Variant layout must have a Variant dtype, found {dtype}"
        );
        let storage_dtype = storage_dtype(&dtype, shredded.clone());
        vortex_ensure_eq!(
            storage.dtype(),
            &storage_dtype,
            "Variant layout storage does not store its dtype and shredded dtype"
        );
        Ok(LayoutParts::new(
            Variant,
            dtype,
            storage.row_count(),
            Vec::new(),
            OwnedLayoutChildren::layout_children(vec![storage]),
            VariantLayoutData {
                storage_dtype,
                shredded_dtype: shredded,
                complete,
            },
        )
        .into_typed())
    }

    /// The Variant, as an expression over the stored struct.
    fn variant(&self) -> VortexResult<BoundExpression> {
        let complete = self.data().complete;
        let variant = if splits_validity(&self.data().shredded_dtype) {
            variant_pack_with_validity(
                get_item(CORE, root()),
                get_item(SHREDDED, root()),
                get_item(VALIDITY, root()),
                complete,
            )
        } else {
            variant_pack(get_item(CORE, root()), get_item(SHREDDED, root()), complete)
        };
        variant.bind(&self.data().storage_dtype)
    }

    /// Returns the layout of the stored struct.
    pub fn storage(&self) -> VortexResult<LayoutRef> {
        self.children().child(0, &self.data().storage_dtype)
    }

    /// Whether the shredded child holds every value it could hold.
    pub fn complete(&self) -> bool {
        self.data().complete
    }
}
