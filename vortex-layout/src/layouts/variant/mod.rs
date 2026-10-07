// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A layout that stores a shredded Variant column as separate columns.
//!
//! A canonical Variant array is a `core_storage` Variant holding the values not represented by
//! shredding, plus a row-aligned `shredded` tree of typed values for selected paths. Written as a
//! single leaf, every read of the column decodes the whole tree. [`VariantLayout`] instead writes
//! `core_storage` and the `shredded` tree as two children, and the `shredded` tree through the
//! table strategy, so its struct fields become separate columns and its partially shredded fields
//! become nested Variant layouts.
//!
//! Reading `variant_get` paths then touches only the shredded columns on those paths plus the core
//! storage, which the Variant `variant_get` kernel needs for rows the shredded values leave null.
//!
//! The layout is a draft component of the `variant` edition family: sessions write it only after
//! [`enable_variant_layout`] enables that edition, and every session can read it.

mod reader;
pub mod writer;

use std::sync::Arc;

use reader::VariantReader;
use vortex_array::DeserializeMetadata;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::proto::dtype as pb;
use vortex_edition::ComponentKind;
use vortex_edition::Edition;
use vortex_edition::EditionDeclaration;
use vortex_edition::EditionFamily;
use vortex_edition::EditionId;
use vortex_edition::EditionMember;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::SessionExt;
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
use crate::segments::SegmentSource;

/// The serialized id of [`VariantLayout`].
static VARIANT_LAYOUT_ID: CachedId = CachedId::new("vortex.variant");

/// The child holding the Variant core storage.
const CORE_CHILD: usize = 0;
/// The child holding the shredded tree.
const SHREDDED_CHILD: usize = 1;

/// The `variant` edition family: the shredded Variant layout.
pub static VARIANT_EDITION_FAMILY: EditionFamily = EditionFamily {
    name: "variant",
    origin: "vortex-layout",
    doc: "The shredded Variant layout. Readers of every session can read it; a writer emits it \
only once this family is enabled, while the layout is tested before joining preview.",
};

/// The October 2026 draft edition of the `variant` family.
pub const VARIANT_2026_10: EditionId = EditionId::new("variant", 2026, 10, 0);

/// The declaration of [`VARIANT_2026_10`]: a draft with no cross-version compatibility guarantee.
pub static VARIANT_EDITION_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: VARIANT_2026_10,
        min_library_version: None,
    },
    added: &[EditionMember::layout(&"vortex.variant")],
};

/// Declare and enable the draft `variant` edition, so writers of `session` store shredded
/// Variant columns as [`VariantLayout`]s. Idempotent.
pub fn enable_variant_layout(session: &VortexSession) -> VortexResult<()> {
    if session
        .editions()
        .find_family(VARIANT_EDITION_FAMILY.name)
        .is_none()
    {
        session
            .editions()
            .declare_family(&VARIANT_EDITION_FAMILY)
            .map_err(|error| vortex_err!("{error}"))?;
    }
    if session.editions().find(&VARIANT_2026_10).is_none() {
        session
            .register_edition(&VARIANT_EDITION_DECLARATION)
            .map_err(|error| vortex_err!("{error}"))?;
    }
    session.enable_edition(VARIANT_2026_10)
}

/// Whether the enabled editions of `session` permit writing [`VariantLayout`]s.
pub fn variant_layout_enabled(session: &VortexSession) -> bool {
    session
        .enabled_component_ids(ComponentKind::Layout)
        .contains(&*VARIANT_LAYOUT_ID)
}

/// Variant layout vtable.
#[derive(Clone, Debug)]
pub struct Variant;

/// A layout storing a Variant column's core storage and shredded tree as separate children.
pub type VariantLayout = Layout<Variant>;

/// Serialized [`VariantLayout`] metadata: the dtype of the shredded child.
#[derive(Clone, prost::Message)]
pub struct VariantLayoutMetadata {
    #[prost(message, optional, tag = "1")]
    shredded_dtype: Option<pb::DType>,
}

impl VTable for Variant {
    /// The dtype of the shredded tree.
    type LayoutData = DType;
    type Metadata = ProstMetadata<VariantLayoutMetadata>;

    fn id(&self) -> LayoutId {
        *VARIANT_LAYOUT_ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        ProstMetadata(VariantLayoutMetadata {
            shredded_dtype: Some(
                layout
                    .shredded_dtype()
                    .try_into()
                    .vortex_expect("shredded dtype serializes"),
            ),
        })
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
    ) -> VortexResult<Self::LayoutData> {
        vortex_ensure!(
            args.dtype.is_variant(),
            "Variant layout requires a Variant dtype, got {}",
            args.dtype
        );
        vortex_ensure_eq!(args.children.nchildren(), 2);
        for idx in [CORE_CHILD, SHREDDED_CHILD] {
            vortex_ensure_eq!(
                args.children.child_row_count(idx),
                args.row_count,
                "Variant layout child {idx} row count does not match parent"
            );
        }
        let shredded_dtype = metadata
            .shredded_dtype
            .as_ref()
            .ok_or_else(|| vortex_err!("Variant layout metadata is missing the shredded dtype"))?;
        DType::from_proto(shredded_dtype, args.session)
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        match slot {
            CORE_CHILD => Ok(layout.dtype().clone()),
            SHREDDED_CHILD => Ok(layout.shredded_dtype().clone()),
            _ => vortex_bail!("Variant layout has no child {slot}"),
        }
    }

    fn child_type(_layout: &Layout<Self>, slot: usize) -> LayoutChildType {
        if slot == CORE_CHILD {
            LayoutChildType::Transparent("core".into())
        } else {
            LayoutChildType::Auxiliary("shredded".into())
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(VariantReader::new(
            layout.clone(),
            name,
            segment_source,
            session.session(),
            ctx.clone(),
        )))
    }
}

impl Layout<Variant> {
    /// A Variant layout of `row_count` rows of `dtype`, from the layouts of its core storage and of
    /// its shredded tree of dtype `shredded_dtype`.
    pub fn new(
        row_count: u64,
        dtype: DType,
        shredded_dtype: DType,
        core: LayoutRef,
        shredded: LayoutRef,
    ) -> Self {
        LayoutParts::new(
            Variant,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(vec![core, shredded]),
            shredded_dtype,
        )
        .into_typed()
    }

    /// The dtype of the shredded tree.
    pub fn shredded_dtype(&self) -> &DType {
        self.data()
    }
}
