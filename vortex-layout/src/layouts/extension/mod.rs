// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A layout for struct-backed extension types.
//!
//! [`ExtensionLayout`] preserves the logical dtype around a single storage child. This lets
//! [`ext_storage`](vortex_array::expr::ext_storage) use field projection and pruning while
//! whole-value reads reconstruct the extension from the storage.
//!
//! The default file writer uses this experimental format for opted-in types in the unstable
//! edition. The wire format can change under the same layout ID.

mod expr;
mod plan;
mod reader;
pub(super) mod writer;

use std::sync::Arc;

use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

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
use crate::lowering::slots_from_layout;
use crate::plan::physical::PlanParts;
use crate::plan::physical::PlanRef;
use crate::segments::SegmentSource;

const STORAGE_SLOT: usize = 0;

/// Vtable for the experimental `vortex.extension` layout.
#[derive(Clone, Debug)]
pub struct Extension;

/// An extension-typed layout with one child of its struct storage dtype.
pub type ExtensionLayout = Layout<Extension>;

impl ExtensionLayout {
    /// Wraps a storage layout, retaining its row count and exposing the extension dtype.
    ///
    /// # Errors
    ///
    /// The extension must have struct storage, and the child must have the declared storage
    /// dtype and nullability.
    pub fn try_new(ext_dtype: ExtDTypeRef, storage: LayoutRef) -> VortexResult<Self> {
        vortex_ensure_eq!(storage.dtype(), struct_storage_dtype(&ext_dtype)?);

        Ok(LayoutParts::new(
            Extension,
            DType::Extension(ext_dtype),
            storage.row_count(),
            Vec::new(),
            OwnedLayoutChildren::layout_children(vec![storage]),
            (),
        )
        .into_typed())
    }

    /// Returns the extension dtype retained by the layout.
    pub fn ext_dtype(&self) -> &ExtDTypeRef {
        self.dtype()
            .as_extension_opt()
            .vortex_expect("extension dtype is validated when building the extension layout")
    }

    /// Returns the storage child, which has the same row range as this layout.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage child cannot be loaded.
    pub fn storage(&self) -> VortexResult<LayoutRef> {
        self.slot(STORAGE_SLOT)?
            .ok_or_else(|| vortex_err!("extension layout requires a storage child"))
    }
}

impl VTable for Extension {
    type LayoutData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.extension");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        EmptyMetadata
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        let ext_dtype = args.dtype.as_extension_opt().ok_or_else(|| {
            vortex_err!(
                "extension layout requires an extension dtype, got {}",
                args.dtype
            )
        })?;
        struct_storage_dtype(ext_dtype)?;

        vortex_ensure_eq!(
            args.segment_ids.len(),
            0,
            "extension layout must have no direct segments"
        );
        vortex_ensure_eq!(
            args.children.nchildren(),
            1,
            "extension layout must have exactly one storage child"
        );
        vortex_ensure_eq!(
            args.children.child_row_count(STORAGE_SLOT),
            args.row_count,
            "extension layout storage row count must match the parent"
        );

        Ok(())
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        vortex_ensure_eq!(slot, STORAGE_SLOT);
        Ok(layout.ext_dtype().storage_dtype().clone())
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
        Ok(Arc::new(reader::ExtensionReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session,
            ctx,
        )?))
    }

    fn new_plan(layout: &Layout<Self>) -> VortexResult<PlanRef> {
        Ok(PlanParts {
            vtable: plan::ExtensionPack,
            dtype: layout.dtype().clone(),
            row_count: layout.row_count(),
            slots: slots_from_layout(&layout.to_layout()),
            data: (),
        }
        .into_plan())
    }
}

fn struct_storage_dtype(ext_dtype: &ExtDTypeRef) -> VortexResult<&DType> {
    let storage_dtype = ext_dtype.storage_dtype();
    vortex_ensure!(
        storage_dtype.is_struct(),
        "extension layout requires struct storage, got {storage_dtype}"
    );
    Ok(storage_dtype)
}

#[cfg(test)]
mod tests;
