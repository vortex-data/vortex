// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An experimental structural layout for extension-typed columns.
//!
//! An [`ExtensionLayout`] stores extension values as a layout of their storage, so a column of
//! extension values with struct storage is split into one column per storage field. Its single
//! child holds the storage, and the extension dtype is the layout dtype.
//!
//! Extension types are opaque: readers reach the storage only through the explicit
//! [`ext_storage`](vortex_array::expr::ext_storage) expression. An expression that reads the
//! extension root only inside `ext_storage(root)` is rebound to the storage and pushed down to the
//! child, so reading one storage field reads only that field, and the storage validity when the
//! extension is nullable. The planner-driven scan does not push expressions through a nullable
//! struct yet, so there it reads every column of a nullable struct storage. Any other use of the
//! root reads the whole storage and wraps it back into extension values.
//!
//! The writer decomposes only the extension types that opt in, see
//! [`TableStrategy::with_extension_layout`](crate::layouts::table::TableStrategy::with_extension_layout).

mod expr;
mod plan;
mod reader;
#[cfg(test)]
mod tests;
pub mod writer;

use std::sync::Arc;

pub use plan::ExtPack;
pub use plan::ExtPackData;
pub use plan::ExtPackPlan;
use reader::ExtensionReader;
use vortex_array::EmptyMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
pub use writer::ExtensionStrategy;

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
use crate::plan::physical::PlanRef;
use crate::segments::SegmentSource;

/// Slot of the storage child.
pub const STORAGE_SLOT: usize = 0;

/// Extension layout vtable.
#[derive(Clone, Debug)]
pub struct Extension;

/// An extension-typed layout whose only child holds the extension storage.
pub type ExtensionLayout = Layout<Extension>;

impl VTable for Extension {
    type LayoutData = ();
    type Metadata = EmptyMetadata;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.ext");
        *ID
    }

    fn metadata(_layout: &Layout<Self>) -> Self::Metadata {
        // The extension dtype, and with it the storage dtype, is the layout dtype.
        EmptyMetadata
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        _metadata: &EmptyMetadata,
    ) -> VortexResult<Self::LayoutData> {
        let ext_dtype = args
            .dtype
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("ExtensionLayout requires an extension dtype"))?;
        vortex_ensure_eq!(args.children.nchildren(), 1);
        let storage = args
            .children
            .child(STORAGE_SLOT, ext_dtype.storage_dtype())?;
        vortex_ensure_eq!(
            storage.row_count(),
            args.row_count,
            "ExtensionLayout storage row count does not match parent"
        );
        Ok(())
    }

    fn child_dtype(layout: &Layout<Self>, slot: usize) -> VortexResult<DType> {
        match slot {
            STORAGE_SLOT => Ok(layout.ext_dtype().storage_dtype().clone()),
            _ => vortex_bail!("Invalid child index {slot} for ExtensionLayout"),
        }
    }

    fn child_type(_layout: &Layout<Self>, slot: usize) -> LayoutChildType {
        match slot {
            STORAGE_SLOT => LayoutChildType::Transparent("storage".into()),
            _ => vortex_panic!("Invalid child index {slot} for ExtensionLayout"),
        }
    }

    fn new_reader(
        layout: &Layout<Self>,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(ExtensionReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session,
            ctx,
        )?))
    }

    fn new_plan(layout: &Layout<Self>) -> VortexResult<PlanRef> {
        // SAFETY: Extension layout construction validates that the only slot holds the storage
        // dtype of the layout's extension dtype, over the layout's rows.
        let plan = unsafe {
            ExtPack::from_slots_unchecked(
                layout.ext_dtype().clone(),
                layout.row_count(),
                slots_from_layout(&layout.to_layout()),
            )
        };
        Ok(plan.into_plan())
    }
}

impl Layout<Extension> {
    /// Construct an extension layout over a layout of its storage.
    ///
    /// # Errors
    ///
    /// Returns an error if `storage` does not have the storage dtype of `ext_dtype`.
    pub fn try_new(ext_dtype: ExtDTypeRef, storage: LayoutRef) -> VortexResult<Self> {
        vortex_ensure_eq!(storage.dtype(), ext_dtype.storage_dtype());
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

    /// Returns the extension dtype of this layout.
    pub fn ext_dtype(&self) -> &ExtDTypeRef {
        self.dtype()
            .as_extension_opt()
            .vortex_expect("ExtensionLayout dtype must be an extension")
    }

    /// Returns the layout of the extension storage.
    pub fn storage(&self) -> VortexResult<LayoutRef> {
        self.slot(STORAGE_SLOT)?
            .ok_or_else(|| vortex_err!("ExtensionLayout storage slot is absent"))
    }
}
