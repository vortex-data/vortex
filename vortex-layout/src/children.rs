use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use flatbuffers::Follow;
use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_err, vortex_panic};
use vortex_flatbuffers::{FlatBuffer, layout as fbl};

use crate::segments::SegmentId;
use crate::{LayoutContext, LayoutRef};

/// Abstract way of accessing the children of a layout.
///
/// This allows us to abstract over the lazy flatbuffer-based layouts, as well as the in-memory
/// layout trees.
pub trait LayoutChildren: 'static + Send + Sync {
    fn to_arc(&self) -> Arc<dyn LayoutChildren>;

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef>;

    fn child_row_count(&self, idx: usize) -> u64;

    fn nchildren(&self) -> usize;
}

impl Debug for dyn LayoutChildren {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayoutChildren")
            .field("nchildren", &self.nchildren())
            .finish()
    }
}

impl LayoutChildren for Arc<dyn LayoutChildren> {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        self.clone()
    }

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef> {
        self.as_ref().child(idx, dtype)
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        self.as_ref().child_row_count(idx)
    }

    fn nchildren(&self) -> usize {
        self.as_ref().nchildren()
    }
}

/// An implementation of [`LayoutChildren`] for in-memory owned children.
/// See also [`ViewLayoutChildren`] for lazily deserialized children from flatbuffers.
#[derive(Clone)]
pub(crate) struct OwnedLayoutChildren(Vec<LayoutRef>);

impl OwnedLayoutChildren {
    pub fn layout_children(children: Vec<LayoutRef>) -> Arc<dyn LayoutChildren> {
        Arc::new(Self(children))
    }
}

/// In-memory implementation of [`LayoutChildren`].
impl LayoutChildren for OwnedLayoutChildren {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        Arc::new(self.clone())
    }

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef> {
        if idx >= self.0.len() {
            vortex_bail!("Child index out of bounds: {} of {}", idx, self.0.len());
        }
        let child = &self.0[idx];
        if child.dtype() != dtype {
            vortex_panic!("Child dtype mismatch: {} != {}", child.dtype(), dtype);
        }
        Ok(child.clone())
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        self.0[idx].row_count()
    }

    fn nchildren(&self) -> usize {
        self.0.len()
    }
}

#[derive(Clone)]
pub(crate) struct ViewedLayoutChildren {
    flatbuffer: FlatBuffer,
    flatbuffer_loc: usize,
    ctx: LayoutContext,
}

impl ViewedLayoutChildren {
    /// Create a new [`ViewedLayoutChildren`] from the given parameters.
    ///
    /// # Safety
    ///
    /// Assumes the flatbuffer is validated and that the `flatbuffer_loc` is the correct offset
    pub(super) unsafe fn new_unchecked(
        flatbuffer: FlatBuffer,
        flatbuffer_loc: usize,
        ctx: LayoutContext,
    ) -> Self {
        Self {
            flatbuffer,
            flatbuffer_loc,
            ctx,
        }
    }

    /// Return the flatbuffer layout message.
    fn flatbuffer(&self) -> fbl::Layout<'_> {
        unsafe { fbl::Layout::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
    }
}

impl LayoutChildren for ViewedLayoutChildren {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        Arc::new(self.clone())
    }

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef> {
        if idx >= self.nchildren() {
            vortex_bail!("Child index out of bounds: {} of {}", idx, self.nchildren());
        }
        let fb_child = self.flatbuffer().children().unwrap_or_default().get(idx);

        let viewed_children = ViewedLayoutChildren {
            flatbuffer: self.flatbuffer.clone(),
            flatbuffer_loc: fb_child._tab.loc(),
            ctx: self.ctx.clone(),
        };
        let encoding = self
            .ctx
            .lookup_encoding(fb_child.encoding())
            .ok_or_else(|| vortex_err!("Encoding not found: {}", fb_child.encoding()))?;

        encoding.build(
            dtype,
            fb_child.row_count(),
            fb_child
                .metadata()
                .map(|m| m.bytes())
                .unwrap_or_else(|| &[]),
            fb_child
                .segments()
                .unwrap_or_default()
                .iter()
                .map(SegmentId::from)
                .collect_vec(),
            &viewed_children,
        )
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        // Efficiently get the row count of the child at the given index, without a full
        // deserialization.
        self.flatbuffer()
            .children()
            .unwrap_or_default()
            .get(idx)
            .row_count()
    }

    fn nchildren(&self) -> usize {
        self.flatbuffer().children().unwrap_or_default().len()
    }
}
