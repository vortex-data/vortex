// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::layout as fbl;
use vortex_session::registry::ReadContext;

use crate::LayoutRef;
use crate::flatbuffers::build_layout_from_path;
use crate::session::LayoutRegistry;

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
            vortex_bail!("Child dtype mismatch: {} != {}", child.dtype(), dtype);
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
struct ViewedLayoutChild {
    path: Arc<[usize]>,
    row_count: u64,
}

#[derive(Clone)]
pub(crate) struct ViewedLayoutChildren {
    flatbuffer: FlatBuffer,
    children: Arc<[ViewedLayoutChild]>,
    array_read_ctx: ReadContext,
    layout_read_ctx: ReadContext,
    layouts: LayoutRegistry,
}

impl ViewedLayoutChildren {
    pub(super) fn new(
        flatbuffer: FlatBuffer,
        parent_path: &[usize],
        parent: fbl::LayoutRef<'_>,
        array_read_ctx: ReadContext,
        layout_read_ctx: ReadContext,
        layouts: LayoutRegistry,
    ) -> VortexResult<Self> {
        let children: Arc<[ViewedLayoutChild]> = parent
            .children()?
            .map(|children| {
                children
                    .iter()
                    .enumerate()
                    .map(|(idx, child)| {
                        let child = child?;
                        let mut path = parent_path.to_vec();
                        path.push(idx);
                        Ok::<ViewedLayoutChild, vortex_flatbuffers::planus::Error>(
                            ViewedLayoutChild {
                                path: path.into(),
                                row_count: child.row_count()?,
                            },
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default()
            .into();

        Ok(Self {
            flatbuffer,
            children,
            array_read_ctx,
            layout_read_ctx,
            layouts,
        })
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
        build_layout_from_path(
            self.flatbuffer.clone(),
            self.children[idx].path.as_ref(),
            dtype,
            &self.layout_read_ctx,
            &self.array_read_ctx,
            &self.layouts,
        )
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        self.children[idx].row_count
    }

    fn nchildren(&self) -> usize {
        self.children.len()
    }
}
