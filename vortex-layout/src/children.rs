// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use flatbuffers::Follow;
use itertools::Itertools;
use once_cell::sync::OnceCell;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::layout as fbl;
use vortex_layout_commons::LayoutBuildContext;
use vortex_layout_commons::LayoutChildren;
use vortex_layout_commons::LayoutRef;
pub(crate) use vortex_layout_commons::OwnedLayoutChildren;
use vortex_layout_commons::segments::SegmentId;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::layouts::foreign::new_foreign_layout;
use crate::session::LayoutRegistry;

#[derive(Clone)]
pub(crate) struct ViewedLayoutChildren {
    flatbuffer: FlatBuffer,
    flatbuffer_loc: usize,
    array_read_ctx: ReadContext,
    layout_read_ctx: ReadContext,
    layouts: LayoutRegistry,
    allow_unknown: bool,
    session: VortexSession,
    cache: Arc<[OnceCell<LayoutRef>]>,
    /// Per-child answers to [`LayoutChildren::child_is_indivisible`], precomputed at
    /// construction from the flatbuffer encoding tags alone.
    indivisible: Arc<[bool]>,
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
        array_read_ctx: ReadContext,
        layout_read_ctx: ReadContext,
        layouts: LayoutRegistry,
        allow_unknown: bool,
        session: VortexSession,
    ) -> Self {
        // SAFETY: guaranteed by caller
        let fb_children = unsafe { fbl::Layout::follow(flatbuffer.as_ref(), flatbuffer_loc) }
            .children()
            .unwrap_or_default();
        let cache = vec![OnceCell::new(); fb_children.len()]
            .into_boxed_slice()
            .into();
        // Unknown encodings are conservatively not indivisible so callers fall back to
        // materializing the child.
        let indivisible = fb_children
            .iter()
            .map(|child| {
                layout_read_ctx
                    .resolve(child.encoding())
                    .and_then(|encoding_id| layouts.get(&encoding_id))
                    .is_some_and(|encoding| encoding.is_indivisible())
            })
            .collect::<Arc<[bool]>>();
        Self {
            flatbuffer,
            flatbuffer_loc,
            array_read_ctx,
            layout_read_ctx,
            layouts,
            allow_unknown,
            session,
            cache,
            indivisible,
        }
    }

    /// Return the flatbuffer layout message.
    fn flatbuffer(&self) -> fbl::Layout<'_> {
        // SAFETY: flatbuffer_loc is guaranteed to be a valid offset into the flatbuffer
        // as it was constructed from a validated flatbuffer in ViewedLayoutChildren::try_new.
        // The lifetime of the returned Layout is tied to self, ensuring the buffer remains valid.
        unsafe { fbl::Layout::follow(self.flatbuffer.as_ref(), self.flatbuffer_loc) }
    }

    fn foreign_layout_from_fb(
        &self,
        fb_layout: fbl::Layout<'_>,
        dtype: &DType,
    ) -> VortexResult<LayoutRef> {
        let encoding_id = self
            .layout_read_ctx
            .resolve(fb_layout.encoding())
            .ok_or_else(|| vortex_err!("Encoding not found: {}", fb_layout.encoding()))?;

        let children = fb_layout
            .children()
            .unwrap_or_default()
            .iter()
            .map(|child| self.foreign_layout_from_fb(child, dtype))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(new_foreign_layout(
            encoding_id,
            dtype.clone(),
            fb_layout.row_count(),
            fb_layout
                .metadata()
                .map(|m| m.bytes().to_vec())
                .unwrap_or_default(),
            fb_layout
                .segments()
                .unwrap_or_default()
                .iter()
                .map(SegmentId::from)
                .collect_vec(),
            children,
        ))
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

        let layout_ref = self.cache[idx].get_or_try_init(|| {
            let fb_child = self.flatbuffer().children().unwrap_or_default().get(idx);

            // SAFETY: same validated flatbuffer; fb_child._tab.loc() is a valid offset
            // We need this to avoid re-initializing cache here
            let viewed_children = unsafe {
                ViewedLayoutChildren::new_unchecked(
                    self.flatbuffer.clone(),
                    fb_child._tab.loc(),
                    self.array_read_ctx.clone(),
                    self.layout_read_ctx.clone(),
                    self.layouts.clone(),
                    self.allow_unknown,
                    self.session.clone(),
                )
            };

            let encoding_id = self
                .layout_read_ctx
                .resolve(fb_child.encoding())
                .ok_or_else(|| {
                    vortex_err!("Unknown layout encoding index: {}", fb_child.encoding())
                })?;
            let Some(encoding) = self.layouts.get(&encoding_id) else {
                if self.allow_unknown {
                    return viewed_children.foreign_layout_from_fb(fb_child, dtype);
                }
                vortex_bail!("Unknown layout encoding: {encoding_id}");
            };

            let build_ctx = LayoutBuildContext {
                session: &self.session,
                array_read_ctx: &self.array_read_ctx,
            };
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
                &build_ctx,
            )
        })?;
        Ok(Arc::clone(layout_ref))
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
        self.cache.len()
    }

    fn child_is_indivisible(&self, idx: usize) -> bool {
        self.indivisible.get(idx).copied().unwrap_or(false)
    }
}
