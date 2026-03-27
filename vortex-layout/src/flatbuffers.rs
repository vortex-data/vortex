// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::FlatBufferBuilder;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::WIPOffset;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::layout;
use vortex_flatbuffers::root;
use vortex_session::registry::ReadContext;

use crate::Layout;
use crate::LayoutContext;
use crate::LayoutRef;
use crate::children::ViewedLayoutChildren;
use crate::segments::SegmentId;
use crate::session::LayoutRegistry;

pub(crate) fn layout_at_path<'a>(
    bytes: &'a [u8],
    path: &[usize],
) -> VortexResult<layout::LayoutRef<'a>> {
    let mut fb_layout = root::<layout::LayoutRef<'_>>(bytes)?;

    for &idx in path {
        let children = fb_layout
            .children()?
            .ok_or_else(|| vortex_err!("Layout node missing children at path {:?}", path))?;
        let Some(child) = children.iter().nth(idx) else {
            vortex_bail!(
                "Layout child index {} out of bounds for path {:?}",
                idx,
                path
            );
        };
        fb_layout = child?;
    }

    Ok(fb_layout)
}

pub(crate) fn build_layout_from_path(
    flatbuffer: FlatBuffer,
    path: &[usize],
    dtype: &DType,
    layout_ctx: &ReadContext,
    ctx: &ReadContext,
    layouts: &LayoutRegistry,
) -> VortexResult<LayoutRef> {
    let flatbuffer_ref = flatbuffer.clone();
    let fb_layout = layout_at_path(flatbuffer_ref.as_ref(), path)?;
    build_layout_from_ref(flatbuffer, path, fb_layout, dtype, layout_ctx, ctx, layouts)
}

fn build_layout_from_ref(
    flatbuffer: FlatBuffer,
    path: &[usize],
    fb_layout: layout::LayoutRef<'_>,
    dtype: &DType,
    layout_ctx: &ReadContext,
    ctx: &ReadContext,
    layouts: &LayoutRegistry,
) -> VortexResult<LayoutRef> {
    let encoding = fb_layout.encoding()?;
    let encoding_id = layout_ctx
        .resolve(encoding)
        .ok_or_else(|| vortex_err!("Invalid encoding ID: {}", encoding))?;
    let encoding = layouts
        .find(&encoding_id)
        .ok_or_else(|| vortex_err!("Invalid encoding ID: {}", encoding))?;

    let viewed_children = ViewedLayoutChildren::new(
        flatbuffer,
        path,
        fb_layout,
        ctx.clone(),
        layout_ctx.clone(),
        layouts.clone(),
    )?;

    let segments = fb_layout
        .segments()?
        .map(|segments| segments.iter().map(SegmentId::from).collect())
        .unwrap_or_default();

    encoding.build(
        dtype,
        fb_layout.row_count()?,
        fb_layout.metadata()?.unwrap_or(&[]),
        segments,
        &viewed_children,
        ctx,
    )
}

/// Parse a [`LayoutRef`] from a layout flatbuffer.
pub fn layout_from_flatbuffer(
    flatbuffer: FlatBuffer,
    dtype: &DType,
    layout_ctx: &ReadContext,
    ctx: &ReadContext,
    layouts: &LayoutRegistry,
) -> VortexResult<LayoutRef> {
    build_layout_from_path(flatbuffer, &[], dtype, layout_ctx, ctx, layouts)
}

impl dyn Layout + '_ {
    /// Serialize the layout into a [`FlatBufferBuilder`].
    pub fn flatbuffer_writer<'a>(
        &'a self,
        ctx: &'a LayoutContext,
    ) -> impl WriteFlatBuffer<Target = layout::Layout> + FlatBufferRoot + 'a {
        LayoutFlatBufferWriter { layout: self, ctx }
    }
}

/// An adapter struct for writing a layout to a FlatBuffer.
struct LayoutFlatBufferWriter<'a> {
    layout: &'a dyn Layout,
    ctx: &'a LayoutContext,
}

impl FlatBufferRoot for LayoutFlatBufferWriter<'_> {}

impl WriteFlatBuffer for LayoutFlatBufferWriter<'_> {
    type Target = layout::Layout;

    fn write_flatbuffer(
        &self,
        fbb: &mut FlatBufferBuilder,
    ) -> VortexResult<WIPOffset<Self::Target>> {
        let child_layouts = self.layout.children()?;
        let children = child_layouts
            .iter()
            .map(|layout| {
                LayoutFlatBufferWriter {
                    layout: layout.as_ref(),
                    ctx: self.ctx,
                }
                .write_flatbuffer(fbb)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let children = (!children.is_empty()).then(|| fbb.create_vector(&children));

        let metadata = self.layout.metadata();
        let metadata = (!metadata.is_empty()).then(|| fbb.create_vector(&metadata));

        let segments = self
            .layout
            .segment_ids()
            .into_iter()
            .map(|s| *s)
            .collect::<Vec<_>>();
        let segments = (!segments.is_empty()).then(|| fbb.create_vector(&segments));

        let encoding = self.ctx.intern(&self.layout.encoding_id()).ok_or_else(|| {
            vortex_err!(
                "Failed to intern layout encoding ID: {}",
                self.layout.encoding_id()
            )
        })?;

        Ok(layout::Layout::create(
            fbb,
            encoding,
            self.layout.row_count(),
            metadata,
            children,
            segments,
        ))
    }
}
