// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
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

/// Parse a [`LayoutRef`] from a layout flatbuffer.
pub fn layout_from_flatbuffer(
    flatbuffer: FlatBuffer,
    dtype: &DType,
    layout_ctx: &ReadContext,
    ctx: &ReadContext,
    layouts: &LayoutRegistry,
) -> VortexResult<LayoutRef> {
    let fb_layout: layout::Layout =
        root::<layout::LayoutRef<'_>>(flatbuffer.as_ref())?.try_into()?;
    let encoding_id = layout_ctx
        .resolve(fb_layout.encoding)
        .ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding))?;
    let encoding = layouts
        .find(&encoding_id)
        .ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding))?;

    let viewed_children = ViewedLayoutChildren::new(
        fb_layout.children.clone().unwrap_or_default().into(),
        ctx.clone(),
        layout_ctx.clone(),
        layouts.clone(),
    );

    let layout = encoding.build(
        dtype,
        fb_layout.row_count,
        fb_layout.metadata.as_deref().unwrap_or(&[]),
        fb_layout
            .segments
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .copied()
            .map(SegmentId::from)
            .collect(),
        &viewed_children,
        ctx,
    )?;

    Ok(layout)
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
