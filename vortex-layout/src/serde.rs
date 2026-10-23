// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::sync::LazyLock;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::VerifierOptions;
use flatbuffers::WIPOffset;
use flatbuffers::root_with_opts;
use vortex_array::dtype::DType;
use vortex_array::flatbuffers::FlatBuffer;
use vortex_array::flatbuffers::FlatBufferRoot;
use vortex_array::flatbuffers::WriteFlatBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::DynLayout;
use crate::LayoutBuildContext;
use crate::LayoutContext;
use crate::LayoutRef;
use crate::children::ViewedLayoutChildren;
use crate::flatbuffers::layout;
use crate::layouts::foreign::new_foreign_layout;
use crate::segments::SegmentId;
use crate::session::LayoutSessionExt;

static LAYOUT_VERIFIER: LazyLock<VerifierOptions> = LazyLock::new(|| {
    VerifierOptions {
        // Overridden
        max_tables: env::var("VORTEX_MAX_LAYOUT_TABLES")
            .ok()
            .and_then(|lmt| lmt.parse::<usize>().ok())
            .unwrap_or(1000000),
        max_depth: env::var("VORTEX_MAX_LAYOUT_DEPTH")
            .ok()
            .and_then(|lmt| lmt.parse::<usize>().ok())
            .unwrap_or(64),
        // Defaults from flatbuffers
        max_apparent_size: 1 << 31,
        ignore_missing_null_terminator: false,
    }
});

/// Parse a [`LayoutRef`] from a layout flatbuffer.
pub fn layout_from_flatbuffer(
    flatbuffer: FlatBuffer,
    dtype: &DType,
    layout_ctx: &ReadContext,
    ctx: &ReadContext,
    session: &VortexSession,
) -> VortexResult<LayoutRef> {
    layout_from_flatbuffer_with_options(flatbuffer, dtype, layout_ctx, ctx, session, false)
}

/// Parse a [`LayoutRef`] from a layout flatbuffer with unknown-encoding behavior control.
pub fn layout_from_flatbuffer_with_options(
    flatbuffer: FlatBuffer,
    dtype: &DType,
    layout_ctx: &ReadContext,
    ctx: &ReadContext,
    session: &VortexSession,
    allow_unknown: bool,
) -> VortexResult<LayoutRef> {
    let layout_session = session.layouts();
    let layouts = layout_session.registry();
    let fb_layout = root_with_opts::<layout::Layout>(&LAYOUT_VERIFIER, &flatbuffer)?;
    let encoding_id = layout_ctx
        .resolve(fb_layout.encoding())
        .ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding()))?;
    let encoding = layouts.get(&encoding_id);

    if encoding.is_none() && allow_unknown {
        return foreign_layout_from_fb(fb_layout, dtype, layout_ctx);
    }
    let encoding =
        encoding.ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding()))?;

    // SAFETY: we validate the flatbuffer above in the `root` call, and extract a loc.
    let viewed_children = unsafe {
        ViewedLayoutChildren::new_unchecked(
            flatbuffer.clone(),
            fb_layout._tab.loc(),
            ctx.clone(),
            layout_ctx.clone(),
            layouts.clone(),
            allow_unknown,
            session.clone(),
        )
    };

    let build_ctx = LayoutBuildContext {
        session,
        array_read_ctx: ctx,
    };
    let layout = encoding.build(
        encoding_id,
        dtype,
        fb_layout.row_count(),
        fb_layout
            .metadata()
            .map(|m| m.bytes())
            .unwrap_or_else(|| &[]),
        fb_layout
            .segments()
            .unwrap_or_default()
            .iter()
            .map(SegmentId::from)
            .collect(),
        &viewed_children,
        &build_ctx,
    )?;

    Ok(layout)
}

fn foreign_layout_from_fb(
    fb_layout: layout::Layout<'_>,
    dtype: &DType,
    layout_ctx: &ReadContext,
) -> VortexResult<LayoutRef> {
    let encoding_id = layout_ctx
        .resolve(fb_layout.encoding())
        .ok_or_else(|| vortex_err!("Invalid encoding ID: {}", fb_layout.encoding()))?;

    let children = fb_layout
        .children()
        .unwrap_or_default()
        .iter()
        .map(|child| foreign_layout_from_fb(child, dtype, layout_ctx))
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
            .collect(),
        children,
    ))
}

impl dyn DynLayout + '_ {
    /// Serialize the layout into a [`FlatBufferBuilder`].
    ///
    /// Each layout is written by the plugin `session` registers for its in-memory ID, which chooses
    /// the wire ID and metadata.
    pub fn flatbuffer_writer<'a>(
        &'a self,
        ctx: &'a LayoutContext,
        session: &'a VortexSession,
    ) -> impl WriteFlatBuffer<Target<'a> = layout::Layout<'a>> + FlatBufferRoot + 'a {
        LayoutFlatBufferWriter {
            layout: self,
            ctx,
            session,
        }
    }
}

/// An adapter struct for writing a layout to a FlatBuffer.
struct LayoutFlatBufferWriter<'a> {
    layout: &'a dyn DynLayout,
    ctx: &'a LayoutContext,
    session: &'a VortexSession,
}

impl FlatBufferRoot for LayoutFlatBufferWriter<'_> {}

impl WriteFlatBuffer for LayoutFlatBufferWriter<'_> {
    type Target<'fb> = layout::Layout<'fb>;

    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>> {
        // First we recurse into the children and write them out
        let child_layouts = self.layout.children()?;
        let children = child_layouts
            .iter()
            .map(|layout| {
                LayoutFlatBufferWriter {
                    layout: layout.as_ref(),
                    ctx: self.ctx,
                    session: self.session,
                }
                .write_flatbuffer(fbb)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let children = (!children.is_empty()).then(|| fbb.create_vector(&children));

        let serialization = self.session.layout_serialize(self.layout)?;

        // Next we write out the metadata if it's non-empty.
        let metadata = &serialization.metadata;
        let metadata = (!metadata.is_empty()).then(|| fbb.create_vector(metadata));

        let segments = self
            .layout
            .segment_ids()
            .into_iter()
            .map(|s| *s)
            .collect::<Vec<_>>();
        let segments = (!segments.is_empty()).then(|| fbb.create_vector(&segments));

        // Dictionary-encode the layout ID
        let encoding = self
            .ctx
            .intern(&serialization.serialized_id)
            .ok_or_else(|| {
                vortex_err!(
                    "Layout encoding {} not permitted by ctx",
                    serialization.serialized_id
                )
            })?;

        Ok(layout::Layout::create(
            fbb,
            &layout::LayoutArgs {
                encoding,
                row_count: self.layout.row_count(),
                metadata,
                children,
                segments,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;
    use std::sync::LazyLock;

    use flatbuffers::FlatBufferBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::flatbuffers::WriteFlatBufferExt;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;
    use vortex_session::registry::ReadContext;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::layout_from_flatbuffer;
    use super::layout_from_flatbuffer_with_options;
    use crate::DynLayout;
    use crate::LayoutBuildContext;
    use crate::LayoutChildren;
    use crate::LayoutContext;
    use crate::LayoutEncodingId;
    use crate::LayoutEncodingRef;
    use crate::LayoutId;
    use crate::LayoutRef;
    use crate::LayoutSerialization;
    use crate::LayoutVTablePlugin;
    use crate::VTable;
    use crate::flatbuffers::layout as fbl;
    use crate::layouts::flat::Flat;
    use crate::layouts::flat::FlatLayout;
    use crate::segments::SegmentId;
    use crate::session::LayoutSession;
    use crate::session::LayoutSessionExt;

    fn flat_v0_id() -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.test.flat_v0");
        *ID
    }

    /// Reads flat layouts under both IDs and writes them under the older `vortex.test.flat_v0`,
    /// which it declares only when `declare_v0` is set.
    #[derive(Debug)]
    struct FlatV0 {
        declare_v0: bool,
    }

    impl LayoutVTablePlugin for FlatV0 {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn id(&self) -> LayoutId {
            VTable::id(&Flat)
        }

        fn serialized_ids(&self) -> Vec<LayoutId> {
            let mut ids = vec![VTable::id(&Flat)];
            if self.declare_v0 {
                ids.push(flat_v0_id());
            }
            ids
        }

        fn serialize(
            &self,
            layout: &dyn DynLayout,
            _session: &VortexSession,
        ) -> VortexResult<LayoutSerialization> {
            Ok(LayoutSerialization {
                serialized_id: flat_v0_id(),
                metadata: layout.metadata(),
            })
        }

        fn build(
            &self,
            _serialized_id: LayoutId,
            dtype: &DType,
            row_count: u64,
            metadata: &[u8],
            segment_ids: Vec<SegmentId>,
            children: &dyn LayoutChildren,
            build_ctx: &LayoutBuildContext<'_>,
        ) -> VortexResult<LayoutRef> {
            LayoutVTablePlugin::build(
                &Flat,
                VTable::id(&Flat),
                dtype,
                row_count,
                metadata,
                segment_ids,
                children,
                build_ctx,
            )
        }
    }

    /// Writes a flat layout through `layout_ctx` with `plugin` registered for flat layouts, then
    /// reads it back.
    fn round_trip(plugin: FlatV0, layout_ctx: &LayoutContext) -> VortexResult<LayoutRef> {
        let session = VortexSession::empty().with::<LayoutSession>();
        let plugin: Arc<dyn LayoutVTablePlugin> = Arc::new(plugin);
        session
            .layouts()
            .register(LayoutEncodingRef::new_arc(plugin));
        let layout = FlatLayout::new(
            10,
            DType::Primitive(PType::I32, Nullability::NonNullable),
            SegmentId::from(0),
            ReadContext::new([]),
        )
        .into_layout();
        let flatbuffer = layout
            .flatbuffer_writer(layout_ctx, &session)
            .write_flatbuffer_bytes()?;
        layout_from_flatbuffer(
            flatbuffer,
            layout.dtype(),
            &ReadContext::new(layout_ctx.to_ids()),
            &ReadContext::new([]),
            &session,
        )
    }

    #[test]
    fn plugin_chooses_the_wire_id() -> VortexResult<()> {
        let layout_ctx = LayoutContext::default();
        let layout = round_trip(FlatV0 { declare_v0: true }, &layout_ctx)?;
        assert_eq!(layout_ctx.to_ids(), [flat_v0_id()]);
        assert!(layout.is::<Flat>());
        assert_eq!(layout.row_count(), 10);
        Ok(())
    }

    #[test]
    fn layout_context_checks_the_wire_id() {
        let layout_ctx =
            LayoutContext::default().with_allowed_ids(HashSet::from_iter([VTable::id(&Flat)]));
        let Err(error) = round_trip(FlatV0 { declare_v0: true }, &layout_ctx) else {
            panic!("the context does not permit the wire ID");
        };
        assert!(
            error
                .to_string()
                .contains("vortex.test.flat_v0 not permitted by ctx"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn undeclared_wire_id_fails_the_write() {
        let Err(error) = round_trip(FlatV0 { declare_v0: false }, &LayoutContext::default()) else {
            panic!("the plugin does not declare its wire ID");
        };
        assert!(
            error.to_string().contains("undeclared serialized ID"),
            "unexpected error: {error}"
        );
    }

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<LayoutSession>());

    #[expect(clippy::disallowed_methods, reason = "test-only id")]
    #[test]
    fn unknown_layout_encoding_allow_unknown() {
        let mut fbb = FlatBufferBuilder::new();

        let child_metadata = fbb.create_vector(&[9u8]);
        let child = fbl::Layout::create(
            &mut fbb,
            &fbl::LayoutArgs {
                encoding: 1,
                row_count: 3,
                metadata: Some(child_metadata),
                children: None,
                segments: None,
            },
        );

        let children = fbb.create_vector(&[child]);
        let metadata = fbb.create_vector(&[1u8, 2, 3]);
        let segments = fbb.create_vector(&[7u32]);
        let root = fbl::Layout::create(
            &mut fbb,
            &fbl::LayoutArgs {
                encoding: 0,
                row_count: 10,
                metadata: Some(metadata),
                children: Some(children),
                segments: Some(segments),
            },
        );
        fbb.finish_minimal(root);
        let (buf, start) = fbb.collapse();
        let layout_buffer = vortex_array::flatbuffers::FlatBuffer::align_from(
            vortex_buffer::ByteBuffer::from(buf).slice(start..),
        );

        let layout_ctx = ReadContext::new([
            LayoutEncodingId::new("vortex.test.foreign_layout"),
            LayoutEncodingId::new("vortex.test.foreign_child_layout"),
        ]);
        let array_ctx = ReadContext::new([]);

        let layout = layout_from_flatbuffer_with_options(
            layout_buffer,
            &DType::Variant(Nullability::Nullable),
            &layout_ctx,
            &array_ctx,
            &SESSION,
            true,
        )
        .unwrap();

        assert_eq!(layout.encoding_id().as_ref(), "vortex.test.foreign_layout");
        assert_eq!(layout.row_count(), 10);
        assert_eq!(layout.metadata(), vec![1, 2, 3]);
        assert_eq!(layout.segment_ids().len(), 1);
        assert_eq!(*layout.segment_ids()[0], 7);
        assert_eq!(layout.nchildren(), 1);

        let child = layout.slot(0).unwrap().unwrap();
        assert_eq!(
            child.encoding_id().as_ref(),
            "vortex.test.foreign_child_layout"
        );
        assert_eq!(child.metadata(), vec![9]);
    }
}
