// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::mem::size_of;
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
use crate::LayoutDeserialization;
use crate::LayoutRef;
use crate::VTable;
use crate::children::ViewedLayoutChildren;
use crate::children::approx_viewed_children_bytes;
use crate::flatbuffers::layout;
use crate::layouts::chunked::Chunked;
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

/// Projects the retained framework storage for a layout tree from its wire structure.
///
/// The serialized layout bytes are retained for lazy children, while parsed nodes may allocate
/// their own metadata and child state. This deliberately estimates common built-in storage rather
/// than introspecting plugin-owned allocations. Metadata payload lengths provide a copy allowance
/// for built-in state. Allocator bookkeeping, Arc control blocks, and hidden plugin heap are
/// omitted, and sharing between cached footers may be overcounted. This standalone estimate
/// verifies the FlatBuffer independently, so normal layout deserialization verifies it again.
pub fn approx_layout_cache_bytes(flatbuffer: &FlatBuffer, layout_ctx: &ReadContext) -> usize {
    let Ok(root) = root_with_opts::<layout::Layout>(&LAYOUT_VERIFIER, flatbuffer.as_ref()) else {
        return flatbuffer.len();
    };

    let chunked_id = Chunked.id();
    let mut pending = vec![root];
    let mut nodes = 0usize;
    let mut segments = 0usize;
    let mut metadata_bytes = 0usize;
    let mut chunk_offsets = 0usize;
    let mut child_storage_bytes = 0usize;

    while let Some(node) = pending.pop() {
        nodes = nodes.saturating_add(1);
        let node_children = node.children();
        let child_count = node_children.as_ref().map_or(0, |children| children.len());
        child_storage_bytes =
            child_storage_bytes.saturating_add(approx_viewed_children_bytes(child_count));
        segments = segments.saturating_add(node.segments().map_or(0, |segments| segments.len()));
        metadata_bytes =
            metadata_bytes.saturating_add(node.metadata().map_or(0, |metadata| metadata.len()));
        if layout_ctx
            .resolve(node.encoding())
            .is_some_and(|id| id == chunked_id)
        {
            chunk_offsets = chunk_offsets.saturating_add(child_count.saturating_add(1));
        }
        if let Some(node_children) = node_children {
            pending.extend(node_children.iter());
        }
    }

    flatbuffer
        .len()
        .saturating_add(nodes.saturating_mul(crate::layout::max_core_layout_allocation_size()))
        .saturating_add(child_storage_bytes)
        .saturating_add(segments.saturating_mul(size_of::<SegmentId>()))
        .saturating_add(metadata_bytes)
        .saturating_add(chunk_offsets.saturating_mul(size_of::<u64>()))
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
    let layout = encoding.deserialize(
        LayoutDeserialization {
            serialized_id: encoding_id,
            dtype,
            row_count: fb_layout.row_count(),
            metadata: fb_layout
                .metadata()
                .map(|m| m.bytes())
                .unwrap_or_else(|| &[]),
            segment_ids: fb_layout
                .segments()
                .unwrap_or_default()
                .iter()
                .map(SegmentId::from)
                .collect(),
            children: &viewed_children,
        },
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
    use crate::LayoutContext;
    use crate::LayoutDeserialization;
    use crate::LayoutId;
    use crate::LayoutPlugin;
    use crate::LayoutPluginRef;
    use crate::LayoutRef;
    use crate::LayoutSerialization;
    use crate::VTable;
    use crate::flatbuffers::layout as fbl;
    use crate::layouts::chunked::Chunked;
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

    impl LayoutPlugin for FlatV0 {
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

        fn deserialize(
            &self,
            parts: LayoutDeserialization<'_>,
            ctx: &LayoutBuildContext<'_>,
        ) -> VortexResult<LayoutRef> {
            // Both IDs share the flat wire format.
            LayoutPlugin::deserialize(
                &Flat,
                LayoutDeserialization {
                    serialized_id: VTable::id(&Flat),
                    ..parts
                },
                ctx,
            )
        }
    }

    /// Writes a flat layout through `layout_ctx` with `plugin` registered for flat layouts, then
    /// reads it back.
    fn round_trip(plugin: FlatV0, layout_ctx: &LayoutContext) -> VortexResult<LayoutRef> {
        let session = VortexSession::empty().with::<LayoutSession>();
        let plugin: Arc<dyn LayoutPlugin> = Arc::new(plugin);
        session.layouts().register(LayoutPluginRef::new_arc(plugin));
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

    #[test]
    fn test_cache_projection_counts_layout_metadata_and_chunk_offsets() -> VortexResult<()> {
        const METADATA_BYTES: usize = 4096;

        let mut fbb = FlatBufferBuilder::new();
        let metadata = vec![0u8; METADATA_BYTES];
        let child_metadata = fbb.create_vector(&metadata);
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
        let metadata = fbb.create_vector(&metadata);
        let segments = fbb.create_vector(&[7u32, 8]);
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
        let flatbuffer = vortex_array::flatbuffers::FlatBuffer::align_from(
            vortex_buffer::ByteBuffer::from(buf).slice(start..),
        );
        let layout_context = ReadContext::new([VTable::id(&Chunked)]);
        let minimum = flatbuffer
            .len()
            .saturating_add(2 * METADATA_BYTES)
            .saturating_add(2 * size_of::<u32>())
            .saturating_add(2 * size_of::<u64>());
        let without_chunk_offsets =
            super::approx_layout_cache_bytes(&flatbuffer, &ReadContext::new([]));
        let with_chunk_offsets = super::approx_layout_cache_bytes(&flatbuffer, &layout_context);

        assert!(with_chunk_offsets >= minimum);
        assert_eq!(
            with_chunk_offsets,
            without_chunk_offsets + 2 * size_of::<u64>()
        );
        Ok(())
    }

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
            LayoutId::new("vortex.test.foreign_layout"),
            LayoutId::new("vortex.test.foreign_child_layout"),
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
