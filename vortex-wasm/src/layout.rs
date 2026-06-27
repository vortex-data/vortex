// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`WasmLayout`] — a layout node whose arrays are decoded by an embedded WebAssembly kernel.
//!
//! A `WasmLayout` holds:
//! - **child layouts** providing the decoded inputs the kernel consumes (served to the guest via
//!   the `vx_decode_child` host import). Each child carries the layout's own output dtype — the
//!   same dtype a native VTable would read the bytes with — so it is recovered from the layout
//!   dtype on deserialization and never stored in the metadata.
//! - a **kernel segment** containing the embedded `.wasm` blob (written at end-of-file);
//! - an optional **payload segment** containing the encoding's own header bytes that the guest
//!   parses.
//!
//! See `docs/design/wasm-encodings.md`.

use std::sync::Arc;

use vortex_array::DeserializeMetadata;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_layout::LayoutBuildContext;
use vortex_layout::LayoutChildType;
use vortex_layout::LayoutChildren;
use vortex_layout::LayoutEncodingRef;
use vortex_layout::LayoutId;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutRef;
use vortex_layout::VTable;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::vtable;
use vortex_session::VortexSession;

use crate::abi::ABI_VERSION;
use crate::reader::WasmReader;

vtable!(Wasm);

/// Layout encoding id for [`WasmLayout`].
pub const WASM_LAYOUT_ID: &str = "vortex.wasm";

impl VTable for Wasm {
    type Layout = WasmLayout;
    type Encoding = WasmLayoutEncoding;
    type Metadata = ProstMetadata<WasmLayoutMetadata>;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new(WASM_LAYOUT_ID)
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(WasmLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        layout.row_count
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        ProstMetadata(WasmLayoutMetadata {
            encoding_id: layout.encoding_id.clone(),
            abi_version: layout.abi_version,
            has_payload: layout.payload_segment.is_some(),
        })
    }

    fn segment_ids(layout: &Self::Layout) -> Vec<SegmentId> {
        match layout.payload_segment {
            Some(payload) => vec![layout.kernel_segment, payload],
            None => vec![layout.kernel_segment],
        }
    }

    fn nchildren(layout: &Self::Layout) -> usize {
        layout.children.len()
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        layout
            .children
            .get(idx)
            .cloned()
            .ok_or_else(|| vortex_err!("WasmLayout child index {idx} out of bounds"))
    }

    fn child_type(_layout: &Self::Layout, idx: usize) -> LayoutChildType {
        LayoutChildType::Auxiliary(format!("input[{idx}]").into())
    }

    fn new_reader(
        layout: &Self::Layout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Ok(Arc::new(WasmReader::try_new(
            layout.clone(),
            name,
            segment_source,
            session.clone(),
            ctx.clone(),
        )?))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        row_count: u64,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _build_ctx: &LayoutBuildContext<'_>,
    ) -> VortexResult<Self::Layout> {
        if metadata.abi_version != ABI_VERSION {
            vortex_bail!(
                "WasmLayout abi version {} is not supported by this reader (expected {})",
                metadata.abi_version,
                ABI_VERSION
            );
        }
        let (kernel_segment, payload_segment) = match (metadata.has_payload, segment_ids.as_slice())
        {
            (false, [kernel]) => (*kernel, None),
            (true, [kernel, payload]) => (*kernel, Some(*payload)),
            _ => vortex_bail!(
                "WasmLayout expects {} segments, got {}",
                if metadata.has_payload { 2 } else { 1 },
                segment_ids.len()
            ),
        };

        // Each child carries the layout's own output dtype (the dtype a native VTable would read
        // the same bytes with), so it is recovered from `dtype` rather than stored in the metadata.
        let n = children.nchildren();
        let mut child_layouts = Vec::with_capacity(n);
        for idx in 0..n {
            child_layouts.push(children.child(idx, dtype)?);
        }

        Ok(WasmLayout {
            dtype: dtype.clone(),
            row_count,
            encoding_id: metadata.encoding_id.clone(),
            abi_version: metadata.abi_version,
            kernel_segment,
            payload_segment,
            children: child_layouts.into(),
        })
    }
}

/// Encoding marker for [`WasmLayout`].
#[derive(Debug)]
pub struct WasmLayoutEncoding;

/// A layout whose arrays are decoded by an embedded WebAssembly kernel.
#[derive(Clone, Debug)]
pub struct WasmLayout {
    dtype: DType,
    row_count: u64,
    encoding_id: String,
    abi_version: u32,
    /// Segment holding the embedded `.wasm` kernel blob.
    kernel_segment: SegmentId,
    /// Optional segment holding the encoding's own payload (parsed by the guest).
    payload_segment: Option<SegmentId>,
    /// Child layouts providing decoded inputs to the kernel, each with its own dtype.
    children: Arc<[LayoutRef]>,
}

impl WasmLayout {
    /// Construct a `WasmLayout` directly (used by the writer and tests).
    pub fn new(
        dtype: DType,
        row_count: u64,
        encoding_id: impl Into<String>,
        kernel_segment: SegmentId,
        payload_segment: Option<SegmentId>,
        children: Vec<LayoutRef>,
    ) -> Self {
        Self {
            dtype,
            row_count,
            encoding_id: encoding_id.into(),
            abi_version: ABI_VERSION,
            kernel_segment,
            payload_segment,
            children: children.into(),
        }
    }

    /// The guest encoding id recorded in the layout metadata.
    pub fn encoding_id(&self) -> &str {
        &self.encoding_id
    }

    /// The segment id of the embedded kernel.
    pub fn kernel_segment(&self) -> SegmentId {
        self.kernel_segment
    }

    /// The optional payload segment id.
    pub fn payload_segment(&self) -> Option<SegmentId> {
        self.payload_segment
    }

    pub(crate) fn dtype_ref(&self) -> &DType {
        &self.dtype
    }

    pub(crate) fn row_count_val(&self) -> u64 {
        self.row_count
    }

    pub(crate) fn child_layouts(&self) -> &[LayoutRef] {
        &self.children
    }
}

/// An in-memory [`LayoutChildren`] of same-dtype children, used to wrap per-chunk `WasmLayout`s in
/// a `ChunkedLayout` (`vortex-layout`'s own `OwnedLayoutChildren` is crate-private).
#[derive(Clone)]
struct SameDTypeChildren(Arc<[LayoutRef]>);

impl LayoutChildren for SameDTypeChildren {
    fn to_arc(&self) -> Arc<dyn LayoutChildren> {
        Arc::new(self.clone())
    }

    fn child(&self, idx: usize, dtype: &DType) -> VortexResult<LayoutRef> {
        let child = self
            .0
            .get(idx)
            .ok_or_else(|| vortex_err!("child index {idx} out of bounds"))?;
        vortex_ensure!(
            child.dtype() == dtype,
            "child {idx} dtype mismatch: {} != {}",
            child.dtype(),
            dtype
        );
        Ok(Arc::clone(child))
    }

    fn child_row_count(&self, idx: usize) -> u64 {
        self.0[idx].row_count()
    }

    fn nchildren(&self) -> usize {
        self.0.len()
    }
}

/// Build an [`Arc<dyn LayoutChildren>`] from same-dtype child layouts (e.g. chunks).
pub fn same_dtype_children(children: Vec<LayoutRef>) -> Arc<dyn LayoutChildren> {
    Arc::new(SameDTypeChildren(children.into()))
}

/// Serialized metadata for [`WasmLayout`].
///
/// Deliberately minimal: the child layouts carry their own encoding ids and the layout dtype is
/// recorded in the file, so neither child encoding ids nor child dtypes are duplicated here.
#[derive(Clone, PartialEq, prost::Message)]
pub struct WasmLayoutMetadata {
    /// Guest encoding id (e.g. `"acme.delta"`).
    #[prost(string, tag = "1")]
    pub encoding_id: String,
    /// Host/guest ABI version the kernel targets.
    #[prost(uint32, tag = "2")]
    pub abi_version: u32,
    /// Whether a payload segment follows the kernel segment.
    #[prost(bool, tag = "3")]
    pub has_payload: bool,
}
