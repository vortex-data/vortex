// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use futures::FutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use serde::Serialize;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::array::buffer::BufferHandle;
use vortex::array::dtype::DType;
use vortex::array::serde::SerializedArray;
use vortex::array::session::ArraySessionExt;
use vortex::array::stream::ArrayStream;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VERSION;
use vortex::file::VortexFile;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::wasm::WasmRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutChildType;
use vortex::layout::LayoutRef;
use vortex::layout::layouts::flat::Flat;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::session::VortexSession;
use vortex::session::registry::ReadContext;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Initialize the WASM module (sets up panic hook for better error messages).
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// A `VortexReadAt` backed by a `web_sys::Blob`, enabling lazy range-based reads
/// via `Blob.slice()` + `arrayBuffer()`.
struct BlobReadAt {
    blob: web_sys::Blob,
    size: u64,
}

// SAFETY: WASM is single-threaded — Blob is never accessed from multiple threads.
unsafe impl Send for BlobReadAt {}
unsafe impl Sync for BlobReadAt {}

/// Wrapper to mark a `JsFuture` as `Send`.
///
/// SAFETY: WASM is single-threaded, so `JsFuture` is never accessed from multiple threads.
struct SendFuture(JsFuture);

unsafe impl Send for SendFuture {}

impl Future for SendFuture {
    type Output = Result<JsValue, JsValue>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We never move the inner JsFuture after pinning.
        unsafe { Pin::new_unchecked(&mut self.0).poll(cx) }
    }
}

impl VortexReadAt for BlobReadAt {
    fn concurrency(&self) -> usize {
        4
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::in_memory())
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let size = self.size;
        async move { Ok(size) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let start = offset as f64;
        let end = (offset + length as u64) as f64;
        let slice = self
            .blob
            .slice_with_f64_and_f64(start, end)
            .expect("Blob.slice() failed");
        // SAFETY: WASM is single-threaded so the non-Send JsFuture is safe to wrap.
        let future = SendFuture(JsFuture::from(slice.array_buffer()));
        async move {
            let array_buffer = future.await.expect("Blob.arrayBuffer() failed");
            let uint8 = js_sys::Uint8Array::new(&array_buffer);
            let mut buffer =
                ByteBufferMut::with_capacity_aligned(uint8.length() as usize, alignment);
            buffer.extend_from_slice(&uint8.to_vec());
            Ok(BufferHandle::new_host(buffer.freeze()))
        }
        .boxed()
    }
}

/// Open a Vortex file from a `File` handle and return a handle for exploration.
///
/// The `File` (a `Blob`) is read lazily — only the footer is read at open time.
#[wasm_bindgen]
pub async fn open_vortex_file(file: web_sys::File) -> Result<VortexFileHandle, JsValue> {
    let session = VortexSession::default().with_handle(WasmRuntime::handle());
    let blob: &web_sys::Blob = file.as_ref();
    let file_size = blob.size() as usize;
    let reader = Arc::new(BlobReadAt {
        blob: blob.clone(),
        size: file_size as u64,
    });

    let vxf = session
        .open_options()
        .open(reader)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Extract the array ReadContext from any FlatLayout in the tree.
    let array_read_ctx = find_array_read_ctx(vxf.footer().layout());

    Ok(VortexFileHandle {
        vxf,
        session,
        file_size,
        array_read_ctx,
    })
}

/// Walk the layout tree to find the first FlatLayout and extract its ReadContext.
fn find_array_read_ctx(layout: &LayoutRef) -> Option<ReadContext> {
    if let Some(flat) = layout.as_opt::<Flat>() {
        return Some(flat.array_ctx().clone());
    }
    if let Ok(children) = layout.children() {
        for child in children {
            if let Some(ctx) = find_array_read_ctx(&child) {
                return Some(ctx);
            }
        }
    }
    None
}

/// A handle to an opened Vortex file, exposing metadata to JavaScript.
#[wasm_bindgen]
pub struct VortexFileHandle {
    vxf: VortexFile,
    session: VortexSession,
    file_size: usize,
    array_read_ctx: Option<ReadContext>,
}

#[wasm_bindgen]
impl VortexFileHandle {
    /// The total number of rows in the file.
    #[wasm_bindgen(getter)]
    pub fn row_count(&self) -> u64 {
        self.vxf.row_count()
    }

    /// The top-level DType of the file as a string.
    #[wasm_bindgen(getter)]
    pub fn dtype(&self) -> String {
        format!("{}", self.vxf.dtype())
    }

    /// Returns the layout tree as a JSON string matching the TS `LayoutTreeNode` type.
    pub fn layout_tree(&self) -> Result<String, JsValue> {
        let root = self.vxf.footer().layout().clone();
        let tree = build_layout_tree(root, "root".to_string(), 0, &ChildKindJson::Root)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        serde_json::to_string(&tree)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {e}")))
    }

    /// Returns the segment map as a JSON string matching the TS `SegmentMapEntry[]` type.
    pub fn segment_map(&self) -> Result<String, JsValue> {
        let footer = self.vxf.footer();
        let root_layout = footer.layout().clone();
        let segment_map = footer.segment_map();

        // BFS to map each segment ID to its layout path and column name.
        let mut segment_paths: Vec<Option<(String, Option<String>)>> =
            vec![None; segment_map.len()];

        let mut queue: VecDeque<(String, Option<String>, LayoutRef)> =
            VecDeque::from([(String::from("root"), None, root_layout)]);

        while let Some((path, column, layout)) = queue.pop_front() {
            for segment in layout.segment_ids() {
                let idx = *segment as usize;
                if idx < segment_paths.len() {
                    segment_paths[idx] = Some((path.clone(), column.clone()));
                }
            }

            if let Ok(children) = layout.children() {
                for (i, child_layout) in children.into_iter().enumerate() {
                    let child_type = layout.child_type(i);
                    let child_name = child_type.name();
                    let child_path = format!("{path}.{child_name}");

                    // Track the column: use this field's name if it's a Field, otherwise
                    // inherit the parent's column.
                    let child_column = match &child_type {
                        LayoutChildType::Field(name) => Some(name.to_string()),
                        _ => column.clone(),
                    };

                    queue.push_back((child_path, child_column, child_layout));
                }
            }
        }

        let entries: Vec<SegmentMapEntryJson> = segment_map
            .iter()
            .enumerate()
            .map(|(i, spec)| {
                let (layout_path, column) = segment_paths[i]
                    .clone()
                    .unwrap_or_else(|| (String::from("<unknown>"), None));
                SegmentMapEntryJson {
                    index: i,
                    byte_offset: spec.offset,
                    byte_length: spec.length,
                    alignment: *spec.alignment,
                    column,
                    layout_path,
                }
            })
            .collect();

        serde_json::to_string(&entries)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {e}")))
    }

    /// Returns file structure info as a JSON string matching the TS `FileStructureInfo` type.
    pub fn file_structure(&self) -> Result<String, JsValue> {
        let footer = self.vxf.footer();
        let segment_map = footer.segment_map();

        let total_data_bytes: u64 = segment_map.iter().map(|s| s.length as u64).sum();

        let total_metadata_bytes =
            sum_metadata_bytes(footer.layout()).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let info = FileStructureJson {
            file_size: self.file_size as u64,
            version: VERSION,
            postscript_size: 64,
            total_data_bytes,
            total_metadata_bytes,
        };

        serde_json::to_string(&info)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {e}")))
    }

    /// Preview data from a specific layout node, returning Arrow IPC stream bytes.
    ///
    /// Navigates to the layout node identified by `node_id` (e.g. "root.customer_id.[0]"),
    /// creates a layout reader, scans up to `row_limit` rows, and returns Arrow IPC bytes.
    pub async fn preview_data(
        &self,
        node_id: &str,
        row_limit: u32,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let layout = find_layout_by_id(self.vxf.footer().layout(), node_id)
            .ok_or_else(|| JsValue::from_str(&format!("Layout node not found: {node_id}")))?;

        let segment_source = self.vxf.segment_source();
        let reader = layout
            .new_reader(node_id.into(), segment_source, &self.session)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let stream = ScanBuilder::new(self.session.clone(), reader)
            .with_limit(row_limit as u64)
            .into_array_stream()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let dtype = stream.dtype().clone();
        let chunks: Vec<ArrayRef> = stream
            .try_collect()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let schema =
            dtype_to_schema(&dtype, "value").map_err(|e| JsValue::from_str(&e.to_string()))?;
        let arrow_schema = Arc::new(schema);

        let mut buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, &arrow_schema)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            for chunk in chunks {
                let batch = array_to_record_batch(chunk, &dtype, &arrow_schema)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                writer
                    .write(&batch)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
            }

            writer
                .finish()
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(js_sys::Uint8Array::from(buf.as_slice()))
    }

    /// Fetch the array encoding tree for a flat layout node.
    ///
    /// Finds the layout by node ID, reads the segment, fully decodes the array
    /// to extract dtype, child names, and buffer names from the encoding vtables.
    pub async fn fetch_encoding_tree(&self, node_id: String) -> Result<String, JsValue> {
        let ctx = self
            .array_read_ctx
            .as_ref()
            .ok_or_else(|| JsValue::from_str("No array ReadContext available"))?;

        let layout = find_layout_by_id(self.vxf.footer().layout(), &node_id)
            .ok_or_else(|| JsValue::from_str(&format!("Layout node not found: {node_id}")))?;

        let flat = layout
            .as_opt::<Flat>()
            .ok_or_else(|| JsValue::from_str("Node is not a flat layout"))?;

        let segment_id = flat.segment_id();
        let dtype = layout.dtype().clone();
        let row_count = layout.row_count() as usize;

        let buf = self
            .vxf
            .segment_source()
            .request(segment_id)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let parts =
            SerializedArray::try_from(buf).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let array = parts
            .decode(&dtype, row_count, ctx, &self.session)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let tree = build_array_encoding_tree_from_array(&array, &self.session);
        serde_json::to_string(&tree)
            .map_err(|e| JsValue::from_str(&format!("JSON serialization failed: {e}")))
    }

    /// Fetch a buffer from a decoded array node.
    ///
    /// `layout_node_id` identifies the flat layout, `array_path` is a list of
    /// child names to navigate within the decoded array tree (e.g. `["values", "encoded"]`),
    /// and `buffer_index` selects which buffer of the target array node to return.
    pub async fn fetch_array_buffer(
        &self,
        layout_node_id: String,
        array_path: Vec<String>,
        buffer_index: usize,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let ctx = self
            .array_read_ctx
            .as_ref()
            .ok_or_else(|| JsValue::from_str("No array ReadContext available"))?;

        let layout =
            find_layout_by_id(self.vxf.footer().layout(), &layout_node_id).ok_or_else(|| {
                JsValue::from_str(&format!("Layout node not found: {layout_node_id}"))
            })?;

        let flat = layout
            .as_opt::<Flat>()
            .ok_or_else(|| JsValue::from_str("Node is not a flat layout"))?;

        let segment_id = flat.segment_id();
        let dtype = layout.dtype().clone();
        let row_count = layout.row_count() as usize;

        let buf = self
            .vxf
            .segment_source()
            .request(segment_id)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let parts =
            SerializedArray::try_from(buf).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let root_array = parts
            .decode(&dtype, row_count, ctx, &self.session)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Navigate the array tree by child names.
        let mut current = root_array;
        for child_name in &array_path {
            let named = current.named_children();
            let child = named
                .into_iter()
                .find(|(name, _)| name == child_name)
                .map(|(_, arr)| arr)
                .ok_or_else(|| {
                    JsValue::from_str(&format!("Array child not found: {child_name}"))
                })?;
            current = child;
        }

        let handles = current.buffer_handles();
        let handle = handles.get(buffer_index).ok_or_else(|| {
            JsValue::from_str(&format!(
                "Buffer index {buffer_index} out of range ({})",
                handles.len()
            ))
        })?;

        Ok(js_sys::Uint8Array::from(handle.as_host().as_slice()))
    }

    /// Preview data from a specific array node within a flat layout.
    ///
    /// Decodes the array from the flat layout's segment, navigates to the
    /// target array child by name path, and returns Arrow IPC bytes.
    pub async fn preview_array_data(
        &self,
        layout_node_id: String,
        array_path: Vec<String>,
        row_limit: u32,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let ctx = self
            .array_read_ctx
            .as_ref()
            .ok_or_else(|| JsValue::from_str("No array ReadContext available"))?;

        let layout =
            find_layout_by_id(self.vxf.footer().layout(), &layout_node_id).ok_or_else(|| {
                JsValue::from_str(&format!("Layout node not found: {layout_node_id}"))
            })?;

        let flat = layout
            .as_opt::<Flat>()
            .ok_or_else(|| JsValue::from_str("Node is not a flat layout"))?;

        let segment_id = flat.segment_id();
        let dtype = layout.dtype().clone();
        let row_count = layout.row_count() as usize;

        let buf = self
            .vxf
            .segment_source()
            .request(segment_id)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let parts =
            SerializedArray::try_from(buf).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let root_array = parts
            .decode(&dtype, row_count, ctx, &self.session)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Navigate to the target array node.
        let mut current = root_array;
        for child_name in &array_path {
            let named = current.named_children();
            let child = named
                .into_iter()
                .find(|(name, _)| name == child_name)
                .map(|(_, arr)| arr)
                .ok_or_else(|| {
                    JsValue::from_str(&format!("Array child not found: {child_name}"))
                })?;
            current = child;
        }

        // Slice to row_limit.
        let len = current.len().min(row_limit as usize);
        if len < current.len() {
            current = current
                .slice(0..len)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        // Convert to Arrow IPC.
        let array_dtype = current.dtype().clone();
        let schema = dtype_to_schema(&array_dtype, "value")
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let arrow_schema = Arc::new(schema);

        let batch = array_to_record_batch(current, &array_dtype, &arrow_schema)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mut ipc_buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut ipc_buf, &arrow_schema)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            writer
                .write(&batch)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            writer
                .finish()
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(js_sys::Uint8Array::from(ipc_buf.as_slice()))
    }
}

/// Recursively build the layout tree JSON structure.
fn build_layout_tree(
    layout: LayoutRef,
    id: String,
    parent_row_offset: u64,
    child_type: &ChildKindJson,
) -> VortexResult<LayoutTreeNodeJson> {
    let row_count = layout.row_count();
    let children_result = layout.children();

    let mut children_json = Vec::new();
    if let Ok(children) = children_result {
        for (i, child_layout) in children.iter().enumerate() {
            let ct = layout.child_type(i);
            let child_name = ct.name();
            let child_id = format!("{id}.{child_name}");

            let child_row_offset = match &ct {
                LayoutChildType::Chunk((_, rel)) => parent_row_offset + rel,
                LayoutChildType::Auxiliary(_) => 0,
                LayoutChildType::Transparent(_) | LayoutChildType::Field(_) => parent_row_offset,
            };

            let child_kind = match &ct {
                LayoutChildType::Transparent(name) => ChildKindJson::Transparent {
                    name: name.to_string(),
                },
                LayoutChildType::Auxiliary(name) => ChildKindJson::Auxiliary {
                    name: name.to_string(),
                },
                LayoutChildType::Chunk((idx, row_offset)) => ChildKindJson::Chunk {
                    chunk_index: *idx,
                    row_offset: parent_row_offset + row_offset,
                },
                LayoutChildType::Field(name) => ChildKindJson::Field {
                    field_name: name.to_string(),
                },
            };

            children_json.push(build_layout_tree(
                child_layout.clone(),
                child_id,
                child_row_offset,
                &child_kind,
            )?);
        }
    }

    // For flat layouts, extract the array encoding tree if available.
    let array_encoding_tree = layout.as_opt::<Flat>().and_then(|flat| {
        let tree_buf = flat.array_tree()?;
        let ctx = flat.array_ctx();
        let parts = SerializedArray::from_array_tree(tree_buf.as_ref().to_vec()).ok()?;
        Some(build_array_encoding_tree(&parts, ctx))
    });

    Ok(LayoutTreeNodeJson {
        id,
        encoding: layout.encoding().id().to_string(),
        dtype: layout.dtype().to_string(),
        row_count,
        row_offset: parent_row_offset,
        metadata_bytes: layout.metadata().len(),
        segment_ids: layout.segment_ids().iter().map(|s| **s).collect(),
        child_type: child_type.clone(),
        children: children_json,
        array_encoding_tree,
    })
}

/// DFS to sum metadata bytes across all layout nodes.
fn sum_metadata_bytes(layout: &LayoutRef) -> VortexResult<u64> {
    let mut total = 0u64;
    for node in layout.depth_first_traversal() {
        total += node?.metadata().len() as u64;
    }
    Ok(total)
}

/// Recursively build the array encoding tree from `ArrayParts` (used for inline trees
/// where we don't have a fully decoded array).
fn build_array_encoding_tree(parts: &SerializedArray, ctx: &ReadContext) -> ArrayEncodingNodeJson {
    let encoding = ctx
        .resolve(parts.encoding_id())
        .map(|id| id.to_string())
        .unwrap_or_else(|| format!("unknown({})", parts.encoding_id()));

    let nchildren = parts.nchildren();
    let children: Vec<ArrayEncodingNodeJson> = (0..nchildren)
        .map(|i| build_array_encoding_tree(&parts.child(i), ctx))
        .collect();

    ArrayEncodingNodeJson {
        encoding,
        dtype: String::new(),
        metadata_bytes: parts.metadata().len(),
        num_buffers: parts.nbuffers(),
        buffer_lengths: parts.buffer_lengths(),
        buffer_names: Vec::new(),
        children,
        child_names: (0..nchildren).map(|i| format!("child {i}")).collect(),
    }
}

/// Recursively build the array encoding tree from a fully decoded array,
/// extracting dtype, child names, and buffer names from the encoding vtables.
fn build_array_encoding_tree_from_array(
    array: &ArrayRef,
    session: &VortexSession,
) -> ArrayEncodingNodeJson {
    let encoding = array.encoding_id().to_string();
    let dtype = array.dtype().to_string();
    let buffer_names = array.buffer_names();
    let buffer_handles = array.buffer_handles();
    let buffer_lengths: Vec<usize> = buffer_handles.iter().map(|b| b.len()).collect();
    let metadata_bytes = session
        .array_serialize(array)
        .ok()
        .flatten()
        .map(|m| m.len())
        .unwrap_or(0);

    let named_children = array.named_children();
    let child_names: Vec<String> = named_children
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    let children: Vec<ArrayEncodingNodeJson> = named_children
        .iter()
        .map(|(_, child)| build_array_encoding_tree_from_array(child, session))
        .collect();

    ArrayEncodingNodeJson {
        encoding,
        dtype,
        metadata_bytes,
        num_buffers: buffer_lengths.len(),
        buffer_lengths,
        buffer_names,
        children,
        child_names,
    }
}

/// Navigate the layout tree to find a node by its dot-separated ID path.
///
/// IDs match the format: "root.field_name.chunked.[0]" where each segment
/// corresponds to a `LayoutChildType::name()`.
fn find_layout_by_id(root: &LayoutRef, node_id: &str) -> Option<LayoutRef> {
    let segments: Vec<&str> = node_id.split('.').collect();
    if segments.is_empty() || segments[0] != "root" {
        return None;
    }
    if segments.len() == 1 {
        return Some(root.clone());
    }

    let mut current = root.clone();
    for seg in &segments[1..] {
        let children = current.children().ok()?;
        let mut found = false;
        for (i, child) in children.into_iter().enumerate() {
            let name = current.child_type(i).name();
            if name.as_ref() == *seg {
                current = child;
                found = true;
                break;
            }
        }
        if !found {
            return None;
        }
    }
    Some(current)
}

/// Downgrade Arrow `*View` types to their non-view equivalents so the JS
/// `apache-arrow` library can decode them.
fn downgrade_arrow_type(dt: DataType) -> DataType {
    match dt {
        DataType::Utf8View => DataType::LargeUtf8,
        DataType::BinaryView => DataType::LargeBinary,
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|f| {
                    Arc::new(Field::new(
                        f.name(),
                        downgrade_arrow_type(f.data_type().clone()),
                        f.is_nullable(),
                    ))
                })
                .collect(),
        ),
        DataType::List(f) => DataType::List(Arc::new(Field::new(
            f.name(),
            downgrade_arrow_type(f.data_type().clone()),
            f.is_nullable(),
        ))),
        DataType::LargeList(f) => DataType::LargeList(Arc::new(Field::new(
            f.name(),
            downgrade_arrow_type(f.data_type().clone()),
            f.is_nullable(),
        ))),
        other => other,
    }
}

/// Create an Arrow Schema from a Vortex DType, with view types downgraded.
fn dtype_to_schema(dtype: &DType, default_name: &str) -> VortexResult<Schema> {
    let schema = match dtype {
        DType::Struct(..) => dtype.to_arrow_schema()?,
        other => {
            let arrow_dt = other.to_arrow_dtype()?;
            let nullable = other.is_nullable();
            Schema::new(vec![Field::new(default_name, arrow_dt, nullable)])
        }
    };
    // Downgrade view types in all fields.
    Ok(Schema::new(
        schema
            .fields()
            .iter()
            .map(|f| {
                Field::new(
                    f.name(),
                    downgrade_arrow_type(f.data_type().clone()),
                    f.is_nullable(),
                )
            })
            .collect::<Vec<_>>(),
    ))
}

/// Convert a Vortex ArrayRef into an Arrow RecordBatch using the given schema.
///
/// Always uses `execute_arrow` with explicit types to ensure view types are avoided.
fn array_to_record_batch(
    array: ArrayRef,
    dtype: &DType,
    schema: &Arc<Schema>,
) -> VortexResult<RecordBatch> {
    let data_type = match dtype {
        DType::Struct(..) => DataType::Struct(schema.fields().clone()),
        _ => schema.field(0).data_type().clone(),
    };
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    let arrow = array.execute_arrow(Some(&data_type), &mut ctx)?;
    match dtype {
        DType::Struct(..) => Ok(RecordBatch::from(arrow.as_struct().clone())),
        _ => Ok(RecordBatch::try_new(schema.clone(), vec![arrow])?),
    }
}

// --- JSON serialization types ---

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LayoutTreeNodeJson {
    id: String,
    encoding: String,
    dtype: String,
    row_count: u64,
    row_offset: u64,
    metadata_bytes: usize,
    segment_ids: Vec<u32>,
    child_type: ChildKindJson,
    children: Vec<LayoutTreeNodeJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    array_encoding_tree: Option<ArrayEncodingNodeJson>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ArrayEncodingNodeJson {
    encoding: String,
    dtype: String,
    metadata_bytes: usize,
    num_buffers: usize,
    buffer_lengths: Vec<usize>,
    buffer_names: Vec<String>,
    children: Vec<ArrayEncodingNodeJson>,
    child_names: Vec<String>,
}

#[derive(Serialize, Clone)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum ChildKindJson {
    #[serde(rename_all = "camelCase")]
    Root,
    #[serde(rename_all = "camelCase")]
    Field { field_name: String },
    #[serde(rename_all = "camelCase")]
    Chunk { chunk_index: usize, row_offset: u64 },
    #[serde(rename_all = "camelCase")]
    Transparent { name: String },
    #[serde(rename_all = "camelCase")]
    Auxiliary { name: String },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentMapEntryJson {
    index: usize,
    byte_offset: u64,
    byte_length: u32,
    alignment: usize,
    column: Option<String>,
    layout_path: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FileStructureJson {
    file_size: u64,
    version: u16,
    postscript_size: u64,
    total_data_bytes: u64,
    total_metadata_bytes: u64,
}
