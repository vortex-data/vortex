// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::future::BoxFuture;
use serde::Serialize;
use vortex::VortexSessionDefault;
use vortex::array::buffer::BufferHandle;
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
use vortex::session::VortexSession;
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

    Ok(VortexFileHandle { vxf, file_size })
}

/// A handle to an opened Vortex file, exposing metadata to JavaScript.
#[wasm_bindgen]
pub struct VortexFileHandle {
    vxf: VortexFile,
    file_size: usize,
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
