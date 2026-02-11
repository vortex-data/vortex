// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex file scanning for ClickHouse.
//!
//! This module implements the read path for Vortex files in ClickHouse.
//! It provides the core logic for the `VortexBlockInputFormat` C++ class.
//!
//! # FFI Interface
//!
//! The following C functions are exported for ClickHouse to use:
//!
//! - `vortex_scanner_new` - Create a new scanner
//! - `vortex_scanner_free` - Free a scanner
//! - `vortex_scanner_num_columns` - Get number of columns
//! - `vortex_scanner_column_name` - Get column name by index
//! - `vortex_scanner_column_type` - Get ClickHouse type string for column
//! - `vortex_scanner_set_projection` - Set columns to read
//! - `vortex_scanner_read_batch` - Read a batch of data
//!
//! # Thread Safety
//!
//! A [`VortexScanner`] instance is **not** thread-safe. The FFI functions
//! dereference raw pointers into shared or exclusive references, so concurrent
//! calls on the same scanner handle from multiple threads cause undefined
//! behavior. The caller on the C++ side must serialize all access to a given
//! scanner handle, or create a separate scanner per thread.
//!
//! The error reporting functions (`vortex_get_last_error`, `vortex_has_error`,
//! `vortex_clear_error`) use thread-local storage and are safe to call from any
//! thread.
//!
//! # Remote File Support
//!
//! This module supports reading Vortex files from remote storage systems:
//! - S3 (`s3://bucket/path/to/file.vortex`)
//! - Google Cloud Storage (`gs://bucket/path/to/file.vortex`)
//! - Azure Blob Storage (`az://container/path/to/file.vortex`)
//! - HTTP/HTTPS (`https://example.com/path/to/file.vortex`)

use std::collections::VecDeque;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;
use std::ptr;

use futures::TryStreamExt;
use parking_lot::Mutex;
use vortex::array::ArrayRef;
use vortex::dtype::DType;
use vortex::error::{VortexResult, vortex_bail, vortex_err};
use vortex::expr::{root, select};
use vortex::file::{OpenOptionsSessionExt, VortexFile};
use vortex::io::runtime::BlockingRuntime;

use crate::convert::dtype::vortex_to_clickhouse_type;
use crate::error::{clear_last_error, set_last_error};
use crate::exporter::{ColumnExporter, ExporterKind, new_exporter};
use crate::utils::object_store::{is_remote_path, make_object_store};
use crate::{RUNTIME, SESSION};

/// Vortex file scanner that implements the read logic for ClickHouse.
///
/// This struct holds the state needed to read data from one or more Vortex files.
/// It manages file handles, schema information, projection settings, and
/// provides an iterator-like interface for reading data batches.
///
/// # Thread Safety
///
/// This type is **not** thread-safe. Although `cached_total_rows` is wrapped in
/// a [`Mutex`] (to allow interior mutability behind `&self`), the remaining
/// fields are unprotected mutable state. Concurrent access from multiple threads
/// — including through the FFI functions that dereference raw pointers — is
/// undefined behavior. The caller must serialize all access to a given instance.
pub struct VortexScanner {
    /// Path or glob pattern for files to scan.
    file_paths: Vec<String>,
    /// Index of the current file being scanned.
    current_file_idx: usize,
    /// The currently open file, if any.
    current_file: Option<VortexFile>,
    /// Column indices to project (None = all columns).
    projection: Option<Vec<usize>>,
    /// Column names for projection.
    projection_names: Option<Vec<String>>,
    /// The schema of the Vortex file.
    schema: DType,
    /// Cached column names from schema.
    column_names: Vec<String>,
    /// Cached ClickHouse type strings.
    column_types: Vec<String>,
    /// Current row offset within the file.
    current_row_offset: u64,
    /// Total rows read so far.
    total_rows_read: u64,
    /// Batch size for reading (rows per batch).
    batch_size: usize,
    /// Current batch being exported, if any.
    current_batch: Option<Box<dyn ColumnExporter>>,
    /// Pending chunks from the last file read (avoids merging all chunks).
    pending_chunks: VecDeque<ArrayRef>,
    /// Whether we've finished reading all files.
    finished: bool,
    /// Cached total row count across all files.
    cached_total_rows: Mutex<Option<u64>>,
}

impl VortexScanner {
    /// Create a new scanner for the given file path or glob pattern.
    ///
    /// The path can be:
    /// - A local file path: `/path/to/file.vortex`
    /// - A local glob pattern: `/path/to/*.vortex`
    /// - A remote URL: `s3://bucket/path/to/file.vortex`
    /// - A remote URL with glob pattern: `s3://bucket/path/to/*.vortex`
    pub fn new(path: &str) -> VortexResult<Self> {
        if path.is_empty() {
            vortex_bail!("Path cannot be empty");
        }

        let file_paths = expand_glob(path)?;

        // Read schema from first file
        let schema = read_schema_from_file(&file_paths[0])?;

        // Extract column names and types from schema
        let (column_names, column_types) = extract_column_info(&schema)?;

        Ok(Self {
            file_paths,
            current_file_idx: 0,
            current_file: None,
            projection: None,
            projection_names: None,
            schema,
            column_names,
            column_types,
            current_row_offset: 0,
            total_rows_read: 0,
            batch_size: 65536, // Default batch size
            current_batch: None,
            pending_chunks: VecDeque::new(),
            finished: false,
            cached_total_rows: Mutex::new(None),
        })
    }

    /// Set the columns to project by name.
    pub fn set_projection(&mut self, columns: Vec<String>) -> VortexResult<()> {
        // Map column names to indices
        let indices: VortexResult<Vec<usize>> = columns
            .iter()
            .map(|name| {
                self.column_names
                    .iter()
                    .position(|n| n == name)
                    .ok_or_else(|| vortex_err!("Column not found: {}", name))
            })
            .collect();

        self.projection = Some(indices?);
        self.projection_names = Some(columns);
        Ok(())
    }

    /// Set the columns to project by index.
    pub fn set_projection_indices(&mut self, indices: Vec<usize>) -> VortexResult<()> {
        // Validate indices
        for &idx in &indices {
            if idx >= self.column_names.len() {
                vortex_bail!(
                    "Column index {} out of bounds (max: {})",
                    idx,
                    self.column_names.len() - 1
                );
            }
        }

        let names: Vec<String> = indices
            .iter()
            .map(|&idx| self.column_names[idx].clone())
            .collect();

        self.projection_names = Some(names);
        self.projection = Some(indices);
        Ok(())
    }

    /// Set the batch size for reading.
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size.max(1);
    }

    /// Get the schema of the Vortex file.
    pub fn schema(&self) -> &DType {
        &self.schema
    }

    /// Get the list of file paths to scan.
    pub fn file_paths(&self) -> &[String] {
        &self.file_paths
    }

    /// Get the number of columns in the schema.
    pub fn num_columns(&self) -> usize {
        self.column_names.len()
    }

    /// Get a column name by index.
    pub fn column_name(&self, index: usize) -> Option<&str> {
        self.column_names.get(index).map(|s| s.as_str())
    }

    /// Get a column's ClickHouse type string by index.
    pub fn column_type(&self, index: usize) -> Option<&str> {
        self.column_types.get(index).map(|s| s.as_str())
    }

    /// Check if there are more batches to read.
    pub fn has_more(&self) -> bool {
        !self.finished
    }

    /// Read the next batch of data.
    ///
    /// Returns an exporter for the batch, or None if no more data.
    pub fn read_next_batch(&mut self) -> VortexResult<Option<Box<dyn ColumnExporter>>> {
        if self.finished {
            return Ok(None);
        }

        // Check if we have a current batch with remaining data
        if let Some(ref batch) = self.current_batch {
            if batch.has_more() {
                return Ok(self.current_batch.take());
            }
        }

        // Check if we have pending chunks from a previous read
        if let Some(chunk) = self.pending_chunks.pop_front() {
            self.total_rows_read += chunk.len() as u64;
            let exporter = new_exporter(chunk)?;
            return Ok(Some(exporter));
        }

        // Try to read from current or next file
        loop {
            // Open next file if needed
            if self.current_file.is_none() {
                if self.current_file_idx >= self.file_paths.len() {
                    self.finished = true;
                    return Ok(None);
                }

                let path = &self.file_paths[self.current_file_idx];
                self.current_file = Some(open_vortex_file(path)?);
                self.current_row_offset = 0;
            }

            let file = self.current_file.as_ref().unwrap();
            let row_count = file.row_count();

            // Check if we've read all rows in this file
            if self.current_row_offset >= row_count {
                self.current_file = None;
                self.current_file_idx += 1;
                continue;
            }

            // Calculate the range to read
            let start = self.current_row_offset;
            let end = (start + self.batch_size as u64).min(row_count);

            // Read the batch — returns individual chunks without merging
            let chunks = read_batch_from_file(file, start..end, self.projection_names.as_ref())?;

            self.current_row_offset = end;

            // Return the first chunk, queue the rest
            if chunks.is_empty() {
                continue;
            }

            let mut chunks = VecDeque::from(chunks);
            let first = chunks.pop_front().unwrap();
            self.total_rows_read += first.len() as u64;
            self.pending_chunks = chunks;

            let exporter = new_exporter(first)?;
            return Ok(Some(exporter));
        }
    }

    /// Get the total number of rows read so far.
    pub fn total_rows_read(&self) -> u64 {
        self.total_rows_read
    }

    /// Get the total row count across all files.
    /// This reads metadata from all files to compute the total.
    /// The result is cached after the first call.
    pub fn total_row_count(&self) -> VortexResult<u64> {
        let mut cached = self.cached_total_rows.lock();
        if let Some(total) = *cached {
            return Ok(total);
        }
        let mut total: u64 = 0;
        for path in &self.file_paths {
            let file = open_vortex_file(path)?;
            total += file.row_count();
        }
        *cached = Some(total);
        Ok(total)
    }

    /// Get the number of files to scan.
    pub fn num_files(&self) -> usize {
        self.file_paths.len()
    }

    /// Get the current file index.
    pub fn current_file_index(&self) -> usize {
        self.current_file_idx
    }
}

/// Extract column names and ClickHouse type strings from schema.
fn extract_column_info(schema: &DType) -> VortexResult<(Vec<String>, Vec<String>)> {
    match schema {
        DType::Struct(fields, _) => {
            let mut names = Vec::with_capacity(fields.nfields());
            let mut types = Vec::with_capacity(fields.nfields());

            for i in 0..fields.nfields() {
                let name = fields
                    .field_name(i)
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| format!("_{}", i));
                names.push(name);

                let dtype = fields
                    .field_by_index(i)
                    .ok_or_else(|| vortex_err!("Failed to get field dtype at index {}", i))?;
                let ch_type = vortex_to_clickhouse_type(&dtype)?;
                types.push(ch_type);
            }

            Ok((names, types))
        }
        _ => {
            // For non-struct types, treat as a single column
            let ch_type = vortex_to_clickhouse_type(schema)?;
            Ok((vec!["value".to_string()], vec![ch_type]))
        }
    }
}

/// Open a Vortex file (local or remote).
fn open_vortex_file(path: &str) -> VortexResult<VortexFile> {
    if is_remote_path(path) {
        open_remote_vortex_file_with_retry(path)
    } else {
        open_local_vortex_file(path)
    }
}

/// Open a local Vortex file.
fn open_local_vortex_file(path: &str) -> VortexResult<VortexFile> {
    (*RUNTIME).block_on(async { SESSION.open_options().open_path(path).await })
}

/// Retry configuration for remote operations.
const MAX_RETRIES: u32 = 3;
const INITIAL_BACKOFF_MS: u64 = 100;
const MAX_BACKOFF_MS: u64 = 5000;

/// Open a remote Vortex file with retry logic.
fn open_remote_vortex_file_with_retry(path: &str) -> VortexResult<VortexFile> {
    (*RUNTIME).block_on(async {
        let mut backoff_ms = INITIAL_BACKOFF_MS;

        for attempt in 0..MAX_RETRIES {
            match open_remote_vortex_file_async(path).await {
                Ok(file) => return Ok(file),
                Err(e) => {
                    if !is_retryable_error(&e) || attempt == MAX_RETRIES - 1 {
                        return Err(e);
                    }

                    tracing::warn!(
                        "Retrying remote file open for '{}' (attempt {}/{}): {}",
                        path,
                        attempt + 1,
                        MAX_RETRIES,
                        e
                    );

                    // Exponential backoff with simple jitter
                    let jitter_factor = (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .subsec_nanos() as u64)
                        % (backoff_ms / 5 + 1);
                    let sleep_ms = backoff_ms + jitter_factor;
                    smol::Timer::after(std::time::Duration::from_millis(sleep_ms)).await;

                    backoff_ms = (backoff_ms * 2).min(MAX_BACKOFF_MS);
                }
            }
        }

        vortex_bail!("Max retries exceeded for '{}'", path)
    })
}

/// Check if an error is retryable by walking the error source chain.
fn is_retryable_error(e: &vortex::error::VortexError) -> bool {
    let mut source: Option<&dyn std::error::Error> = Some(e);
    while let Some(err) = source {
        // Check for std::io errors
        if let Some(io_err) = err.downcast_ref::<std::io::Error>() {
            return matches!(
                io_err.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::ConnectionRefused
            );
        }
        // Fallback: check the error message for common retryable patterns
        let msg = err.to_string();
        if msg.contains("503")
            || msg.contains("429")
            || msg.contains("timeout")
            || msg.contains("Timeout")
            || msg.contains("connection reset")
            || msg.contains("temporarily unavailable")
        {
            return true;
        }
        source = err.source();
    }
    false
}

/// Async helper for opening remote files.
async fn open_remote_vortex_file_async(path: &str) -> VortexResult<VortexFile> {
    let store_info = make_object_store(path)?;
    SESSION
        .open_options()
        .open_object_store(&store_info.store, store_info.path.as_ref())
        .await
}

/// Read a batch of rows from a Vortex file.
///
/// Returns individual chunks without merging, to avoid unnecessary copies.
fn read_batch_from_file(
    file: &VortexFile,
    row_range: std::ops::Range<u64>,
    projection: Option<&Vec<String>>,
) -> VortexResult<Vec<ArrayRef>> {
    (*RUNTIME).block_on(async {
        let mut scan_builder = file.scan()?.with_row_range(row_range);

        // Apply projection if specified
        if let Some(columns) = projection {
            let column_names: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
            let projection_expr = select(column_names, root());
            scan_builder = scan_builder.with_projection(projection_expr);
        }

        let stream = scan_builder.into_array_stream()?;
        let chunks: Vec<ArrayRef> = stream.try_collect().await?;

        Ok(chunks)
    })
}

/// Expand a glob pattern to a list of file paths.
fn expand_glob(pattern: &str) -> VortexResult<Vec<String>> {
    if is_remote_path(pattern) {
        expand_remote_glob(pattern)
    } else {
        expand_local_glob(pattern)
    }
}

/// Expand a local glob pattern to a list of file paths.
fn expand_local_glob(pattern: &str) -> VortexResult<Vec<String>> {
    if pattern.contains('*') || pattern.contains('?') {
        let paths: Vec<_> = glob::glob(pattern)
            .map_err(|e| vortex_err!("Invalid glob pattern: {}", e))?
            .filter_map(|r| r.ok())
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        if paths.is_empty() {
            vortex_bail!("No files found matching pattern: {}", pattern);
        }

        Ok(paths)
    } else {
        if !Path::new(pattern).exists() {
            vortex_bail!("File not found: {}", pattern);
        }
        Ok(vec![pattern.to_string()])
    }
}

/// Expand a remote glob pattern using object_store's list API.
fn expand_remote_glob(pattern: &str) -> VortexResult<Vec<String>> {
    (*RUNTIME).block_on(async { expand_remote_glob_async(pattern).await })
}

/// Async implementation of remote glob expansion.
async fn expand_remote_glob_async(pattern: &str) -> VortexResult<Vec<String>> {
    let has_glob = pattern.contains('*') || pattern.contains('?');

    if !has_glob {
        let store_info = make_object_store(pattern)?;
        store_info
            .store
            .head(&store_info.path)
            .await
            .map_err(|e| vortex_err!("Remote file not found '{}': {}", pattern, e))?;
        return Ok(vec![pattern.to_string()]);
    }

    let (base_url, glob_pattern) = split_glob_pattern(pattern)?;

    let store_info = make_object_store(&base_url)?;
    let prefix = store_info.path.clone();

    let list_stream = store_info.store.list(Some(&prefix));
    let objects: Vec<_> = list_stream
        .try_collect()
        .await
        .map_err(|e| vortex_err!("Failed to list remote directory '{}': {}", base_url, e))?;

    let glob_matcher = glob::Pattern::new(&glob_pattern)
        .map_err(|e| vortex_err!("Invalid glob pattern '{}': {}", glob_pattern, e))?;

    let url = store_info.url;
    let base = format!("{}://{}", url.scheme(), url.host_str().unwrap_or_default());

    let matching_paths: Vec<String> = objects
        .into_iter()
        .filter(|obj| glob_matcher.matches(obj.location.as_ref()))
        .map(|obj| format!("{}/{}", base, obj.location))
        .collect();

    if matching_paths.is_empty() {
        vortex_bail!("No files found matching pattern: {}", pattern);
    }

    Ok(matching_paths)
}

/// Split a glob pattern into base URL and glob pattern.
fn split_glob_pattern(pattern: &str) -> VortexResult<(String, String)> {
    let glob_pos = pattern.find(|c| c == '*' || c == '?');

    match glob_pos {
        Some(pos) => {
            let base_end = pattern[..pos].rfind('/').unwrap_or(0);
            let base_url = pattern[..base_end].to_string();
            let glob_pattern = pattern[base_end + 1..].to_string();
            Ok((base_url, glob_pattern))
        }
        None => Ok((pattern.to_string(), String::new())),
    }
}

/// Read the schema from a Vortex file.
fn read_schema_from_file(path: &str) -> VortexResult<DType> {
    if is_remote_path(path) {
        read_schema_from_remote_file(path)
    } else {
        read_schema_from_local_file(path)
    }
}

/// Read the schema from a local Vortex file.
fn read_schema_from_local_file(path: &str) -> VortexResult<DType> {
    (*RUNTIME).block_on(async {
        let vortex_file = SESSION.open_options().open_path(path).await?;
        Ok(vortex_file.dtype().clone())
    })
}

/// Read the schema from a remote Vortex file.
fn read_schema_from_remote_file(path: &str) -> VortexResult<DType> {
    (*RUNTIME).block_on(async {
        let store_info = make_object_store(path)?;
        let vortex_file = SESSION
            .open_options()
            .open_object_store(&store_info.store, store_info.path.as_ref())
            .await?;
        Ok(vortex_file.dtype().clone())
    })
}

// =============================================================================
// FFI Exports for C++
// =============================================================================
//
// Thread safety: All FFI functions that take a `*const VortexScanner` or
// `*mut VortexScanner` assume exclusive, single-threaded access to that handle.
// The caller must never invoke any of these functions concurrently on the same
// scanner pointer. See the module-level and struct-level docs for details.

/// Create a new Vortex scanner.
///
/// # Safety
/// The `path` parameter must be a valid null-terminated C string.
/// Returns NULL on error. Call `vortex_get_last_error()` for error details.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_new(path: *const c_char) -> *mut VortexScanner {
    clear_last_error();

    if path.is_null() {
        set_last_error("vortex_scanner_new: path is null");
        return ptr::null_mut();
    }

    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("vortex_scanner_new: invalid UTF-8 in path: {}", e));
            return ptr::null_mut();
        }
    };

    match VortexScanner::new(path_str) {
        Ok(scanner) => Box::into_raw(Box::new(scanner)),
        Err(e) => {
            set_last_error(&format!("vortex_scanner_new: {}", e));
            ptr::null_mut()
        }
    }
}

/// Free a Vortex scanner.
///
/// # Safety
/// The `scanner` parameter must be a valid pointer returned by `vortex_scanner_new`,
/// or NULL (which is safely ignored).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_free(scanner: *mut VortexScanner) {
    if !scanner.is_null() {
        drop(unsafe { Box::from_raw(scanner) });
    }
}

/// Get the number of columns in the schema.
///
/// # Safety
/// The `scanner` parameter must be a valid pointer or NULL.
/// Returns 0 if scanner is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_num_columns(scanner: *const VortexScanner) -> usize {
    if scanner.is_null() {
        return 0;
    }
    unsafe { &*scanner }.num_columns()
}

/// Get a column name by index.
///
/// # Safety
/// - The `scanner` parameter must be a valid pointer.
/// - The returned string must be freed with `vortex_free_string()`.
/// - Returns NULL if index is out of bounds or scanner is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_column_name(
    scanner: *const VortexScanner,
    index: usize,
) -> *mut c_char {
    if scanner.is_null() {
        return ptr::null_mut();
    }

    unsafe { &*scanner }
        .column_name(index)
        .map(|name| {
            CString::new(name)
                .map(|c_string| c_string.into_raw())
                .unwrap_or(ptr::null_mut())
        })
        .unwrap_or(ptr::null_mut())
}

/// Get the ClickHouse type string for a column.
///
/// # Safety
/// - The `scanner` parameter must be a valid pointer.
/// - The returned string must be freed with `vortex_free_string()`.
/// - Returns NULL if index is out of bounds or scanner is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_column_type(
    scanner: *const VortexScanner,
    index: usize,
) -> *mut c_char {
    if scanner.is_null() {
        return ptr::null_mut();
    }

    unsafe { &*scanner }
        .column_type(index)
        .map(|type_str| {
            CString::new(type_str)
                .map(|c_string| c_string.into_raw())
                .unwrap_or(ptr::null_mut())
        })
        .unwrap_or(ptr::null_mut())
}

/// Set the columns to project (by index).
///
/// # Safety
/// - The `scanner` parameter must be a valid pointer.
/// - The `indices` parameter must point to an array of `num_indices` elements.
/// Returns 0 on success, non-zero on error. Call `vortex_get_last_error()` for details.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_set_projection(
    scanner: *mut VortexScanner,
    indices: *const usize,
    num_indices: usize,
) -> i32 {
    clear_last_error();

    if scanner.is_null() {
        set_last_error("vortex_scanner_set_projection: scanner is null");
        return -1;
    }

    if indices.is_null() && num_indices > 0 {
        set_last_error("vortex_scanner_set_projection: indices is null but num_indices > 0");
        return -2;
    }

    let scanner = unsafe { &mut *scanner };

    let indices_vec = if num_indices > 0 {
        unsafe { std::slice::from_raw_parts(indices, num_indices) }.to_vec()
    } else {
        // Empty projection means select all columns
        (0..scanner.num_columns()).collect()
    };

    match scanner.set_projection_indices(indices_vec) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(&format!("vortex_scanner_set_projection: {}", e));
            -3
        }
    }
}

/// Set the batch size for reading.
///
/// # Safety
/// The `scanner` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_set_batch_size(
    scanner: *mut VortexScanner,
    batch_size: usize,
) {
    if !scanner.is_null() {
        unsafe { &mut *scanner }.set_batch_size(batch_size);
    }
}

/// Check if there are more batches to read.
///
/// # Safety
/// The `scanner` parameter must be a valid pointer.
/// Returns 0 if no more data, 1 if more data available.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_has_more(scanner: *const VortexScanner) -> i32 {
    if scanner.is_null() {
        return 0;
    }
    if unsafe { &*scanner }.has_more() {
        1
    } else {
        0
    }
}

/// Opaque handle to a column exporter.
pub struct VortexExporterHandle {
    kind: ExporterKind,
    exporter: Box<dyn ColumnExporter>,
}

impl VortexExporterHandle {
    /// Create a new handle, reading the kind from the exporter itself.
    fn new(exporter: Box<dyn ColumnExporter>) -> Self {
        let kind = exporter.kind();
        Self { kind, exporter }
    }
}

/// Read the next batch of data.
///
/// # Safety
/// - The `scanner` parameter must be a valid pointer.
/// - Returns NULL if no more data or on error. Call `vortex_get_last_error()` for details.
/// - The returned handle must be freed with `vortex_exporter_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_read_batch(
    scanner: *mut VortexScanner,
) -> *mut VortexExporterHandle {
    clear_last_error();

    if scanner.is_null() {
        set_last_error("vortex_scanner_read_batch: scanner is null");
        return ptr::null_mut();
    }

    let scanner = unsafe { &mut *scanner };

    match scanner.read_next_batch() {
        Ok(Some(exporter)) => Box::into_raw(Box::new(VortexExporterHandle::new(exporter))),
        Ok(None) => ptr::null_mut(), // No error, just no more data
        Err(e) => {
            set_last_error(&format!("vortex_scanner_read_batch: {}", e));
            ptr::null_mut()
        }
    }
}

/// Get the number of files to scan.
///
/// # Safety
/// The `scanner` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_num_files(scanner: *const VortexScanner) -> usize {
    if scanner.is_null() {
        return 0;
    }
    unsafe { &*scanner }.num_files()
}

/// Get the current file index being scanned.
///
/// # Safety
/// The `scanner` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_current_file_index(scanner: *const VortexScanner) -> usize {
    if scanner.is_null() {
        return 0;
    }
    unsafe { &*scanner }.current_file_index()
}

/// Get the total number of rows read so far.
///
/// # Safety
/// The `scanner` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_total_rows_read(scanner: *const VortexScanner) -> u64 {
    if scanner.is_null() {
        return 0;
    }
    unsafe { &*scanner }.total_rows_read()
}

/// Get the total row count across all files.
///
/// This function reads metadata from all files to compute the total row count.
/// Note: This may be slow for large file sets as it opens each file's metadata.
///
/// # Safety
/// The `scanner` parameter must be a valid pointer.
/// Returns 0 on error. Call `vortex_get_last_error()` for error details.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_scanner_total_row_count(scanner: *const VortexScanner) -> u64 {
    clear_last_error();

    if scanner.is_null() {
        set_last_error("vortex_scanner_total_row_count: scanner is null");
        return 0;
    }

    match unsafe { &*scanner }.total_row_count() {
        Ok(count) => count,
        Err(e) => {
            set_last_error(&format!("vortex_scanner_total_row_count: {}", e));
            0
        }
    }
}

// =============================================================================
// Exporter FFI Functions
// =============================================================================

/// Free an exporter handle.
///
/// # Safety
/// The `handle` parameter must be a valid pointer returned by `vortex_scanner_read_batch`,
/// or NULL (which is safely ignored).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_free(handle: *mut VortexExporterHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Check if the exporter has more data.
///
/// # Safety
/// The `handle` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_has_more(handle: *const VortexExporterHandle) -> i32 {
    if handle.is_null() {
        return 0;
    }
    if unsafe { &*handle }.exporter.has_more() {
        1
    } else {
        0
    }
}

/// Get the total number of rows in the exporter.
///
/// # Safety
/// The `handle` parameter must be a valid pointer.
/// Returns 0 if handle is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_len(handle: *const VortexExporterHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    unsafe { &*handle }.exporter.len()
}

/// Export data to a buffer.
///
/// # Safety
/// - The `handle` parameter must be a valid pointer.
/// - The `buffer` must point to allocated memory of at least `buffer_size_bytes` bytes.
/// - `buffer_size_bytes` must be the total size of the buffer in bytes. The
///   exporter will refuse to write more data than fits in the buffer.
///   Use `vortex_exporter_element_size_bytes` to query the per-row size
///   and allocate `element_size * max_rows` bytes.
/// - Returns the number of rows exported, or negative on error.
///   Call `vortex_get_last_error()` for error details.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_export(
    handle: *mut VortexExporterHandle,
    buffer: *mut c_void,
    buffer_size_bytes: usize,
    max_rows: usize,
) -> i64 {
    clear_last_error();

    if handle.is_null() {
        set_last_error("vortex_exporter_export: handle is null");
        return -1;
    }
    if buffer.is_null() {
        set_last_error("vortex_exporter_export: buffer is null");
        return -1;
    }

    let handle = unsafe { &mut *handle };

    match handle.exporter.export(buffer, buffer_size_bytes, max_rows) {
        Ok(rows) => rows as i64,
        Err(e) => {
            set_last_error(&format!("vortex_exporter_export: {}", e));
            -2
        }
    }
}

/// Get the number of fields in a struct exporter.
///
/// # Safety
/// The `handle` parameter must be a valid pointer.
/// Returns 0 if not a struct exporter or handle is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_num_fields(handle: *const VortexExporterHandle) -> usize {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };

    // Try to downcast to StructExporter
    if let Some(struct_exporter) = handle
        .exporter
        .as_any()
        .downcast_ref::<crate::exporter::StructExporter>()
    {
        struct_exporter.num_fields()
    } else {
        0
    }
}

/// Get a field exporter from a struct exporter.
///
/// # Safety
/// - The `handle` parameter must be a valid pointer to a struct exporter.
/// - Returns NULL if not a struct exporter or index is out of bounds.
/// - The returned handle must be freed with `vortex_exporter_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_get_field(
    handle: *mut VortexExporterHandle,
    index: usize,
) -> *mut VortexExporterHandle {
    clear_last_error();

    if handle.is_null() {
        set_last_error("vortex_exporter_get_field: handle is null");
        return ptr::null_mut();
    }

    let handle = unsafe { &mut *handle };

    // Try to downcast to StructExporter
    if let Some(struct_exporter) = handle
        .exporter
        .as_any_mut()
        .downcast_mut::<crate::exporter::StructExporter>()
    {
        if let Some(field_exporter) = struct_exporter.take_field_exporter(index) {
            Box::into_raw(Box::new(VortexExporterHandle::new(field_exporter)))
        } else {
            set_last_error(&format!(
                "vortex_exporter_get_field: field index {} out of bounds or already taken",
                index
            ));
            ptr::null_mut()
        }
    } else {
        set_last_error("vortex_exporter_get_field: handle is not a struct exporter");
        ptr::null_mut()
    }
}

/// Check if the exporter is for nullable data.
///
/// # Safety
/// The `handle` parameter must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_is_nullable(handle: *const VortexExporterHandle) -> i32 {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };

    if handle.exporter.is_nullable() { 1 } else { 0 }
}

/// Export validity (null) bitmap.
///
/// # Safety
/// - The `handle` parameter must be a valid pointer.
/// - The `validity_bitmap` must point to allocated memory of at least (max_rows + 7) / 8 bytes.
/// - Returns the number of rows, or negative on error.
///   Call `vortex_get_last_error()` for error details.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_export_validity(
    handle: *mut VortexExporterHandle,
    validity_bitmap: *mut u8,
    max_rows: usize,
) -> i64 {
    clear_last_error();

    if handle.is_null() {
        set_last_error("vortex_exporter_export_validity: handle is null");
        return -1;
    }
    if validity_bitmap.is_null() {
        set_last_error("vortex_exporter_export_validity: validity_bitmap is null");
        return -1;
    }

    let handle = unsafe { &mut *handle };

    match handle.exporter.export_validity(validity_bitmap, max_rows) {
        Ok(rows) => rows as i64,
        Err(e) => {
            set_last_error(&format!("vortex_exporter_export_validity: {}", e));
            -2
        }
    }
}

/// Export string data.
///
/// # Safety
/// - The `handle` parameter must be a valid pointer to a string exporter.
/// - All buffer pointers must be valid and properly sized.
/// - Returns the number of rows exported, or negative on error.
///   Call `vortex_get_last_error()` for error details.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_export_strings(
    handle: *mut VortexExporterHandle,
    data: *mut u8,
    lengths: *mut u32,
    offsets: *mut u64,
    max_rows: usize,
) -> i64 {
    clear_last_error();

    if handle.is_null() {
        set_last_error("vortex_exporter_export_strings: handle is null");
        return -1;
    }
    if data.is_null() || lengths.is_null() || offsets.is_null() {
        set_last_error("vortex_exporter_export_strings: one or more buffer pointers are null");
        return -1;
    }

    let handle = unsafe { &mut *handle };

    match handle
        .exporter
        .export_strings(data, lengths, offsets, max_rows)
    {
        Ok(rows) => rows as i64,
        Err(e) => {
            set_last_error(&format!("vortex_exporter_export_strings: {}", e));
            -2
        }
    }
}

/// Get the total size of string data for remaining rows in the exporter.
///
/// This is useful for pre-allocating buffers before calling `vortex_exporter_export_strings`.
///
/// # Safety
/// - The `handle` parameter must be a valid pointer to a string exporter.
/// - The `total_bytes` and `num_rows` parameters must be valid pointers.
/// - Returns 0 on success, negative on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_string_data_size(
    handle: *const VortexExporterHandle,
    total_bytes: *mut usize,
    num_rows: *mut usize,
) -> i32 {
    clear_last_error();

    if handle.is_null() {
        set_last_error("vortex_exporter_string_data_size: handle is null");
        return -1;
    }
    if total_bytes.is_null() || num_rows.is_null() {
        set_last_error("vortex_exporter_string_data_size: output pointers are null");
        return -1;
    }

    let handle = unsafe { &*handle };

    match handle.exporter.string_data_size() {
        Ok((bytes, rows)) => {
            unsafe {
                *total_bytes = bytes;
                *num_rows = rows;
            }
            0
        }
        Err(e) => {
            set_last_error(&format!("vortex_exporter_string_data_size: {}", e));
            -2
        }
    }
}

// =============================================================================
// List/Array Exporter FFI Functions
// =============================================================================

/// Check if the exporter is a list (array) exporter.
///
/// # Safety
/// The `handle` parameter must be a valid pointer.
/// Returns 1 if it's a list exporter, 0 otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_is_list(handle: *const VortexExporterHandle) -> i32 {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };
    if handle.kind == ExporterKind::List {
        1
    } else {
        0
    }
}

/// Get the exporter kind tag.
///
/// # Safety
/// The `handle` parameter must be a valid pointer.
/// Returns the `ExporterKind` as a u8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_kind(handle: *const VortexExporterHandle) -> u8 {
    if handle.is_null() {
        return 0;
    }
    unsafe { &*handle }.kind as u8
}

/// Get the number of bytes each row occupies in the export buffer.
///
/// For fixed-width exporters (Primitive, Bool, BigInt, Decimal) this returns
/// the element width in bytes.  Variable-length exporters (String, List,
/// Struct) return 0 because they use specialised export paths.
///
/// The value returned by this function can be multiplied by the desired row
/// count to compute the `buffer_size_bytes` argument for
/// `vortex_exporter_export`.
///
/// # Safety
/// The `handle` parameter must be a valid pointer.
/// Returns 0 if the handle is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_element_size_bytes(
    handle: *const VortexExporterHandle,
) -> usize {
    if handle.is_null() {
        return 0;
    }
    unsafe { &*handle }.exporter.element_size_bytes()
}

/// Export list offsets.
///
/// The offsets array will have `num_rows + 1` elements written, where:
/// - `offsets[i]` is the start index of array i in the flattened elements
/// - `offsets[num_rows]` is the total number of elements
///
/// # Safety
/// - The `handle` parameter must be a valid pointer to a list exporter.
/// - The `offsets` buffer must have space for `max_rows + 1` uint64_t values.
/// - Returns the number of rows (not elements) exported, or negative on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_export_list_offsets(
    handle: *mut VortexExporterHandle,
    offsets: *mut u64,
    max_rows: usize,
) -> i64 {
    clear_last_error();

    if handle.is_null() {
        set_last_error("vortex_exporter_export_list_offsets: handle is null");
        return -1;
    }
    if offsets.is_null() {
        set_last_error("vortex_exporter_export_list_offsets: offsets is null");
        return -1;
    }

    let handle = unsafe { &mut *handle };

    // Try to downcast to ListExporter
    if let Some(list_exporter) = handle
        .exporter
        .as_any_mut()
        .downcast_mut::<crate::exporter::ListExporter>()
    {
        match list_exporter.export_offsets(offsets, max_rows) {
            Ok(rows) => rows as i64,
            Err(e) => {
                set_last_error(&format!("vortex_exporter_export_list_offsets: {}", e));
                -2
            }
        }
    } else {
        set_last_error("vortex_exporter_export_list_offsets: handle is not a list exporter");
        -3
    }
}

/// Get the element exporter from a list exporter.
///
/// This returns an exporter for the flattened elements of all arrays.
/// The returned exporter should be used to export the element data.
///
/// # Safety
/// - The `handle` parameter must be a valid pointer to a list exporter.
/// - Returns NULL if not a list exporter or on error.
/// - The returned handle must be freed with `vortex_exporter_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_get_list_elements(
    handle: *mut VortexExporterHandle,
) -> *mut VortexExporterHandle {
    clear_last_error();

    if handle.is_null() {
        set_last_error("vortex_exporter_get_list_elements: handle is null");
        return ptr::null_mut();
    }

    let handle = unsafe { &mut *handle };

    // Try to downcast to ListExporter
    if let Some(list_exporter) = handle
        .exporter
        .as_any_mut()
        .downcast_mut::<crate::exporter::ListExporter>()
    {
        match list_exporter.take_element_exporter() {
            Ok(elem_exporter) => Box::into_raw(Box::new(VortexExporterHandle::new(elem_exporter))),
            Err(e) => {
                set_last_error(&format!("vortex_exporter_get_list_elements: {}", e));
                ptr::null_mut()
            }
        }
    } else {
        set_last_error("vortex_exporter_get_list_elements: handle is not a list exporter");
        ptr::null_mut()
    }
}

/// Get the total number of elements in all arrays (for a list exporter).
///
/// This is useful for pre-allocating the element buffer.
///
/// # Safety
/// The `handle` parameter must be a valid pointer to a list exporter.
/// Returns 0 if not a list exporter or on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vortex_exporter_list_total_elements(
    handle: *const VortexExporterHandle,
) -> usize {
    if handle.is_null() {
        return 0;
    }

    let handle = unsafe { &*handle };

    // Try to downcast to ListExporter
    if let Some(list_exporter) = handle
        .exporter
        .as_any()
        .downcast_ref::<crate::exporter::ListExporter>()
    {
        list_exporter.total_elements().unwrap_or(0)
    } else {
        0
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_glob_pattern() {
        let (base, pattern) = split_glob_pattern("s3://bucket/data/*.vortex").unwrap();
        assert_eq!(base, "s3://bucket/data");
        assert_eq!(pattern, "*.vortex");

        let (base, pattern) = split_glob_pattern("s3://bucket/data/prefix_*.vortex").unwrap();
        assert_eq!(base, "s3://bucket/data");
        assert_eq!(pattern, "prefix_*.vortex");

        let (base, pattern) = split_glob_pattern("s3://bucket/a/b/c/*.vortex").unwrap();
        assert_eq!(base, "s3://bucket/a/b/c");
        assert_eq!(pattern, "*.vortex");
    }

    #[test]
    fn test_is_glob_pattern() {
        assert!("*.vortex".contains('*'));
        assert!("prefix_?.vortex".contains('?'));
        assert!(!"/path/to/file.vortex".contains('*'));
        assert!(!"/path/to/file.vortex".contains('?'));
    }

    #[test]
    fn test_scanner_new_empty_path() {
        let result = VortexScanner::new("");
        assert!(result.is_err());
    }

    #[test]
    fn test_scanner_new_invalid_path() {
        let result = VortexScanner::new("/nonexistent/path/file.vortex");
        assert!(result.is_err());
    }
}
