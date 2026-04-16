// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FFI interface for Vortex File I/O.

use std::ffi::CStr;
use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_uint;
use std::ffi::c_ulong;
use std::ops::Range;
use std::slice;
use std::str::FromStr;
use std::sync::Arc;

use itertools::Itertools;
use object_store::ObjectStore;
use object_store::ObjectStoreScheme;
use object_store::aws::AmazonS3Builder;
use object_store::aws::AmazonS3ConfigKey;
use object_store::azure::AzureConfigKey;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::gcp::GoogleConfigKey;
use object_store::local::LocalFileSystem;
use prost::Message;
use url::Url;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::stream::ArrayStream;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::proto::expr::Expr;
use vortex::session::VortexSession;

use crate::RUNTIME;
use crate::arc_wrapper;
use crate::array::vx_array;
use crate::array_iterator::vx_array_iterator;
use crate::dtype::vx_dtype;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::session::vx_session;
use crate::to_string_vec;

arc_wrapper!(
    /// A handle to a Vortex file encapsulating the footer and logic for instantiating a reader.
    VortexFile,
    vx_file
);

/// Options supplied for opening a file.
#[repr(C)]
// FIXME(ngates): we cannot have transparent structs in FFI since we cannot break them.
pub struct vx_file_open_options {
    /// URI for opening the file.
    /// This must be a valid URI, even for files (file:///path/to/file)
    pub uri: *const c_char,
    /// Additional configuration for the file source (e.g. "s3.accessKey").
    /// This may be null, in which case it is treated as empty.
    pub property_keys: *const *const c_char,
    /// Additional configuration values for the file source (e.g. S3 credentials).
    pub property_vals: *const *const c_char,
    /// Number of properties in `property_keys` and `property_vals`.
    pub property_len: c_int,
}

/// Scan options provided by an FFI client calling the `vx_file_scan` function.
#[repr(C)]
// FIXME(ngates): we cannot have transparent structs in FFI since we cannot break them.
pub struct vx_file_scan_options {
    /// Column names to project out in the scan. These must be null-terminated C strings.
    pub projection_expression: *const c_char,

    /// Number of columns in `projection`.
    pub projection_expr_len: c_uint,

    /// Serialized expressions for pushdown
    pub filter_expression: *const c_char,

    /// The len in bytes of the filter expression
    pub filter_expression_len: c_uint,

    /// Splits the file into chunks of this size, if zero then we use the write layout.
    pub split_by_row_count: c_int,

    /// First row of a range to scan.
    pub row_range_start: c_ulong,

    /// Last row of a range to scan.
    pub row_range_end: c_ulong,

    /// The row offset of the file in a multi-file scan.
    pub row_offset: c_ulong,
}

fn extract_expression(
    session: &VortexSession,
    expression: *const c_char,
    expression_len: c_uint,
) -> VortexResult<Option<Expression>> {
    Ok((!expression.is_null() && expression_len > 0).then_some({
        let bytes =
            unsafe { slice::from_raw_parts(expression.cast::<u8>(), expression_len as usize) };

        // Decode the protobuf message.
        Expression::from_proto(&Expr::decode(bytes)?, session)
            .map_err(|e| e.with_context("deserializing expr"))?
    }))
}

impl vx_file_scan_options {
    /// Processes FFI scan options.
    ///
    /// Extracts and converts a scan configuration from an FFI options struct.
    fn process_scan_options(&self, session: &VortexSession) -> VortexResult<ScanOptions> {
        // Extract field names for projection.
        let projection_expr = extract_expression(
            session,
            self.projection_expression,
            self.projection_expr_len,
        )?;

        let filter_expr =
            extract_expression(session, self.filter_expression, self.filter_expression_len)?;

        // On Windows, c_ulong is u32, so we need to convert to u64
        // On Unix, c_ulong is already u64, so we can use it directly
        #[cfg(windows)]
        let row_range = (self.row_range_end > self.row_range_start)
            .then_some(self.row_range_start as u64..self.row_range_end as u64);
        #[cfg(not(windows))]
        let row_range = (self.row_range_end > self.row_range_start)
            .then_some(self.row_range_start..self.row_range_end);

        let split_by = (self.split_by_row_count > 0)
            .then_some(SplitBy::RowCount(self.split_by_row_count as usize));

        #[cfg(windows)]
        let row_offset = self.row_offset as u64;
        #[cfg(not(windows))]
        let row_offset = self.row_offset;

        Ok(ScanOptions {
            projection_expr,
            filter_expr,
            split_by,
            row_range,
            row_offset,
        })
    }
}

/// Open a file at the given path on the file system.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_open_reader(
    session: *const vx_session,
    options: *const vx_file_open_options,
    error_out: *mut *mut vx_error,
) -> *const vx_file {
    let session = vx_session::as_ref(session);

    try_or_default(error_out, || {
        let options = unsafe {
            options
                .as_ref()
                .ok_or_else(|| vortex_err!("null options"))?
        };

        if options.uri.is_null() {
            vortex_bail!("null uri")
        }
        let uri_str = unsafe { CStr::from_ptr(options.uri) }.to_string_lossy();
        let uri: Url = uri_str
            .parse()
            .map_err(|e| vortex_err!("Failed to parse URI '{}': {}", uri_str, e))?;

        let prop_keys =
            unsafe { to_string_vec(options.property_keys, options.property_len as usize) };
        let prop_vals =
            unsafe { to_string_vec(options.property_vals, options.property_len as usize) };

        let object_store = make_object_store(&uri, &prop_keys, &prop_vals)?;

        let file = session.open_options();
        let vxf = RUNTIME
            .block_on(async move { file.open_object_store(&object_store, uri.path()).await })?;

        Ok(vx_file::new(Arc::new(vxf)))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_write_array(
    session: *const vx_session,
    path: *const c_char,
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) {
    let array = vx_array::as_ref(array);
    try_or_default(error_out, || {
        let session = vx_session::as_ref(session);

        let path = unsafe { CStr::from_ptr(path) }
            .to_str()
            .map_err(|e| vortex_err!("invalid utf-8: {e}"))?;

        let options = session.write_options();
        RUNTIME.block_on(async move {
            options
                .write(
                    &mut async_fs::File::create(path).await?,
                    array.to_array_stream(),
                )
                .await?;
            Ok(())
        })
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_row_count(file: *const vx_file) -> u64 {
    vx_file::as_ref(file).row_count()
}

#[derive(Default, Debug)]
struct ScanOptions {
    projection_expr: Option<Expression>,
    filter_expr: Option<Expression>,
    split_by: Option<SplitBy>,
    row_range: Option<Range<u64>>,
    row_offset: u64,
}

/// Return the DType of the file.
///
/// The returned pointer is valid as long as the file is valid.
/// Do NOT free the returned dtype pointer - it shares the lifetime of the file.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_dtype(file: *const vx_file) -> *const vx_dtype {
    vx_dtype::new_ref(vx_file::as_ref(file).dtype())
}

/// Can we prune the whole file using file stats and an expression
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_can_prune(
    session: *const vx_session,
    file: *const vx_file,
    filter_expression: *const c_char,
    filter_expression_len: c_uint,
    error_out: *mut *mut vx_error,
) -> bool {
    try_or_default(error_out, || {
        let session = vx_session::as_ref(session);
        let file = vx_file::as_ref(file);
        let filter_expr = extract_expression(session, filter_expression, filter_expression_len)?;
        Ok(filter_expr
            .map(|expr| file.can_prune(&expr))
            .transpose()?
            .unwrap_or(false))
    })
}

/// Build a new `vx_array_iterator` that returns a series of `vx_array`s from a scan over a `vx_layout_reader`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_scan(
    session: *const vx_session,
    file: *const vx_file,
    opts: *const vx_file_scan_options,
    error_out: *mut *mut vx_error,
) -> *mut vx_array_iterator {
    try_or_default(error_out, || {
        let session = vx_session::as_ref(session);
        let file = vx_file::as_ref(file);

        let scan_options = unsafe { opts.as_ref() }.map_or_else(
            || Ok(ScanOptions::default()),
            |options| options.process_scan_options(session),
        )?;

        let layout_reader = file.layout_reader()?;
        let mut scan_builder = ScanBuilder::new(session.clone(), layout_reader)
            .with_row_offset(scan_options.row_offset);

        // Apply options if provided.
        if let Some(projection_expr) = scan_options.projection_expr {
            scan_builder = scan_builder.with_projection(projection_expr);
        }

        if let Some(expr) = scan_options.filter_expr {
            scan_builder = scan_builder.with_filter(expr);
        }

        if let Some(range) = scan_options.row_range {
            scan_builder = scan_builder.with_row_range(range);
        }

        if let Some(split_by_value) = scan_options.split_by {
            scan_builder = scan_builder.with_split_by(split_by_value);
        }

        let stream = scan_builder.into_array_stream()?;
        let iter =
            ArrayIteratorAdapter::new(stream.dtype().clone(), RUNTIME.block_on_stream(stream));

        Ok(vx_array_iterator::new(Box::new(iter)))
    })
}

#[expect(clippy::cognitive_complexity)]
fn make_object_store(
    url: &Url,
    property_keys: &[String],
    property_vals: &[String],
) -> VortexResult<Arc<dyn ObjectStore>> {
    let (scheme, _) = ObjectStoreScheme::parse(url)
        .map_err(|error| VortexError::from(object_store::Error::from(error)))?;

    if property_vals.len() != property_keys.len() {
        vortex_bail!(
            "property_vals len: {}, != property_keys len {}",
            property_vals.len(),
            property_keys.len()
        )
    }

    // Configure extra properties on that scheme instead.
    match scheme {
        ObjectStoreScheme::Local => {
            tracing::trace!("using LocalFileSystem object store");
            Ok(Arc::new(LocalFileSystem::default()))
        }
        ObjectStoreScheme::AmazonS3 => {
            tracing::trace!("using AmazonS3 object store");
            let mut builder = AmazonS3Builder::new().with_url(url.to_string());
            for (key, val) in property_keys.iter().zip_eq(property_vals.iter()) {
                if let Ok(config_key) = AmazonS3ConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    tracing::warn!("Skipping unknown Amazon S3 config key: {key}");
                }
            }

            if property_keys.is_empty() {
                builder = AmazonS3Builder::from_env();
                if let Some(domain) = url.domain() {
                    builder = builder.with_bucket_name(domain);
                }
            }

            let store = Arc::new(builder.build()?);
            Ok(store)
        }
        ObjectStoreScheme::MicrosoftAzure => {
            tracing::trace!("using MicrosoftAzure object store");

            let mut builder = MicrosoftAzureBuilder::new().with_url(url.to_string());
            for (key, val) in property_keys.iter().zip(property_vals.iter()) {
                if let Ok(config_key) = AzureConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    tracing::warn!("Skipping unknown Azure config key: {key}");
                }
            }

            let store = Arc::new(builder.build()?);
            Ok(store)
        }
        ObjectStoreScheme::GoogleCloudStorage => {
            tracing::trace!("using GoogleCloudStorage object store");

            let mut builder = GoogleCloudStorageBuilder::new().with_url(url.to_string());
            for (key, val) in property_keys.iter().zip(property_vals.iter()) {
                if let Ok(config_key) = GoogleConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    tracing::warn!("Skipping unknown Google Cloud Storage config key: {key}");
                }
            }

            let store = Arc::new(builder.build()?);
            Ok(store)
        }
        store => {
            vortex_bail!("Unsupported store scheme: {store:?}");
        }
    }
}
