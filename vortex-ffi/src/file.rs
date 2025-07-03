// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FFI interface for Vortex File I/O.

use std::ffi::{CStr, c_char, c_int, c_uint, c_ulong};
use std::ops::Range;
use std::str::FromStr;
use std::sync::Arc;
use std::{ptr, slice};

use itertools::Itertools;
use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::local::LocalFileSystem;
use object_store::{ObjectStore, ObjectStoreScheme};
use prost::Message;
use url::Url;
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{ExprRef, deserialize_expr};
use vortex::file::scan::SplitBy;
use vortex::file::{VortexFile, VortexOpenOptions, VortexWriteOptions};
use vortex::layout::layouts::row_id::RowIdLayoutReader;
use vortex::layout::scan::ScanBuilder;
use vortex::proto::expr::Expr;

use crate::array::vx_array;
use crate::array_iterator::vx_array_iterator;
use crate::dtype::vx_dtype;
use crate::error::{try_or, vx_error};
use crate::session::{FileKey, vx_session};
use crate::{RUNTIME, arc_wrapper, to_string_vec};

arc_wrapper!(
    /// A handle to a Vortex file encapsulating ther footer and logic for instantiating a reader.
    VortexFile,
    vx_file
);

/// Options supplied for opening a file.
#[repr(C)]
#[allow(non_camel_case_types)]
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
#[allow(non_camel_case_types)]
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

    /// The index of the file in a multi-file scan.
    pub file_index: c_ulong,
}

fn extract_expression(
    expression: *const c_char,
    expression_len: c_uint,
) -> VortexResult<Option<ExprRef>> {
    Ok((!expression.is_null() && expression_len > 0).then_some({
        let bytes =
            unsafe { slice::from_raw_parts(expression as *const u8, expression_len as usize) };

        // Decode the protobuf message.
        deserialize_expr(&Expr::decode(bytes)?).map_err(|e| e.with_context("deserializing expr"))?
    }))
}

impl vx_file_scan_options {
    /// Processes FFI scan options.
    ///
    /// Extracts and converts a scan configuration from an FFI options struct.
    fn process_scan_options(&self) -> VortexResult<ScanOptions> {
        // Extract field names for projection.
        let projection_expr =
            extract_expression(self.projection_expression, self.projection_expr_len)?;

        let filter_expr = extract_expression(self.filter_expression, self.filter_expression_len)?;

        let row_range = (self.row_range_end > self.row_range_start)
            .then_some(self.row_range_start..self.row_range_end);

        let split_by = (self.split_by_row_count > 0)
            .then_some(SplitBy::RowCount(self.split_by_row_count as usize));

        Ok(ScanOptions {
            projection_expr,
            filter_expr,
            split_by,
            row_range,
            file_index: self.file_index,
        })
    }
}

/// Open a file at the given path on the file system.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_open_reader(
    options: *const vx_file_open_options,
    session: *const vx_session,
    error_out: *mut *mut vx_error,
) -> *const vx_file {
    let session = vx_session::as_ref(session);

    try_or(error_out, ptr::null_mut(), || {
        let options = unsafe {
            options
                .as_ref()
                .ok_or_else(|| vortex_err!("null options"))?
        };

        if options.uri.is_null() {
            vortex_bail!("null uri")
        }
        let uri_str = unsafe { CStr::from_ptr(options.uri) }.to_string_lossy();
        let uri: Url = uri_str.parse().vortex_expect("File_open: parse uri");

        let prop_keys = unsafe { to_string_vec(options.property_keys, options.property_len) };
        let prop_vals = unsafe { to_string_vec(options.property_vals, options.property_len) };

        let object_store = make_object_store(&uri, &prop_keys, &prop_vals)?;

        let mut file = VortexOpenOptions::file();
        let mut cache_hit = false;
        if let Some(footer) = session.get_footer(&FileKey {
            location: uri_str.to_string(),
        }) {
            file = file.with_footer(footer);
            cache_hit = true;
        }

        let vxf = RUNTIME
            .block_on(async move { file.open_object_store(&object_store, uri.path()).await })?;

        if !cache_hit {
            session.put_footer(
                FileKey {
                    location: uri_str.to_string(),
                },
                vxf.footer().clone(),
            );
        }

        Ok(vx_file::new(Arc::new(vxf)))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_write_array(
    path: *const c_char,
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) {
    let array = vx_array::as_ref(array);
    try_or(error_out, (), || {
        let path = unsafe { CStr::from_ptr(path).to_str()? };

        RUNTIME.block_on(async {
            VortexWriteOptions::default()
                .write(
                    &mut tokio::fs::File::create(path).await?,
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
    projection_expr: Option<ExprRef>,
    filter_expr: Option<ExprRef>,
    split_by: Option<SplitBy>,
    row_range: Option<Range<u64>>,
    file_index: u64,
}

/// Return a borrowed reference to the DType of the file.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_dtype(file: *const vx_file) -> *const vx_dtype {
    vx_dtype::new_ref(vx_file::as_ref(file).dtype())
}

/// Can we prune the whole file using file stats and an expression
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_can_prune(
    file: *const vx_file,
    filter_expression: *const c_char,
    filter_expression_len: c_uint,
    file_idx: c_ulong,
    error_out: *mut *mut vx_error,
) -> bool {
    try_or(error_out, false, || {
        let file = vx_file::as_ref(file);
        let filter_expr = extract_expression(filter_expression, filter_expression_len)?;
        Ok(filter_expr
            .map(|expr| file.can_prune(&expr, file_idx))
            .transpose()?
            .unwrap_or(false))
    })
}

/// Build a new `vx_array_iterator` that returns a series of `vx_array`s from a scan over a `vx_layout_reader`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_scan(
    file: *const vx_file,
    opts: *const vx_file_scan_options,
    error_out: *mut *mut vx_error,
) -> *mut vx_array_iterator {
    try_or(error_out, ptr::null_mut(), || {
        let file = vx_file::as_ref(file);

        let scan_options = unsafe { opts.as_ref() }.map_or_else(
            || Ok(ScanOptions::default()),
            |options| options.process_scan_options(),
        )?;

        let layout_reader = file.layout_reader()?;
        let layout_reader = Arc::new(RowIdLayoutReader::new_with_file_index(
            layout_reader,
            scan_options.file_index,
        ));
        let mut scan_builder = ScanBuilder::new(layout_reader);

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

        Ok(vx_array_iterator::new(Box::new(
            scan_builder.into_array_iter()?,
        )))
    })
}

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
            log::trace!("using LocalFileSystem object store");
            Ok(Arc::new(LocalFileSystem::default()))
        }
        ObjectStoreScheme::AmazonS3 => {
            log::trace!("using AmazonS3 object store");
            let mut builder = AmazonS3Builder::new().with_url(url.to_string());
            for (key, val) in property_keys.iter().zip_eq(property_vals.iter()) {
                if let Ok(config_key) = AmazonS3ConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    log::warn!("Skipping unknown Amazon S3 config key: {key}");
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
            log::trace!("using MicrosoftAzure object store");

            let mut builder = MicrosoftAzureBuilder::new().with_url(url.to_string());
            for (key, val) in property_keys.iter().zip(property_vals.iter()) {
                if let Ok(config_key) = AzureConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    log::warn!("Skipping unknown Azure config key: {key}");
                }
            }

            let store = Arc::new(builder.build()?);
            Ok(store)
        }
        ObjectStoreScheme::GoogleCloudStorage => {
            log::trace!("using GoogleCloudStorage object store");

            let mut builder = GoogleCloudStorageBuilder::new().with_url(url.to_string());
            for (key, val) in property_keys.iter().zip(property_vals.iter()) {
                if let Ok(config_key) = GoogleConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    log::warn!("Skipping unknown Google Cloud Storage config key: {key}");
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
