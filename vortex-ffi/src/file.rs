//! FFI interface for Vortex File I/O.

#![allow(non_camel_case_types)]

use std::ffi::{CStr, c_char, c_int, c_uint, c_ulong};
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
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{ExprRef, Identity, deserialize_expr, select};
use vortex::file::scan::SplitBy;
use vortex::file::{VortexFile, VortexOpenOptions, VortexWriteOptions};
use vortex::layout::LayoutReader;
use vortex::layout::scan::ScanBuilder;
use vortex::proto::expr::Expr;

use crate::array::{vx_array, vx_array_iter};
use crate::error::{try_or, vx_error};
use crate::{RUNTIME, to_string, to_string_vec};

/// A file reader that can be used to read from a file.
pub struct vx_file_reader {
    pub inner: VortexFile,
}

/// A Vortex layout reader.
pub struct vx_layout_reader {
    pub inner: Arc<dyn LayoutReader>,
}

/// Options supplied for opening a file.
#[repr(C)]
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
pub struct vx_file_scan_options {
    /// Column names to project out in the scan. These must be null-terminated C strings.
    pub projection: *const *const c_char,

    /// Number of columns in `projection`.
    pub projection_len: c_uint,

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
}

impl vx_file_scan_options {
    /// Processes FFI scan options.
    ///
    /// Extracts and converts a scan configuration from an FFI options struct.
    fn process_scan_options(&self) -> VortexResult<ScanOptions> {
        // Extract field names for projection.
        let field_names = (0..self.projection_len)
            .map(|idx| unsafe { to_string(*self.projection.add(idx as usize)).into() })
            .collect::<Vec<Arc<str>>>();

        let filter_expr = (!self.filter_expression.is_null() && self.filter_expression_len > 0)
            .then_some({
                let bytes = unsafe {
                    slice::from_raw_parts(
                        self.filter_expression as *const u8,
                        self.filter_expression_len as usize,
                    )
                };

                // Decode the protobuf message.
                deserialize_expr(&Expr::decode(bytes)?)
                    .map_err(|e| e.with_context("deserializing expr"))?
            });

        let row_range = (self.row_range_end > self.row_range_start)
            .then_some(self.row_range_start..self.row_range_end);

        let split_by = (self.split_by_row_count > 0)
            .then_some(SplitBy::RowCount(self.split_by_row_count as usize));

        Ok(ScanOptions {
            field_names: Some(field_names),
            filter_expr,
            split_by,
            row_range,
        })
    }
}

/// Open a file at the given path on the file system.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_open_reader(
    options: *const vx_file_open_options,
    error: *mut *mut vx_error,
) -> *mut vx_file_reader {
    try_or(error, ptr::null_mut(), || {
        let options = unsafe {
            options
                .as_ref()
                .ok_or_else(|| vortex_err!("null options"))?
        };

        if options.uri.is_null() {
            vortex_bail!("null uri")
        }
        let uri = unsafe { CStr::from_ptr(options.uri) }.to_string_lossy();
        let uri: Url = uri.parse().vortex_expect("File_open: parse uri");

        let prop_keys = unsafe { to_string_vec(options.property_keys, options.property_len) };
        let prop_vals = unsafe { to_string_vec(options.property_vals, options.property_len) };

        let object_store = make_object_store(&uri, &prop_keys, &prop_vals)?;

        let inner = RUNTIME.block_on(async move {
            VortexOpenOptions::file()
                .open_object_store(&object_store, uri.path())
                .await
        })?;

        Ok(Box::into_raw(Box::new(vx_file_reader { inner })))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_write_array(
    path: *const c_char,
    ffi_array: *mut vx_array,
    error: *mut *mut vx_error,
) {
    try_or(error, (), || {
        let array = unsafe { ffi_array.as_ref().vortex_expect("null array") };
        let path = unsafe { CStr::from_ptr(path).to_str()? };

        RUNTIME.block_on(async {
            VortexWriteOptions::default()
                .write(
                    &mut tokio::fs::File::create(path).await?,
                    array.inner.to_array_stream(),
                )
                .await?;
            Ok(())
        })
    });
}

/// Whole file statistics.
#[repr(C)]
pub struct vx_file_statistics {
    /// The exact number of rows in the file.
    pub num_rows: u64,
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_extract_statistics(
    file: *mut vx_file_reader,
) -> *mut vx_file_statistics {
    Box::into_raw(Box::new(vx_file_statistics {
        num_rows: unsafe { file.as_ref() }
            .vortex_expect("null file ptr")
            .inner
            .row_count(),
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_statistics_free(stat: *mut vx_file_statistics) {
    assert!(!stat.is_null());
    drop(unsafe { Box::from_raw(stat) });
}

#[derive(Default)]
struct ScanOptions {
    field_names: Option<Vec<Arc<str>>>,
    filter_expr: Option<ExprRef>,
    split_by: Option<SplitBy>,
    row_range: Option<std::ops::Range<u64>>,
}

/// Get the DType of the data inside of the file.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_dtype(file: *const vx_file_reader) -> *mut DType {
    Box::into_raw(Box::new(
        unsafe { file.as_ref() }
            .vortex_expect("null file")
            .inner
            .dtype()
            .clone(),
    ))
}

/// Build a new `vx_array_iter` that returns a series of `vx_array`s from a scan over a `vx_layout_reader`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_layout_reader_scan(
    layout_reader: *const vx_layout_reader,
    opts: *const vx_file_scan_options,
    error: *mut *mut vx_error,
) -> *mut vx_array_iter {
    try_or(error, ptr::null_mut(), || {
        let layout_reader = unsafe { layout_reader.as_ref().vortex_expect("null layout reader") };
        let mut scan_builder = ScanBuilder::new(layout_reader.inner.clone());

        let scan_options = unsafe { opts.as_ref() }.map_or_else(
            || Ok(ScanOptions::default()),
            |options| options.process_scan_options(),
        )?;

        // Apply options if provided.
        if let Some(field_names) = scan_options.field_names {
            // Field names are allowed to be `Some` and empty.
            scan_builder = scan_builder.with_projection(select(field_names, Identity::new_expr()));
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

        vx_array_iter(scan_builder.into_array_iter()?)
    })
}

/// Returns the row count for a given file reader.
#[unsafe(no_mangle)]
pub extern "C-unwind" fn vx_file_row_count(
    file_reader: *mut vx_file_reader,
    error: *mut *mut vx_error,
) -> u64 {
    try_or(error, 0, || {
        let file_reader = unsafe { file_reader.as_ref().vortex_expect("null file_reader") };
        Ok(file_reader.inner.row_count())
    })
}

/// Creates a layout reader for a given file.
#[unsafe(no_mangle)]
pub extern "C-unwind" fn vx_layout_reader_create(
    file_reader: *mut vx_file_reader,
    error: *mut *mut vx_error,
) -> *mut vx_layout_reader {
    try_or(error, ptr::null_mut(), || {
        let file_reader = unsafe { file_reader.as_ref().vortex_expect("null file_reader") };
        let inner = file_reader.inner.layout_reader()?;

        Ok(Box::into_raw(Box::new(vx_layout_reader { inner })))
    })
}

#[unsafe(no_mangle)]
pub extern "C-unwind" fn vx_layout_reader_free(layout_reader: *mut vx_layout_reader) {
    if !layout_reader.is_null() {
        drop(unsafe { Box::from_raw(layout_reader) });
    }
}

/// Free the file and all associated resources.
///
/// This function will not automatically free any :c:func:`vx_array_iter` that were built from
/// this file.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_reader_free(file: *mut vx_file_reader) {
    drop(unsafe { Box::from_raw(file) });
}

fn make_object_store(
    url: &Url,
    property_keys: &[String],
    property_vals: &[String],
) -> VortexResult<Arc<dyn ObjectStore>> {
    let (scheme, _) =
        ObjectStoreScheme::parse(url).map_err(|error| VortexError::ObjectStore(error.into()))?;

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
                    log::warn!("Skipping unknown Amazon S3 config key: {}", key);
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
                    log::warn!("Skipping unknown Azure config key: {}", key);
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
                    log::warn!("Skipping unknown Google Cloud Storage config key: {}", key);
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
