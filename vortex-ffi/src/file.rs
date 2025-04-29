//! FFI interface for Vortex File I/O.

use std::ffi::{CStr, c_char, c_int};
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
use tokio::fs::File;
use url::Url;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{Identity, deserialize_expr, select};
use vortex::file::scan::SplitBy;
use vortex::file::{VortexFile, VortexOpenOptions, VortexWriteOptions};
use vortex::proto::expr::Expr;
use vortex::stream::ArrayStreamArrayExt;

use crate::array::vx_array;
use crate::error::{try_or, vx_error};
use crate::stream::{ArrayStreamInner, vx_array_stream};
use crate::{RUNTIME, to_string, to_string_vec};

/// A file reader that can be used to read from a file.
#[allow(non_camel_case_types)]
pub struct vx_file_reader {
    pub(crate) inner: VortexFile,
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
    pub projection_len: c_int,
    // Serialized expressions for pushdown
    pub filter_expression: *const c_char,
    // The len in bytes of the filter expression
    pub filter_expression_len: c_int,

    /// Splits the file into chunks of this size, if zero then we use the write layout.
    pub split_by_row_count: c_int,
}

/// Open a file at the given path on the file system.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_open_reader(
    options: *const vx_file_open_options,
    error: *mut *mut vx_error,
) -> *mut vx_file_reader {
    try_or(error, ptr::null_mut(), || {
        {
            let options = unsafe {
                options
                    .as_ref()
                    .ok_or_else(|| vortex_err!("null options"))?
            };

            if options.uri.is_null() {
                vortex_bail!("null uri")
            }
            let uri = CStr::from_ptr(options.uri).to_string_lossy();
            let uri: Url = uri.parse().vortex_expect("File_open: parse uri");

            let prop_keys = to_string_vec(options.property_keys, options.property_len);
            let prop_vals = to_string_vec(options.property_vals, options.property_len);

            let object_store = make_object_store(&uri, &prop_keys, &prop_vals)?;

            // TODO(joe): replace with futures::executor::block_on, currently vortex-file has a hidden
            // tokio dep
            let inner = RUNTIME.with(|r| {
                r.block_on(async move {
                    VortexOpenOptions::file()
                        .open_object_store(&object_store, uri.path())
                        .await
                })
            })?;
            Ok(Box::into_raw(Box::new(vx_file_reader { inner })))
        }
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
        let path = CStr::from_ptr(path).to_str()?;

        RUNTIME.with(|r| {
            r.block_on(async move {
                VortexWriteOptions::default()
                    .write(
                        &mut File::create(path).await?,
                        array.inner.to_array_stream(),
                    )
                    .await?;
                Ok(())
            })
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
        num_rows: file
            .as_ref()
            .vortex_expect("null file ptr")
            .inner
            .row_count(),
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_statistics_free(stat: *mut vx_file_statistics) {
    assert!(!stat.is_null());
    drop(Box::from_raw(stat));
}

/// Get a readonly pointer to the DType of the data inside of the file.
///
/// The pointer's lifetime is tied to the lifetime of the underlying file, so it should not be
/// dereferenced after the file has been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_dtype(file: *const vx_file_reader) -> *const DType {
    file.as_ref().vortex_expect("null file").inner.dtype()
}

/// Build a new `vx_array_stream` that return a series of `vx_array`s scan over a `vx_file`.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_scan(
    file: *const vx_file_reader,
    opts: *const vx_file_scan_options,
    error: *mut *mut vx_error,
) -> *mut vx_array_stream {
    try_or(error, ptr::null_mut(), || {
        let file = unsafe { file.as_ref().vortex_expect("null file") };
        let mut stream = file.inner.scan().vortex_expect("create scan");

        if let Some(opts) = opts.as_ref() {
            let mut field_names = Vec::new();
            for i in 0..opts.projection_len {
                let col_name = unsafe { *opts.projection.offset(i as isize) };
                let col_name: Arc<str> = to_string(col_name).into();
                field_names.push(col_name);
            }
            let expr_str = opts.filter_expression;
            if !expr_str.is_null() && opts.filter_expression_len > 0 {
                let bytes = unsafe {
                    slice::from_raw_parts(
                        expr_str as *const u8,
                        opts.filter_expression_len as usize,
                    )
                };

                // Decode the protobuf message
                let expr_proto = Expr::decode(bytes)?;
                let expr = deserialize_expr(&expr_proto)
                    .map_err(|e| e.with_context("deserializing expr"))?;
                stream = stream.with_filter(expr)
            }
            if opts.split_by_row_count > 0 {
                stream = stream.with_split_by(SplitBy::RowCount(opts.split_by_row_count as usize));
            }

            stream = stream.with_projection(select(field_names, Identity::new_expr()));
        }

        let stream = stream.into_array_stream()?;

        let inner = Some(Box::new(ArrayStreamInner {
            stream: Box::pin(stream),
        }));

        Ok(Box::into_raw(Box::new(vx_array_stream { inner })))
    })
}

/// Free the file and all associated resources.
///
/// This function will not automatically free any :c:func:`vx_array_stream` that were built from
/// this file.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_reader_free(file: *mut vx_file_reader) {
    drop(Box::from_raw(file));
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
