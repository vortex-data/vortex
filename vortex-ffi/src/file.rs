//! FFI interface for Vortex File I/O.

use std::ffi::{CStr, c_char, c_int};
use std::str::FromStr;
use std::sync::Arc;
use std::{ptr, slice};

use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::local::LocalFileSystem;
use object_store::{ObjectStore, ObjectStoreScheme};
use prost::Message;
use url::Url;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{Identity, deserialize_expr, select};
use vortex::file::scan::SplitBy;
use vortex::file::{VortexFile, VortexOpenOptions, VortexWriteOptions};
use vortex::proto::expr::Expr;
use vortex::stream::ArrayStreamArrayExt;

use crate::array::FFIArray;
use crate::error::{FFIError, try_or};
use crate::stream::{FFIArrayStream, FFIArrayStreamInner};
use crate::{RUNTIME, to_string, to_string_vec};

pub struct FFIFile {
    pub(crate) inner: VortexFile,
}

/// Options supplied for opening a file.
#[repr(C)]
pub struct FileCreateOptions {
    /// path of the file to be created.
    /// This must be a valid URI, even the files (file:///path/to/file)
    pub path: *const c_char,
}

/// Options supplied for opening a file.
#[repr(C)]
pub struct FileOpenOptions {
    /// URI for opening the file.
    /// This must be a valid URI, even the files (file:///path/to/file)
    pub uri: *const c_char,
    /// Additional configuration for the file source (e.g. "s3.accessKey").
    /// This may be null, in which case it is treated as empty.
    pub property_keys: *const *const c_char,
    /// Additional configuration values for the file source (e.g. S3 credentials).
    pub property_vals: *const *const c_char,
    /// Number of properties in `property_keys` and `property_vals`.
    pub property_len: c_int,
}

/// Scan options provided by an FFI client calling the `File_scan` function.
#[repr(C)]
pub struct FileScanOptions {
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
pub unsafe extern "C" fn File_open(
    options: *const FileOpenOptions,
    error: *mut *mut FFIError,
) -> *mut FFIFile {
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
            let result = RUNTIME.block_on(async move {
                VortexOpenOptions::file()
                    .open_object_store(&object_store, uri.path())
                    .await
            });

            let file = result?;
            let ffi_file = FFIFile { inner: file };
            Ok(Box::into_raw(Box::new(ffi_file)))
        }
    })
}

/// This function creates a new file by writing the ffi array to the path in the options args.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_create_and_write_array(
    options: *const FileCreateOptions,
    ffi_array: *mut FFIArray,
    error: *mut *mut FFIError,
) {
    try_or(error, (), || {
        let options = options.as_ref().vortex_expect("null options");
        assert!(!options.path.is_null(), "null path");

        let path = CStr::from_ptr(options.path).to_string_lossy();
        let array = unsafe { ffi_array.as_ref().vortex_expect("null array") };

        RUNTIME.block_on(async move {
            let file = tokio::fs::File::create(path.to_string()).await?;
            let file = VortexWriteOptions::default()
                .write(file, array.inner.to_array_stream())
                .await?;

            file.sync_all().await?;
            Ok(())
        })
    });
}

/// Whole file statistics.
#[repr(C)]
pub struct FileStatistics {
    /// The exact number of rows in the file.
    pub num_rows: u64,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_statistics(file: *mut FFIFile) -> *mut FileStatistics {
    Box::into_raw(Box::new(FileStatistics {
        num_rows: file
            .as_ref()
            .vortex_expect("null file ptr")
            .inner
            .row_count(),
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn FileStatistics_free(stat: *mut FileStatistics) {
    assert!(!stat.is_null());
    drop(Box::from_raw(stat));
}

/// Get a readonly pointer to the DType of the data inside of the file.
///
/// The pointer's lifetime is tied to the lifetime of the underlying file, so it should not be
/// dereferenced after the file has been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_dtype(file: *const FFIFile) -> *const DType {
    file.as_ref().vortex_expect("null file").inner.dtype()
}

/// Build a new `FFIArrayStream` that return a series of `FFIArray`s scan over a `FFIFile`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_scan(
    file: *const FFIFile,
    opts: *const FileScanOptions,
    error: *mut *mut FFIError,
) -> *mut FFIArrayStream {
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

        let inner = Some(Box::new(FFIArrayStreamInner {
            stream: Box::pin(stream),
        }));

        Ok(Box::into_raw(Box::new(FFIArrayStream {
            inner,
            current: None,
        })))
    })
}

/// Free the file and all associated resources.
///
/// This function will not automatically free any `FFIArrayStream`s that were built from this
/// file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_free(file: *mut FFIFile) {
    drop(Box::from_raw(file));
}

fn make_object_store(
    url: &Url,
    property_keys: &[String],
    property_vals: &[String],
) -> VortexResult<Arc<dyn ObjectStore>> {
    let (scheme, _) =
        ObjectStoreScheme::parse(url).map_err(|error| VortexError::ObjectStore(error.into()))?;

    // Configure extra properties on that scheme instead.
    match scheme {
        ObjectStoreScheme::Local => {
            log::trace!("using LocalFileSystem object store");
            Ok(Arc::new(LocalFileSystem::default()))
        }
        ObjectStoreScheme::AmazonS3 => {
            log::trace!("using AmazonS3 object store");
            let mut builder = AmazonS3Builder::new().with_url(url.to_string());
            for (key, val) in property_keys.iter().zip(property_vals.iter()) {
                if let Ok(config_key) = AmazonS3ConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    log::warn!("Skipping unknown Amazon S3 config key: {}", key);
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
