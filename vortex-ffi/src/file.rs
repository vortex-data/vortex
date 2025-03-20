//! FFI interface for Vortex File I/O.

use std::ffi::{CStr, c_char, c_int};
use std::slice;
use std::str::FromStr;
use std::sync::Arc;

use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::local::LocalFileSystem;
use object_store::{ObjectStore, ObjectStoreScheme};
use prost::Message;
use url::Url;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_bail};
use vortex::expr::{Identity, deserialize_expr, select};
use vortex::file::{GenericVortexFile, VortexFile, VortexOpenOptions};
use vortex::io::ObjectStoreReadAt;
use vortex::proto::expr::Expr;

use crate::stream::{FFIArrayStream, FFIArrayStreamInner};
use crate::{RUNTIME, to_string, to_string_vec};

pub struct FFIFile {
    pub(crate) inner: VortexFile<GenericVortexFile<ObjectStoreReadAt>>,
}

/// Options supplied for opening a file.
#[repr(C)]
pub struct FileOpenOptions {
    /// URI for opening the file.
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
}

/// Open a file at the given path on the file system.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_open(options: *const FileOpenOptions) -> *mut FFIFile {
    assert!(!options.is_null(), "File_open: null options");

    let options = &*options;

    assert!(!options.uri.is_null(), "File_open: null uri");
    let uri = CStr::from_ptr(options.uri).to_string_lossy();
    let uri: Url = uri.parse().vortex_expect("File_open: parse uri");

    let prop_keys = to_string_vec(options.property_keys, options.property_len);
    let prop_vals = to_string_vec(options.property_vals, options.property_len);

    let object_store = make_object_store(&uri, &prop_keys, &prop_vals)
        .vortex_expect("File_open: make_object_store");
    let read_at = ObjectStoreReadAt::new(object_store, uri.path().into(), None);

    let result = RUNTIME.block_on(async move { VortexOpenOptions::file(read_at).open().await });

    let file = result.vortex_expect("open");
    let ffi_file = FFIFile { inner: file };

    Box::into_raw(Box::new(ffi_file))
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
        num_rows: (*file).inner.row_count(),
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn FileStatistics_free(stat: *mut FileStatistics) {
    drop(Box::from_raw(stat));
}

/// Get a readonly pointer to the DType of the data inside of the file.
///
/// The pointer's lifetime is tied to the lifetime of the underlying file, so it should not be
/// dereferenced after the file has been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_dtype(file: *const FFIFile) -> *const DType {
    assert!(!file.is_null(), "File_dtype: file is null");

    let file = &*file;
    file.inner.dtype()
}

/// Build a new Scan that will stream batches of `FFIArray` from the file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn File_scan(
    file: *const FFIFile,
    opts: *const FileScanOptions,
) -> *mut FFIArrayStream {
    let file = unsafe { &*file };
    let mut stream = file.inner.scan();

    if !opts.is_null() {
        let opts = &*opts;
        let mut field_names = Vec::new();
        for i in 0..opts.projection_len {
            let col_name = unsafe { *opts.projection.offset(i as isize) };
            let col_name: Arc<str> = to_string(col_name).into();
            field_names.push(col_name);
        }
        let expr_str = opts.filter_expression;
        if !expr_str.is_null() && opts.filter_expression_len > 0 {
            let bytes = unsafe {
                slice::from_raw_parts(expr_str as *const u8, opts.filter_expression_len as usize)
            };

            // Decode the protobuf message
            let expr_proto = Expr::decode(bytes).vortex_expect("decode filter expression");
            let expr = deserialize_expr(&expr_proto).vortex_expect("deserialize filter expression");
            stream = stream.with_filter(expr)
        }
        stream = stream.with_projection(select(field_names, Identity::new_expr()));
    }

    let stream = stream
        .into_array_stream()
        .vortex_expect("into_array_stream");

    let inner = Some(Box::new(FFIArrayStreamInner {
        stream: Box::pin(stream),
    }));

    Box::into_raw(Box::new(FFIArrayStream {
        inner,
        current: None,
    }))
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
