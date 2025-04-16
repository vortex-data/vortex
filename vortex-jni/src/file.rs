use std::str::FromStr;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JLongArray, JObject, JString, ReleaseMode};
use jni::sys::jlong;
use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::local::LocalFileSystem;
use object_store::{ClientOptions, ObjectStore, ObjectStoreScheme};
use prost::Message;
use url::Url;
use vortex::aliases::hash_map::HashMap;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{Identity, deserialize_expr, select};
use vortex::file::{VortexFile, VortexOpenOptions};
use vortex::proto::expr::Expr;

use crate::array_iter::NativeArrayIterator;
use crate::block_on;
use crate::errors::try_or_throw;

pub struct NativeFile {
    inner: VortexFile,
}

impl NativeFile {
    pub fn new(file: VortexFile) -> Box<Self> {
        Box::new(NativeFile { inner: file })
    }

    pub fn into_raw(self: Box<Self>) -> jlong {
        Box::into_raw(self) as jlong
    }

    pub unsafe fn from_raw(pointer: jlong) -> Box<Self> {
        unsafe { Box::from_raw(pointer as *mut NativeFile) }
    }

    #[allow(clippy::expect_used)]
    pub unsafe fn from_ptr<'a>(pointer: jlong) -> &'a Self {
        unsafe {
            (pointer as *const NativeFile)
                .as_ref()
                .expect("Pointer should never be null")
        }
    }
}

/// Open a file from a URL and options object. Returns a `long` representing a raw pointer
/// to a `NativeFile` object on the heap.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFileMethods_open(
    mut env: JNIEnv,
    _class: JClass,
    uri: JString,
    options: JObject,
) -> jlong {
    try_or_throw(&mut env, |env| {
        let file_path: String = env.get_string(&uri)?.into();

        let Ok(url) = Url::parse(&file_path) else {
            throw_runtime!("Invalid URL: {file_path}");
        };

        let mut properties: HashMap<String, String> = HashMap::new();

        if !options.is_null() {
            let opts = env.get_map(&options)?;
            let mut iterator = opts.iter(env)?;
            while let Some((key, val)) = iterator.next(env)? {
                let key = env.auto_local(key);
                let val = env.auto_local(val);
                let key_str = env.get_string(key.as_ref().into())?;
                let val_str = env.get_string(val.as_ref().into())?;
                properties.insert(key_str.into(), val_str.into());
            }
        }

        let start = std::time::Instant::now();
        let (store, _scheme) = make_object_store(&url, &properties)?;
        let duration = std::time::Instant::now().duration_since(start);
        log::debug!("make_object_store latency = {duration:?}");
        let open_file = block_on(
            "VortexOpenOptions.open()",
            VortexOpenOptions::file().open_object_store(&store, url.path()),
        )?;

        Ok(NativeFile::new(open_file).into_raw())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFileMethods_close(
    _env: JNIEnv,
    _class: JClass,
    pointer: jlong,
) {
    drop(unsafe { NativeFile::from_raw(pointer) });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFileMethods_dtype(
    _env: JNIEnv,
    _class: JClass,
    _pointer: jlong,
) -> jlong {
    let file = unsafe { NativeFile::from_ptr(_pointer) };
    file.inner.dtype() as *const DType as jlong
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFileMethods_scan(
    mut env: JNIEnv,
    _class: JClass,
    pointer: jlong,
    project_cols: JObject,
    predicate: JByteArray,
    row_indices: JLongArray,
) -> jlong {
    // Return a new pointer to some native memory for the scan.
    let file = unsafe { NativeFile::from_ptr(pointer) };
    let mut scan_builder = file.inner.scan().vortex_expect("scan builder");

    try_or_throw(&mut env, |env| {
        // Apply the projection if provided
        if !project_cols.is_null() {
            let proj = env.get_list(&project_cols)?;
            if proj.size(env)? > 0 {
                // Convert the JList to a Vec<String>
                let mut projection: Vec<Arc<str>> = Vec::new();
                let mut iterator = proj.iter(env)?;
                while let Some(field) = iterator.next(env)? {
                    let field = env.auto_local(field);

                    let field_name: String = env.get_string(field.as_ref().into())?.into();
                    projection.push(field_name.into());
                }
                let project_expr = select(projection, Identity::new_expr());
                scan_builder = scan_builder.with_projection(project_expr);
            }
        }

        // Apply predicate if one was provided
        if !predicate.is_null() {
            let proto_vec = env.convert_byte_array(predicate)?;
            let expr_proto =
                Expr::decode(proto_vec.as_slice()).map_err(VortexError::ProstDecodeError)?;
            let expr = deserialize_expr(&expr_proto)?;
            scan_builder = scan_builder.with_filter(expr);
        }

        // Apply row indices if provided
        if !row_indices.is_null() {
            let indices = unsafe { env.get_array_elements(&row_indices, ReleaseMode::NoCopyBack) }?;
            let indices_buffer: Buffer<u64> = indices
                .iter()
                .map(|long: &i64| u64::try_from(*long))
                .collect::<Result<Buffer<u64>, _>>()
                .map_err(|_| vortex_err!("row indices can not be negative"))?;
            scan_builder = scan_builder.with_row_indices(indices_buffer);
        }

        // Canonicalize first, to avoid needing to pay decoding cost for every access.
        let scan = scan_builder.with_canonicalize(true);

        Ok(NativeArrayIterator::new(Box::new(scan.into_array_iter()?)).into_raw())
    })
}

fn make_object_store(
    url: &Url,
    properties: &HashMap<String, String>,
) -> VortexResult<(Arc<dyn ObjectStore>, ObjectStoreScheme)> {
    static OBJECT_STORES: LazyLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    let (scheme, _) =
        ObjectStoreScheme::parse(url).map_err(|error| VortexError::ObjectStore(error.into()))?;

    let cache_key = url_cache_key(url);

    {
        if let Some(cached) = OBJECT_STORES.lock().vortex_expect("poison").get(&cache_key) {
            return Ok((cached.clone(), scheme));
        }
        // guard dropped at close of scope
    }

    // Configure extra properties on that scheme instead.
    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => {
            log::trace!("using LocalFileSystem object store");
            Arc::new(LocalFileSystem::default())
        }
        ObjectStoreScheme::AmazonS3 => {
            log::trace!("using AmazonS3 object store");
            let mut builder = AmazonS3Builder::new().with_url(url.to_string());
            for (key, val) in properties {
                if let Ok(config_key) = AmazonS3ConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    log::warn!("Skipping unknown Amazon S3 config key: {}", key);
                }
            }

            Arc::new(builder.build()?)
        }
        ObjectStoreScheme::MicrosoftAzure => {
            log::trace!("using MicrosoftAzure object store");

            // NOTE(aduffy): anecdotally Azure often times out after 30 seconds, this bumps us up
            //  to avoid that.
            let client_opts = ClientOptions::new().with_timeout(Duration::from_secs(120));
            let mut builder = MicrosoftAzureBuilder::new()
                .with_url(url.to_string())
                .with_client_options(client_opts);
            for (key, val) in properties {
                if let Ok(config_key) = AzureConfigKey::from_str(key.as_str()) {
                    log::warn!("setting azure config {key:?} = {val}");
                    builder = builder.with_config(config_key, val);
                } else {
                    log::warn!("Skipping unknown Azure config key: {}", key);
                }
            }

            Arc::new(builder.build()?)
        }
        ObjectStoreScheme::GoogleCloudStorage => {
            log::trace!("using GoogleCloudStorage object store");

            let mut builder = GoogleCloudStorageBuilder::new().with_url(url.to_string());
            for (key, val) in properties {
                if let Ok(config_key) = GoogleConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    log::warn!("Skipping unknown Google Cloud Storage config key: {}", key);
                }
            }

            Arc::new(builder.build()?)
        }
        store => {
            vortex_bail!("Unsupported store scheme: {store:?}");
        }
    };

    {
        OBJECT_STORES
            .lock()
            .vortex_expect("poison")
            .insert(cache_key, store.clone());
        // Guard dropped at close of scope.
    }

    Ok((store, scheme))
}

fn url_cache_key(url: &Url) -> String {
    format!(
        "{}://{}",
        url.scheme(),
        &url[url::Position::BeforeHost..url::Position::AfterPort],
    )
}
