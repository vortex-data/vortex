use std::str::FromStr;
use std::sync::Arc;

use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JObject, JString};
use jni::sys::jlong;
use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::azure::{AzureConfigKey, MicrosoftAzureBuilder};
use object_store::gcp::{GoogleCloudStorageBuilder, GoogleConfigKey};
use object_store::local::LocalFileSystem;
use object_store::{ObjectStore, ObjectStoreScheme};
use prost::Message;
use url::Url;
use vortex::aliases::hash_map::HashMap;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexResult, vortex_bail};
use vortex::expr::{Identity, deserialize_expr, select};
use vortex::file::{GenericVortexFile, VortexFile, VortexOpenOptions};
use vortex::io::ObjectStoreReadAt;
use vortex::proto::expr::Expr;
use vortex::stream::ArrayStreamExt;

use crate::array_stream::NativeArrayStream;
use crate::block_on;
use crate::errors::try_or_throw;

pub struct NativeFile {
    inner: VortexFile<GenericVortexFile<ObjectStoreReadAt>>,
}

impl NativeFile {
    pub fn new(file: VortexFile<GenericVortexFile<ObjectStoreReadAt>>) -> Box<Self> {
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
                let key_str: JString = key.into();
                let val_str: JString = val.into();
                let key_str = env.get_string(&key_str)?;
                let val_str = env.get_string(&val_str)?;
                properties.insert(key_str.into(), val_str.into());
            }
        }

        let (store, scheme) = make_object_store(&url, &properties)?;
        let reader = ObjectStoreReadAt::new(store.clone(), url.path().into(), Some(scheme));
        let open_file = block_on(
            "VortexOpenOptions.open()",
            VortexOpenOptions::file(reader).open(),
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
) -> jlong {
    // Return a new pointer to some native memory for the scan.
    let file = unsafe { NativeFile::from_ptr(pointer) };
    let mut scan_builder = file.inner.scan();

    try_or_throw(&mut env, |env| {
        // Apply the projection if provided
        if !project_cols.is_null() {
            let proj = env.get_list(&project_cols)?;
            if proj.size(env)? > 0 {
                // Convert the JList to a Vec<String>
                let mut projection: Vec<Arc<str>> = Vec::new();
                let mut iterator = proj.iter(env)?;
                while let Some(field) = iterator.next(env)? {
                    let field_name: String = env.get_string(&JString::from(field))?.into();
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

        // Canonicalize first, to avoid needing to pay decoding cost for every access.
        let scan = scan_builder.with_canonicalize(true).build()?;
        Ok(NativeArrayStream::new(scan.into_array_stream()?.boxed()).into_raw())
    })
}

fn make_object_store(
    url: &Url,
    properties: &HashMap<String, String>,
) -> VortexResult<(Arc<dyn ObjectStore>, ObjectStoreScheme)> {
    let (scheme, _) =
        ObjectStoreScheme::parse(url).map_err(|error| VortexError::ObjectStore(error.into()))?;

    // Configure extra properties on that scheme instead.
    match scheme {
        ObjectStoreScheme::Local => {
            log::trace!("using LocalFileSystem object store");
            Ok((Arc::new(LocalFileSystem::default()), scheme))
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

            let store = Arc::new(builder.build()?);
            Ok((store, scheme))
        }
        ObjectStoreScheme::MicrosoftAzure => {
            log::trace!("using MicrosoftAzure object store");

            let mut builder = MicrosoftAzureBuilder::new().with_url(url.to_string());
            for (key, val) in properties {
                if let Ok(config_key) = AzureConfigKey::from_str(key.as_str()) {
                    builder = builder.with_config(config_key, val);
                } else {
                    log::warn!("Skipping unknown Azure config key: {}", key);
                }
            }

            let store = Arc::new(builder.build()?);
            Ok((store, scheme))
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

            let store = Arc::new(builder.build()?);
            Ok((store, scheme))
        }
        store => {
            vortex_bail!("Unsupported store scheme: {store:?}");
        }
    }
}
