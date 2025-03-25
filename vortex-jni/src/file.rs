use std::str::FromStr;
use std::sync::Arc;

use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JList, JMap, JObject, JString};
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
use vortex::error::{VortexError, VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex::expr::{Identity, deserialize_expr, select};
use vortex::file::{GenericVortexFile, VortexFile, VortexOpenOptions};
use vortex::io::ObjectStoreReadAt;
use vortex::proto::expr::Expr;
use vortex::stream::ArrayStreamExt;

use crate::TOKIO_RUNTIME;
use crate::array_stream::NativeArrayStream;
use crate::errors::Throwable;

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
    let file_path: String = env
        .get_string(&uri)
        .expect("Failed to convert JString")
        .into();

    let Ok(url) = Url::parse(&file_path) else {
        vortex_err!("Invalid URL: {file_path}").throw_illegal_argument(&mut env);
        return 0;
    };

    // Convert the options map to a hashmap
    let mut properties: HashMap<String, String> = HashMap::new();
    if !env
        .is_same_object(&options, JObject::null())
        .expect("same_object")
    {
        env.with_local_frame(1_024, |env| {
            let opts = JMap::from_env(env, &options).expect("JMap.from_env");
            let mut iterator = opts.iter(env).expect("JMap.iter");
            while let Some((key, val)) = iterator.next(env).expect("JMap.iter") {
                let key_str = env.get_string((&key).into()).expect("get_string");
                let val_str = env.get_string((&val).into()).expect("get_string");

                properties.insert(key_str.into(), val_str.into());
            }

            Ok::<(), jni::errors::Error>(())
        })
        .expect("Failed to read properties");
    }

    match make_object_store(&url, &properties) {
        Ok((store, scheme)) => {
            let reader = ObjectStoreReadAt::new(store.clone(), url.path().into(), Some(scheme));
            let open_file = TOKIO_RUNTIME.block_on(VortexOpenOptions::file(reader).open());
            match open_file {
                Ok(open_file) => NativeFile::new(open_file).into_raw(),
                Err(err) => {
                    err.throw_runtime(&mut env, "open_file");
                    0
                }
            }
        }
        Err(err) => {
            err.throw_illegal_argument(&mut env);
            0
        }
    }
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

    // Apply the projection if provided
    if !env
        .is_same_object(&project_cols, JObject::null())
        .expect("same_object")
        && JList::from_env(&mut env, &project_cols)
            .expect("JList")
            .size(&mut env)
            .expect("JList.size")
            > 0
    {
        // Convert the JList to a Vec<String>
        let mut projection: Vec<Arc<str>> = Vec::new();
        env.with_local_frame(1_024, |env| {
            let proj = JList::from_env(env, &project_cols).expect("JList.from_env");
            let mut iterator = proj.iter(env).expect("project_cols.iter");
            while let Some(field) = iterator.next(env).expect("project_cols.next") {
                let field_name: String = env
                    .get_string(&JString::from(field))
                    .expect("Failed to convert JString")
                    .into();
                projection.push(field_name.into());
            }

            Ok::<(), jni::errors::Error>(())
        })
        .expect("Failed to read projection columns");
        let project_expr = select(projection, Identity::new_expr());
        scan_builder = scan_builder.with_projection(project_expr);
    }

    // Apply predicate if one was provided
    if !env
        .is_same_object(&predicate, JObject::null())
        .expect("same_object")
    {
        let proto_vec = env
            .convert_byte_array(predicate)
            .expect("convert byte array");
        let expr_proto =
            Expr::decode(proto_vec.as_slice()).vortex_expect("decode filter expression");
        match deserialize_expr(&expr_proto) {
            Ok(expr) => {
                scan_builder = scan_builder.with_filter(expr);
            }
            Err(err) => {
                err.throw_illegal_argument(&mut env);
                return -1;
            }
        }
    }

    // Canonicalize first, to avoid needing to pay decoding cost for every access.
    scan_builder = scan_builder.with_canonicalize(true);

    // build and wrap scan with native object
    match scan_builder.build() {
        Ok(scan) => NativeArrayStream::new(
            scan.into_array_stream()
                .vortex_expect("into_array_stream")
                .boxed(),
        )
        .into_raw(),

        Err(err) => {
            err.throw_runtime(&mut env, "scan_builder");
            -1
        }
    }
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
