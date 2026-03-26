// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::StreamExt;
use jni::JNIEnv;
use jni::objects::JByteArray;
use jni::objects::JClass;
use jni::objects::JLongArray;
use jni::objects::JObject;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::objects::ReleaseMode;
use jni::sys::jlong;
use jni::sys::jobject;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path;
use prost::Message;
use url::Url;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::error::VortexError;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::proto::expr as pb;
use vortex::utils::aliases::hash_map::HashMap;

use crate::RUNTIME;
use crate::SESSION;
use crate::TOKIO_RUNTIME;
use crate::array_iter::NativeArrayIterator;
use crate::errors::try_or_throw;
use crate::object_store::make_object_store;

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

    #[expect(
        clippy::expect_used,
        reason = "JNI contract guarantees non-null pointer"
    )]
    pub unsafe fn from_ptr<'a>(pointer: jlong) -> &'a Self {
        unsafe {
            (pointer as *const NativeFile)
                .as_ref()
                .expect("Pointer should never be null")
        }
    }
}

/// List Vortex files underneath a root path.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFileMethods_listVortexFiles<'local>(
    mut env: JNIEnv,
    _class: JClass,
    path: JString<'local>,
    options: JObject<'local>,
) -> jobject {
    try_or_throw(&mut env, |env| {
        let root_path: String = env
            .get_string(&path)
            .map_err(|e| vortex_err!("get_string error: {e}"))?
            .into();

        let Ok(url) = Url::parse(&root_path) else {
            throw_runtime!("Invalid URL: {root_path}");
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

        let (store, _) = make_object_store(&url, &properties)?;
        let prefix = Path::from_url_path(url.path())
            .map_err(|_| vortex_err!("Cannot parse root_path as object_store Path"))?;

        let mut stream = store.list(Some(&prefix));

        let paths_vec = TOKIO_RUNTIME.block_on(async move {
            let mut paths = Vec::new();
            while let Some(file) = stream.next().await {
                let file = file.map_err(VortexError::from)?;
                if file.location.as_ref().ends_with(".vortex") {
                    let mut found = url.clone();
                    found.set_path(file.location.as_ref());
                    paths.push(found.to_string());
                }
            }

            VortexResult::Ok(paths)
        })?;

        let paths_result = env.new_object("java/util/ArrayList", "()V", &[])?;
        let paths_list = env.get_list(&paths_result)?;
        for path in paths_vec.into_iter() {
            let path_string = env.new_string(path)?;
            paths_list.add(env, &path_string)?;
        }

        Ok(paths_result.into_raw())
    })
}

/// Delete files from the target
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFileMethods_delete<'local>(
    mut env: JNIEnv,
    _class: JClass,
    uris: JObjectArray<'local>,
    options: JObject<'local>,
) {
    try_or_throw(&mut env, |env| {
        let mut delete_uris = Vec::new();

        let num_uris = env.get_array_length(&uris)?;
        for idx in 0..num_uris {
            let uri = env.get_object_array_element(&uris, idx)?;
            delete_uris.push(
                env.get_string(&JString::from(uri))?
                    .to_string_lossy()
                    .to_string(),
            );
        }

        // Nothing to delete
        if delete_uris.is_empty() {
            return Ok(());
        }

        // Pick the first URL to use for building the client
        let store_url = Url::parse(&delete_uris[0]).map_err(|e| vortex_err!(External: e))?;

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

        let (store, _) = make_object_store(&store_url, &properties)?;

        for uri in delete_uris {
            let url = Url::parse(&uri).map_err(|e| vortex_err!(External: e))?;
            // TODO(aduffy): block on all of them
            TOKIO_RUNTIME
                .block_on(
                    store.delete(
                        &Path::from_url_path(url.path())
                            .map_err(|_| vortex_err!("invalid path for url {url}"))?,
                    ),
                )
                .map_err(VortexError::from)?;
        }

        Ok(())
    });
}

/// Open a file from a URL and options object. Returns a `long` representing a raw pointer
/// to a `NativeFile` object on the heap.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFileMethods_open<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass,
    uri: JString<'local>,
    options: JObject<'local>,
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

        let (store, _scheme) = make_object_store(&url, &properties)?;
        let open_file =
            TOKIO_RUNTIME.block_on(SESSION.open_options().open_object_store(&store, url.path()))?;

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
pub extern "system" fn Java_dev_vortex_jni_NativeFileMethods_rowCount(
    mut env: JNIEnv,
    _class: JClass,
    pointer: jlong,
) -> jlong {
    let file = unsafe { NativeFile::from_ptr(pointer) };
    try_or_throw(&mut env, |_| {
        let row_count = jlong::try_from(file.inner.row_count())
            .map_err(|_| vortex_err!("Overflow converting row count to jlong"))?;
        Ok(row_count)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFileMethods_scan(
    mut env: JNIEnv,
    _class: JClass,
    pointer: jlong,
    project_cols: JObject,
    predicate: JByteArray,
    row_range: JLongArray,
    row_indices: JLongArray,
) -> jlong {
    // Return a new pointer to some native memory for the scan.
    let file = unsafe { NativeFile::from_ptr(pointer) };
    // TODO: propagate this error up instead of expecting
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
                let project_expr = select(projection, root());
                scan_builder = scan_builder.with_projection(project_expr);
            }
        }

        // Apply predicate if one was provided
        if !predicate.is_null() {
            let proto_vec = env.convert_byte_array(predicate)?;
            let expr_proto = pb::Expr::decode(proto_vec.as_slice()).map_err(VortexError::from)?;
            let expr = Expression::from_proto(&expr_proto, &SESSION)?;
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

        if !row_range.is_null() {
            let indices = unsafe { env.get_array_elements(&row_range, ReleaseMode::NoCopyBack) }?;
            let start_idx =
                u64::try_from(indices[0]).map_err(|_| vortex_err!("i64 row_index overflow"))?;
            let end_idx =
                u64::try_from(indices[1]).map_err(|_| vortex_err!("i64 row_index overflow"))?;
            scan_builder = scan_builder.with_row_range(start_idx..end_idx);
        }

        Ok(NativeArrayIterator::new(Box::new(scan_builder.into_array_iter(&*RUNTIME)?)).into_raw())
    })
}
