// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Small utility JNI entry points for listing, deleting, and reading the user-defined
//! metadata of Vortex files.

use std::sync::Arc;

use futures::StreamExt;
use jni::EnvUnowned;
use jni::objects::JByteArray;
use jni::objects::JClass;
use jni::objects::JMap;
use jni::objects::JObject;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::sys::jlong;
use jni::sys::jobject;
use object_store::path::Path;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::multi::parse_uri_or_path;
use vortex::io::VortexReadAt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;

use crate::RUNTIME;
use crate::errors::try_or_throw;
use crate::io::java_readable;
use crate::object_store::object_store_fs;
use crate::session::session_ref;

/// Extract a Java `Map<String, String>` into a Rust [`HashMap`].
pub(crate) fn extract_properties(
    env: &mut jni::Env,
    options: &JObject,
) -> Result<HashMap<String, String>, crate::errors::JNIError> {
    let mut properties = HashMap::new();
    if !options.is_null() {
        let options_ref = env.new_local_ref(options)?;
        let opts = env.cast_local::<JMap>(options_ref)?;
        let mut iterator = opts.iter(env)?;
        while let Some(entry) = iterator.next(env)? {
            let key_obj = entry.key(env)?;
            let val_obj = entry.value(env)?;
            let key_str = env.cast_local::<JString>(key_obj)?;
            let val_str = env.cast_local::<JString>(val_obj)?;
            properties.insert(key_str.try_to_string(env)?, val_str.try_to_string(env)?);
        }
    }
    Ok(properties)
}

/// Extract a Java `Map<String, byte[]>` of user-defined metadata segments.
///
/// Values are opaque bytes, so they are copied out of the Java arrays rather than
/// interpreted. A null map yields an empty set of segments.
pub(crate) fn extract_metadata(
    env: &mut jni::Env,
    metadata: &JObject,
) -> Result<HashMap<String, ByteBuffer>, crate::errors::JNIError> {
    let mut segments = HashMap::new();
    if metadata.is_null() {
        return Ok(segments);
    }

    let metadata_ref = env.new_local_ref(metadata)?;
    let map = env.cast_local::<JMap>(metadata_ref)?;
    let mut iterator = map.iter(env)?;
    while let Some(entry) = iterator.next(env)? {
        let key_obj = entry.key(env)?;
        let val_obj = entry.value(env)?;
        let key = env.cast_local::<JString>(key_obj)?.try_to_string(env)?;
        if val_obj.is_null() {
            return Err(vortex_err!("null metadata value for key '{key}'").into());
        }
        let bytes = env.cast_local::<JByteArray>(val_obj)?;
        segments.insert(key, ByteBuffer::from(env.convert_byte_array(&bytes)?));
    }

    Ok(segments)
}

/// Build a `java.util.HashMap<String, byte[]>` from user-defined metadata segments.
fn metadata_to_java(
    env: &mut jni::Env,
    segments: Vec<(String, ByteBuffer)>,
) -> Result<jobject, crate::errors::JNIError> {
    let map = env.new_object(
        jni::jni_str!("java/util/HashMap"),
        jni::jni_sig!("()V"),
        &[],
    )?;
    let raw = map.as_raw();
    let map = env.cast_local::<JMap>(map)?;
    for (key, value) in segments {
        let key = env.new_string(key)?;
        let value = env.byte_array_from_slice(value.as_slice())?;
        // A file may hold several segments and each entry costs two local refs, so release
        // them as we go rather than relying on the frame's guaranteed capacity.
        if let Some(previous) = map.put(env, key.as_ref(), value.as_ref())? {
            env.delete_local_ref(previous);
        }
        env.delete_local_ref(value);
        env.delete_local_ref(key);
    }

    Ok(raw)
}

/// Open a Vortex file with metadata resolution enabled and collect its metadata segments.
fn read_metadata_segments(
    session: &VortexSession,
    source: Arc<dyn VortexReadAt>,
) -> VortexResult<Vec<(String, ByteBuffer)>> {
    RUNTIME.block_on(async move {
        let file = session
            .open_options()
            .include_metadata()
            .open(source)
            .await?;
        Ok(file
            .metadata_segments()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect())
    })
}

/// Read the user-defined metadata segments of the Vortex file at `uri`.
///
/// Returns a `java.util.HashMap<String, byte[]>`, empty when the file carries no metadata.
/// Metadata lives in its own segments, so this may issue a read beyond the file tail.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFiles_readMetadata(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    uri: JString,
    options: JObject,
) -> jobject {
    try_or_throw(&mut env, |env| {
        let session = unsafe { session_ref(session_ptr) };
        let uri: String = uri.try_to_string(env)?;
        let url = parse_uri_or_path(&uri)?;
        let properties = extract_properties(env, &options)?;

        let fs = object_store_fs(&url, &properties, session.handle())?;
        // `FileSystem` keys are literal, already-decoded paths, so decode as `listFiles` does.
        let path = Path::from_url_path(url.path())
            .map_err(|_| vortex_err!("cannot parse uri as object_store Path"))?
            .to_string();
        let source = RUNTIME.block_on(async move { fs.open_read(&path).await })?;

        let segments = read_metadata_segments(session, source)?;
        metadata_to_java(env, segments)
    })
}

/// Read the user-defined metadata segments of a Vortex file through a caller-provided
/// `dev.vortex.io.NativeReadable`, so no native storage client is created.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFiles_readMetadataFromReadable(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    readable: JObject,
    length: jlong,
) -> jobject {
    try_or_throw(&mut env, |env| {
        let session = unsafe { session_ref(session_ptr) };
        if readable.is_null() {
            throw_runtime!("null readable");
        }
        let length =
            u64::try_from(length).map_err(|_| vortex_err!("negative readable length: {length}"))?;

        let vm = env.get_java_vm()?;
        let readable = Arc::new(env.new_global_ref(&readable)?);
        let source = java_readable(vm, readable, length, session.handle());

        let segments = read_metadata_segments(session, source)?;
        metadata_to_java(env, segments)
    })
}

/// List Vortex files under the given URI prefix. Returns a `java.util.ArrayList<String>`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFiles_listFiles(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    path: JString,
    options: JObject,
) -> jobject {
    try_or_throw(&mut env, |env| {
        let session = unsafe { session_ref(session_ptr) };
        let root_path: String = path.try_to_string(env)?;
        let url = parse_uri_or_path(&root_path)?;

        let properties = extract_properties(env, &options)?;

        let fs = object_store_fs(&url, &properties, session.handle())?;
        let prefix = Path::from_url_path(url.path())
            .map_err(|_| vortex_err!("cannot parse root_path as object_store Path"))?;

        let mut stream = fs.list(prefix.as_ref());

        let paths_vec = RUNTIME.block_on(async move {
            let mut paths = Vec::new();
            while let Some(file) = stream.next().await {
                let mut found = url.clone();
                found.set_path(&file?.path);
                paths.push(found.to_string());
            }

            VortexResult::Ok(paths)
        })?;

        let paths_result = env.new_object(
            jni::jni_str!("java/util/ArrayList"),
            jni::jni_sig!("()V"),
            &[],
        )?;
        let raw = paths_result.as_raw();
        let paths_list = env.cast_local::<jni::objects::JList>(paths_result)?;
        for path in paths_vec.into_iter() {
            let path_string = env.new_string(path)?;
            paths_list.add(env, path_string.as_ref())?;
        }

        Ok(raw)
    })
}

/// Delete Vortex files at the given URIs.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeFiles_delete(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    uris: JObjectArray,
    options: JObject,
) {
    try_or_throw(&mut env, |env| {
        let session = unsafe { session_ref(session_ptr) };
        let mut delete_uris = Vec::new();

        let num_uris = uris.len(env)?;
        for idx in 0..num_uris {
            let uri = uris.get_element(env, idx)?;
            let uri_str = env.cast_local::<JString>(uri)?;
            delete_uris.push(uri_str.try_to_string(env)?);
        }

        if delete_uris.is_empty() {
            return Ok(());
        }

        let store_url = parse_uri_or_path(&delete_uris[0])?;

        let properties = extract_properties(env, &options)?;

        let fs = object_store_fs(&store_url, &properties, session.handle())?;

        RUNTIME.block_on(async {
            for uri in delete_uris {
                let url = parse_uri_or_path(&uri)?;
                fs.delete(url.path()).await?;
            }
            VortexResult::Ok(())
        })?;

        Ok(())
    });
}
