// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Small utility JNI entry points for listing and deleting Vortex files via an object store.

use futures::StreamExt;
use jni::EnvUnowned;
use jni::objects::JClass;
use jni::objects::JMap;
use jni::objects::JObject;
use jni::objects::JObjectArray;
use jni::objects::JString;
use jni::sys::jlong;
use jni::sys::jobject;
use object_store::path::Path;
use url::Url;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::utils::aliases::hash_map::HashMap;

use crate::RUNTIME;
use crate::errors::try_or_throw;
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

        let Ok(url) = Url::parse(&root_path) else {
            throw_runtime!("invalid URL: {root_path}");
        };

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

        let store_url = Url::parse(&delete_uris[0]).map_err(|e| vortex_err!(External: e))?;

        let properties = extract_properties(env, &options)?;

        let fs = object_store_fs(&store_url, &properties, session.handle())?;

        RUNTIME.block_on(async {
            for uri in delete_uris {
                let url = Url::parse(&uri).map_err(|e| vortex_err!(External: e))?;
                fs.delete(url.path()).await?;
            }
            VortexResult::Ok(())
        })?;

        Ok(())
    });
}
