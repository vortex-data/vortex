// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use jni::JNIEnv;
use jni::objects::{JByteArray, JClass, JLongArray, JObject, JString, ReleaseMode};
use jni::sys::jlong;
use prost::Message;
use url::Url;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::error::{VortexError, VortexExpect, vortex_err};
use vortex::expr::proto::deserialize_expr_proto;
use vortex::expr::{root, select};
use vortex::file::{VortexFile, VortexOpenOptions};
use vortex::proto::expr as pb;
use vortex::utils::aliases::hash_map::HashMap;

use crate::array_iter::NativeArrayIterator;
use crate::errors::try_or_throw;
use crate::object_store::make_object_store;
use crate::{SESSION, block_on};

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
            VortexOpenOptions::file()
                .with_array_registry(Arc::new(SESSION.arrays().clone()))
                .with_layout_registry(Arc::new(SESSION.layouts().clone()))
                .open_object_store(&store, url.path()),
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
            let expr = deserialize_expr_proto(&expr_proto, SESSION.expressions())?;
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

        Ok(NativeArrayIterator::new(Box::new(scan_builder.into_array_iter()?)).into_raw())
    })
}
