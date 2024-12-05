mod blob;
mod utils;

use std::cell::RefCell;
use std::convert::Into;
use std::rc::Rc;

use futures_util::StreamExt;
use vortex::array::ChunkedArray;
use vortex::compute::{scalar_at, slice};
use vortex::dtype::{DType, PType};
use vortex::file::{LayoutContext, LayoutDeserializer, VortexReadBuilder};
use vortex::sampling_compressor::ALL_ENCODINGS_CONTEXT;
use vortex::scalar::Scalar;
use vortex::{ArrayData, IntoArrayData};
use wasm_bindgen::prelude::*;
use web_sys::js_sys::{Map, Object, Uint8Array};
use web_sys::Blob;

use crate::blob::BlobReader;
use crate::utils::set_panic_hook;

#[wasm_bindgen(js_name = File)]
pub struct VortexFile {
    reader: BlobReader,
}

#[wasm_bindgen(start)]
fn start() {
    web_sys::console::log_1(&"vortex-wasm starting".into());
    web_sys::console::log_1(&"setting panic hook...".into());
    set_panic_hook();
}

#[wasm_bindgen(js_class = File)]
impl VortexFile {
    #[wasm_bindgen(js_name = fromBlob)]
    pub async fn from_blob(blob: Blob) -> Self {
        Self {
            reader: BlobReader(Rc::new(RefCell::new(blob))),
        }
    }

    /// Log the DType to the console.
    #[wasm_bindgen(js_name = printSchema)]
    pub async fn print_schema(&self) {
        // let buffer = self.buffer.clone();
        let inner = self.reader.clone();
        let reader = VortexReadBuilder::new(
            inner,
            LayoutDeserializer::new(
                ALL_ENCODINGS_CONTEXT.clone(),
                LayoutContext::default().into(),
            ),
        )
        .build()
        .await
        .expect("building reader");

        web_sys::console::log_1(&format!("dtype = {}", reader.dtype()).into());
        web_sys::console::log_1(&format!("row_count = {}", reader.row_count()).into());
    }

    /// Materialize the entire array.
    #[wasm_bindgen]
    pub async fn collect(&self) -> ArrayBatch {
        let mut reader = VortexReadBuilder::new(
            self.reader.clone(),
            // self.buffer.clone(),
            LayoutDeserializer::new(
                ALL_ENCODINGS_CONTEXT.clone(),
                LayoutContext::default().into(),
            ),
        )
        .build()
        .await
        .expect("building reader");

        let dtype = reader.dtype().clone();
        let mut chunks = Vec::new();
        while let Some(next) = reader.next().await {
            let next = next.unwrap();
            web_sys::console::log_1(&format!("loaded another chunk of len {}", next.len()).into());
            chunks.push(next);
        }

        let chunked = ChunkedArray::try_new(chunks, dtype).unwrap().into_array();

        ArrayBatch { inner: chunked }
    }
}

#[wasm_bindgen]
pub struct ArrayBatch {
    inner: ArrayData,
}

#[wasm_bindgen]
impl ArrayBatch {
    /// Get the number of elements in this array.
    #[wasm_bindgen]
    pub fn length(&self) -> u32 {
        self.inner.len() as u32
    }

    /// Get the n-th value of an array.
    #[wasm_bindgen]
    pub fn scalar_at(&self, index: u32) -> JsValue {
        let scalar = scalar_at(&self.inner, index as usize).unwrap();
        to_js_val(scalar)
    }

    /// Slice the array to an element range.
    #[wasm_bindgen]
    pub fn slice(&self, start: u32, end: u32) -> Self {
        Self {
            inner: slice(&self.inner, start as usize, end as usize).unwrap(),
        }
    }

    /// Return the column names if array is of Struct-type.
    ///
    /// Returns `undefined` for all other types.
    #[wasm_bindgen]
    pub fn columns(&self) -> JsValue {
        let Some(struct_array) = self.inner.as_struct_array() else {
            return JsValue::undefined();
        };

        // Get a column description for each name.
        let names = js_sys::Array::new();
        for name in struct_array.names().iter() {
            names.push(&JsValue::from_str(name.as_ref()));
        }

        names.into()
    }

    /// Get the WASM bindgen types.
    #[wasm_bindgen]
    pub fn types(&self) -> JsValue {
        let Some(struct_array) = self.inner.as_struct_array() else {
            return JsValue::undefined();
        };

        let dtypes = js_sys::Array::new();
        for dtype in struct_array.dtypes() {
            dtypes.push(&JsValue::from_str(dtype.to_string().as_str()));
        }

        dtypes.into()
    }

    // Get the column from an array.
    #[wasm_bindgen]
    pub fn column(&self, name: &str) -> Self {
        let array = self
            .inner
            .as_struct_array()
            .expect("StructArray")
            .field_by_name(name)
            .expect("field not found on struct");

        Self { inner: array }
    }

    // Materialize the all the data as a JS Array.
    //
    // Note: This is very slow and should be avoided.
    pub fn to_js(&self) -> JsValue {
        let js_array = js_sys::Array::new();

        for i in 0..self.length() {
            js_array.push(&self.scalar_at(i));
        }

        js_array.into()
    }
}

fn to_js_val(scalar: Scalar) -> JsValue {
    match scalar.dtype() {
        DType::Null => JsValue::null(),
        DType::Bool(_) => scalar
            .as_bool()
            .value()
            .map(JsValue::from_bool)
            .unwrap_or_else(JsValue::null),
        DType::Primitive(ptype, _) => {
            // The scalar needs to be up-cast to f64 because that is all
            // JavaScript can represent.
            let maybe_f64_scalar = match ptype {
                PType::U8 => scalar.as_primitive().typed_value::<u8>().map(JsValue::from),
                PType::U16 => scalar
                    .as_primitive()
                    .typed_value::<u16>()
                    .map(JsValue::from),
                PType::U32 => scalar
                    .as_primitive()
                    .typed_value::<u32>()
                    .map(JsValue::from),
                PType::U64 => scalar
                    .as_primitive()
                    .typed_value::<u64>()
                    .map(JsValue::from),
                PType::I8 => scalar.as_primitive().typed_value::<i8>().map(JsValue::from),
                PType::I16 => scalar
                    .as_primitive()
                    .typed_value::<i16>()
                    .map(JsValue::from),
                PType::I32 => scalar
                    .as_primitive()
                    .typed_value::<i32>()
                    .map(JsValue::from),
                PType::I64 => scalar
                    .as_primitive()
                    .typed_value::<i64>()
                    .map(JsValue::from),
                PType::F16 => {
                    panic!("invalid type");
                }
                PType::F32 => scalar
                    .as_primitive()
                    .typed_value::<f32>()
                    .map(JsValue::from),
                PType::F64 => scalar
                    .as_primitive()
                    .typed_value::<f64>()
                    .map(JsValue::from),
            };

            // fallback to null
            maybe_f64_scalar.unwrap_or_else(JsValue::null)
        }
        DType::Utf8(_) => scalar
            .as_utf8()
            .value()
            .map(|string| JsValue::from_str(string.as_str()))
            .unwrap_or_else(JsValue::null),
        DType::Binary(_) => {
            scalar
                .as_binary()
                .value()
                .map(|binary| {
                    // Copy the data into the Uint8Array.
                    let buffer = Uint8Array::new_with_length(binary.len() as u32);
                    buffer.copy_from(binary.as_slice());
                    JsValue::from(buffer)
                })
                .unwrap_or_else(JsValue::null)
        }
        DType::Struct(..) => {
            // recursively generate the struct
            let struct_scalar = scalar.as_struct();
            let field_names = struct_scalar.dtype().as_struct().unwrap().names().clone();
            let Some(fields) = struct_scalar.fields() else {
                return JsValue::null();
            };

            // Create a new JS Object to hold all the fields.
            let properties = Map::new();
            for (field_name, scalar) in field_names.iter().zip(fields.into_iter()) {
                properties.set(&field_name.to_string().into(), &to_js_val(scalar));
            }

            // Freeze the object
            let js_obj = Object::from_entries(properties.as_ref()).unwrap();
            Object::freeze(&js_obj).into()
        }
        DType::List(..) => {
            panic!("lol");
        }
        DType::Extension(_) => JsValue::from_str("fix handling of ExtensionDType"),
    }
}
