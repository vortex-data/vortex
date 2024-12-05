use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;

use bytes::{Bytes, BytesMut};
use futures_channel::oneshot;
use futures_util::FutureExt;
use js_sys::Uint8Array;
use vortex::io::VortexReadAt;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use web_sys::{Blob, FileReader};

#[derive(Clone)]
pub struct BlobReader(pub Rc<RefCell<Blob>>);

// (•_•)           it's time to get
// ( •_•)>⌐■-■    ...
// (⌐■_■)          Send + Sync
//
// This is safe because the browser runs single-threaded.
// TODO(aduffy): is this actually safe?
unsafe impl Send for BlobReader {}
unsafe impl Sync for BlobReader {}

impl VortexReadAt for BlobReader {
    fn read_byte_range(
        &self,
        pos: u64,
        len: u64,
    ) -> impl Future<Output = std::io::Result<Bytes>> + 'static {
        let this = self.clone();
        web_sys::console::log_1(&format!("read_byte_range({pos}, {len})").into());

        let (tx, rx) = oneshot::channel();

        let start: i32 = pos.try_into().unwrap();
        let end: i32 = (pos + len).try_into().unwrap();
        let sliced = this.0.borrow().slice_with_i32_and_i32(start, end).unwrap();

        let file_reader = FileReader::new().unwrap();
        let file_reader_cb = file_reader.clone();

        // Send the onload handler
        let loadend = Closure::once_into_js(move || {
            let array_buf = file_reader_cb.result().unwrap();
            let array = Uint8Array::new(array_buf.as_ref());
            let mut result = BytesMut::with_capacity(len.try_into().unwrap());
            unsafe {
                result.set_len(result.capacity());
            }
            array.copy_to(&mut result);

            // Send the result to the main thread.
            tx.send(result).unwrap();
        });
        file_reader.set_onloadend(loadend.dyn_ref());

        // Trigger the streaming read.
        file_reader.read_as_array_buffer(&sliced).unwrap();

        // Return the reader which will be awaited.
        rx.map(|res| Ok(res.unwrap().freeze()))
    }

    fn size(&self) -> impl Future<Output = std::io::Result<u64>> + 'static {
        std::future::ready(Ok(self.0.borrow().size() as u64))
    }
}
