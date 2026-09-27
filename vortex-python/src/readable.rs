// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A [`VortexReadAt`] backed by a Python object, so that Python-native IO (file objects,
//! `io.BytesIO`, fsspec or pyarrow files, custom readers) can feed the Rust reader.
//!
//! This is the Python counterpart of the JNI `JavaReadable`. Rust-native object stores are
//! already reachable through the `store=` bridge; this adapter is for IO that only exists as
//! Python code.

use std::ffi::c_int;
use std::sync::Arc;

use async_lock::Semaphore;
use futures::FutureExt;
use futures::future::BoxFuture;
use parking_lot::Mutex;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyBufferError;
use pyo3::exceptions::PyEOFError;
use pyo3::exceptions::PyTypeError;
use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyMemoryView;
use pyo3::types::PySlice;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::Handle;

/// Concurrent upcalls allowed for an object implementing the positional `read_into` protocol.
///
/// Upcalls hold the GIL except where the reader releases it (syscalls, network IO), so a modest
/// limit is enough to overlap IO latency without parking many blocking threads on the GIL.
const POSITIONAL_CONCURRENCY: usize = 16;

/// How positional reads are forwarded to the Python object.
#[derive(Clone, Copy, Debug)]
enum Protocol {
    /// `read_into(offset, buffer) -> int`: a stateless positional read, safe to call concurrently.
    Positional,
    /// `seek(offset)` followed by `readinto(buffer) -> int`: a stateful file object.
    ReadInto,
    /// `seek(offset)` followed by `read(n) -> bytes`: a stateful file object without `readinto`.
    Read,
}

/// A [`VortexReadAt`] backed by a Python object.
///
/// Two shapes of object are accepted:
///
/// - Objects with `size() -> int` and `read_into(offset, buffer) -> int`. These are positional and
///   stateless, so up to [`POSITIONAL_CONCURRENCY`] reads may be in flight at once.
/// - Binary file objects with `seek` and `readinto` (or `read`). Seeking makes these stateful, so
///   reads are serialized. The size is taken from `seek(0, SEEK_END)` at construction time.
///
/// Every read runs on the runtime's blocking pool and takes the GIL there. The destination is a
/// Rust allocation exposed to Python through the buffer protocol, so `readinto`-style readers write
/// straight into the buffer handed to the scan. Short reads are retried until the range is filled.
pub(crate) struct PyReadable {
    obj: Arc<Py<PyAny>>,
    protocol: Protocol,
    len: u64,
    uri: Option<Arc<str>>,
    handle: Handle,
    concurrency: usize,
    semaphore: Arc<Semaphore>,
}

impl PyReadable {
    /// Wrap `obj`, detecting which read protocol it implements.
    pub(crate) fn try_new(obj: &Bound<'_, PyAny>, handle: Handle) -> PyResult<Self> {
        let py = obj.py();
        let has = |name| obj.hasattr(name);

        let (protocol, len) = if has(intern!(py, "read_into"))? && has(intern!(py, "size"))? {
            let len = obj.call_method0(intern!(py, "size"))?.extract::<u64>()?;
            (Protocol::Positional, len)
        } else if has(intern!(py, "seek"))? {
            let protocol = if has(intern!(py, "readinto"))? {
                Protocol::ReadInto
            } else if has(intern!(py, "read"))? {
                Protocol::Read
            } else {
                return Err(not_readable(obj));
            };
            // `io.SEEK_END`
            let len = obj
                .call_method1(intern!(py, "seek"), (0, 2))?
                .extract::<u64>()?;
            (protocol, len)
        } else {
            return Err(not_readable(obj));
        };

        let concurrency = match protocol {
            Protocol::Positional => POSITIONAL_CONCURRENCY,
            Protocol::ReadInto | Protocol::Read => 1,
        };

        // File objects conventionally expose their path as `name`. `io.FileIO` also allows an
        // integer file descriptor there, which says nothing useful about the file.
        let uri = obj
            .getattr_opt(intern!(py, "name"))?
            .and_then(|name| name.extract::<String>().ok())
            .map(Arc::from);

        Ok(Self {
            obj: Arc::new(obj.clone().unbind()),
            protocol,
            len,
            uri,
            handle,
            concurrency,
            semaphore: Arc::new(Semaphore::new(concurrency)),
        })
    }
}

fn not_readable(obj: &Bound<'_, PyAny>) -> PyErr {
    let type_name = obj
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    PyTypeError::new_err(format!(
        "expected a path, a binary file object with `seek` and `readinto` (or `read`), or an \
         object with `size()` and `read_into(offset, buffer)`; got {type_name}"
    ))
}

impl VortexReadAt for PyReadable {
    fn uri(&self) -> Option<&Arc<str>> {
        self.uri.as_ref()
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        // Every upcall pays for taking the GIL and the backing storage is often remote, so favor
        // fewer, larger reads.
        Some(CoalesceConfig::object_storage())
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.len;
        async move { Ok(len) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let obj = Arc::clone(&self.obj);
        let protocol = self.protocol;
        let len = self.len;
        let handle = self.handle.clone();
        let semaphore = Arc::clone(&self.semaphore);

        async move {
            // Take a permit before occupying a blocking thread. For file objects the single permit
            // also keeps each `seek` paired with its read, and is taken without the GIL, so a
            // waiter never holds the GIL that the current reader needs to finish.
            let permit = match semaphore.try_acquire_arc() {
                Some(permit) => permit,
                None => semaphore.acquire_arc().await,
            };

            handle
                .spawn_blocking(move || {
                    // Keep the permit with the blocking work: dropping the read future cannot
                    // interrupt an upcall that has already started.
                    let _permit = permit;
                    let end = offset
                        .checked_add(length as u64)
                        .ok_or_else(|| vortex_err!("read {offset}+{length} overflows u64"))?;
                    if end > len {
                        vortex_bail!("read {offset}..{end} out of bounds for file of length {len}");
                    }

                    let buffer = ByteBufferMut::zeroed_aligned(length, alignment);
                    let buffer = Python::attach(|py| {
                        read_fully(py, obj.bind(py), protocol, offset, buffer).map_err(|err| {
                            vortex_err!("Python read of {offset}..{end} failed: {err}")
                        })
                    })?;
                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
                .await
        }
        .boxed()
    }
}

/// Fill `buffer` with the bytes at `offset..offset + buffer.len()`, retrying short reads.
fn read_fully(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    protocol: Protocol,
    offset: u64,
    buffer: ByteBufferMut,
) -> PyResult<ByteBufferMut> {
    let length = buffer.len();
    let dst = Bound::new(
        py,
        ReadBuffer {
            state: Mutex::new(ReadBufferState {
                buffer: Some(buffer),
                exports: 0,
            }),
        },
    )?;

    let view = PyMemoryView::from(dst.as_any())?;
    let result = (|| {
        if matches!(protocol, Protocol::ReadInto | Protocol::Read) {
            obj.call_method1(intern!(py, "seek"), (offset,))?;
        }

        let mut filled = 0;
        while filled < length {
            let n = match protocol {
                Protocol::Positional => {
                    let window = window(&view, filled, length)?;
                    let n = obj
                        .call_method1(intern!(py, "read_into"), (offset + filled as u64, &window));
                    release(&window)?;
                    n?.extract::<usize>()?
                }
                Protocol::ReadInto => {
                    let window = window(&view, filled, length)?;
                    let n = obj.call_method1(intern!(py, "readinto"), (&window,));
                    release(&window)?;
                    // `readinto` returns `None` for a non-blocking stream with no data ready.
                    n?.extract::<Option<usize>>()?.unwrap_or(0)
                }
                Protocol::Read => {
                    let chunk = obj.call_method1(intern!(py, "read"), (length - filled,))?;
                    let chunk = PyBuffer::<u8>::get(&chunk)?;
                    let n = chunk.len_bytes();
                    if n > length - filled {
                        return Err(PyBufferError::new_err(format!(
                            "read({}) returned {n} bytes",
                            length - filled
                        )));
                    }
                    let mut state = dst.get().state.lock();
                    let buffer = state
                        .buffer
                        .as_mut()
                        .ok_or_else(|| PyBufferError::new_err("read buffer was released"))?;
                    chunk.copy_to_slice(py, &mut buffer.as_mut_slice()[filled..filled + n])?;
                    n
                }
            };
            if n == 0 {
                return Err(PyEOFError::new_err(format!(
                    "reader returned 0 bytes with {} of {length} still to read",
                    length - filled
                )));
            }
            if n > length - filled {
                return Err(PyBufferError::new_err(format!(
                    "reader reported {n} bytes read into a buffer of {} bytes",
                    length - filled
                )));
            }
            filled += n;
        }
        Ok(())
    })();
    // Release the view on failure too, so the export count below reflects only what the reader
    // retained.
    let released = release(&view);
    drop(view);
    let result = match result {
        Ok(()) => released,
        Err(err) => Err(err),
    };

    // Reclaim the allocation only once Python holds no view of it. A reader that kept a view past
    // its call would otherwise see memory the scan owns; in that case the allocation stays with the
    // Python object and is freed along with it.
    let mut state = dst.get().state.lock();
    if state.exports != 0 {
        result?;
        return Err(PyBufferError::new_err(
            "reader retained a view of the read buffer after returning",
        ));
    }
    let buffer = state
        .buffer
        .take()
        .ok_or_else(|| PyBufferError::new_err("read buffer was released"))?;
    result.map(|()| buffer)
}

/// `view[start:stop]`, a writable window over the part of the buffer still to fill.
fn window<'py>(
    view: &Bound<'py, PyMemoryView>,
    start: usize,
    stop: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let start = isize::try_from(start)?;
    let stop = isize::try_from(stop)?;
    view.get_item(PySlice::new(view.py(), start, stop, 1))
}

/// `memoryview.release()`, dropping this view's export of the buffer.
///
/// Fails with `BufferError` if the reader derived another view from it and still holds that.
fn release(view: &Bound<'_, PyAny>) -> PyResult<()> {
    view.call_method0(intern!(view.py(), "release")).map(|_| ())
}

/// A writable, fixed-size Rust buffer exposed to Python through the buffer protocol.
///
/// It counts outstanding exports so the read path can tell when Python has let go of it.
#[pyclass(frozen)]
struct ReadBuffer {
    state: Mutex<ReadBufferState>,
}

struct ReadBufferState {
    /// `None` once the read path has taken the allocation back.
    buffer: Option<ByteBufferMut>,
    exports: usize,
}

#[pymethods]
impl ReadBuffer {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let mut state = slf.get().state.lock();
        let Some(buffer) = state.buffer.as_mut() else {
            return Err(PyBufferError::new_err("read buffer is no longer valid"));
        };
        let len = ffi::Py_ssize_t::try_from(buffer.len())?;
        let ptr = buffer.as_mut_slice().as_mut_ptr().cast();
        // SAFETY: `ptr` points at `len` initialized bytes owned by this object. The allocation is
        // neither moved nor freed while `exports` is non-zero: the read path only takes it back
        // once every export has been released, and `PyBuffer_FillInfo` keeps `slf` alive through
        // the view it fills.
        let rc = unsafe { ffi::PyBuffer_FillInfo(view, slf.as_ptr(), ptr, len, 0, flags) };
        if rc != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        state.exports += 1;
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        let mut state = self.state.lock();
        state.exports = state.exports.saturating_sub(1);
    }
}
