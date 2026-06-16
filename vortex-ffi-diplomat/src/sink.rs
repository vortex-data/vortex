// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for the Vortex array sink.
//!
//! The hand-written C ABI defined `vx_array_sink` (a plain struct holding an mpsc `Sender` and a
//! background writer `Task`) with the free functions `vx_array_sink_open_file` (open a writable
//! stream backed by a file), `vx_array_sink_push` (push an array chunk), and `vx_array_sink_close`
//! (flush and finish, consuming the sink).
//!
//! Under Diplomat the opaque type is `VxArraySink`. `open_file` becomes a named constructor,
//! `push` mutates `&mut self`, and `close` consumes the sink by value (`self: Box<Self>`) so it
//! cannot be used afterward. Fallible operations return `Result<_, Box<VortexFfiError>>` instead
//! of an `error_out` out-parameter, and the path is a `&str` rather than a `*const c_char`.
//!
//! ## Thread safety
//!
//! As in the C ABI, the sink is not safe for concurrent operations. Only one thread should call
//! `push`/`close` at a time, and `close` must be called exactly once after all pushes complete.

#[diplomat::bridge]
pub mod ffi {
    use futures::SinkExt;
    use futures::TryStreamExt;
    use futures::channel::mpsc;
    use futures::channel::mpsc::Sender;
    use vortex::array::ArrayRef;
    use vortex::array::stream::ArrayStreamAdapter;
    use vortex::error::VortexResult;
    use vortex::error::vortex_err;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::file::WriteSummary;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::runtime::Task;
    use vortex::io::session::RuntimeSessionExt;

    use crate::RUNTIME;
    use crate::array::ffi::VxArray;
    use crate::dtype::ffi::VxDType;
    use crate::error::ffi::VortexFfiError;
    use crate::session::ffi::VxSession;

    /// A sink for writing array chunks into an external resource (currently a Vortex file).
    ///
    /// Replaces the C `vx_array_sink` struct. Push chunks with [`Self::push`], then call
    /// [`Self::close`] exactly once to flush and finish. See the module docs for thread-safety
    /// rules.
    #[diplomat::opaque]
    pub struct VxArraySink {
        sink: Sender<VortexResult<ArrayRef>>,
        writer: Task<VortexResult<WriteSummary>>,
    }

    impl VxArraySink {
        /// Open a sink that writes pushed arrays to a Vortex file at `path`.
        ///
        /// Replaces `vx_array_sink_open_file`. The `dtype` is the schema of the arrays that will
        /// be pushed. The path is a UTF-8 `&str` rather than a `*const c_char`, and failures are
        /// returned as a `Result` rather than via an `error_out` out-parameter.
        #[diplomat::attr(auto, named_constructor = "open_file")]
        pub fn open_file(
            session: &VxSession,
            path: &str,
            dtype: &VxDType,
        ) -> Result<Box<VxArraySink>, Box<VortexFfiError>> {
            let session = session.inner().clone();
            let file_dtype = dtype.inner().clone();
            let path = path.to_string();

            // Channel size 32 chosen arbitrarily, matching the C ABI.
            let (sink, rx) = mpsc::channel(32);
            let array_stream = ArrayStreamAdapter::new(file_dtype, rx.into_stream());

            let writer = session.handle().spawn(async move {
                let mut file = async_fs::File::create(path).await?;
                session.write_options().write(&mut file, array_stream).await
            });

            Ok(Box::new(VxArraySink { sink, writer }))
        }

        /// Push an array chunk into the sink.
        ///
        /// Replaces `vx_array_sink_push`. Does not take ownership of the array (it is cloned, a
        /// cheap reference-count bump). Mutates `&mut self`.
        pub fn push(&mut self, array: &VxArray) -> Result<(), Box<VortexFfiError>> {
            RUNTIME
                .block_on(self.sink.send(Ok(array.inner().clone())))
                .map_err(|e| vortex_err!("send error: {e}"))
                .map_err(Into::into)
        }

        /// Close the sink, flushing all pushed arrays to the backing resource.
        ///
        /// Replaces `vx_array_sink_close`. Consumes the sink by value so it cannot be reused;
        /// Diplomat generates no separate destructor call for a `self: Box<Self>` method. Must be
        /// called exactly once after all pushes complete.
        pub fn close(self: Box<Self>) -> Result<(), Box<VortexFfiError>> {
            let VxArraySink { sink, writer } = *self;
            drop(sink);
            RUNTIME
                .block_on(async {
                    let _summary = writer.await?;
                    VortexResult::Ok(())
                })
                .map_err(Into::into)
        }
    }
}
