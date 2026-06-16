// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diplomat bridge for Vortex file IO.
//!
//! The hand-written C ABI exposed `vx_file` (an `arc_wrapper!` over `VortexFile`) and the free
//! function `vx_file_write_array`, which took a `*const c_char` path and an
//! `error_out: *mut *mut vx_error` out-parameter. In the Diplomat port the writer is a method on
//! the session, the path arrives as a `&str`, and failures are reported through
//! `Result<(), Box<VortexFfiError>>` instead of the error out-parameter.

#[diplomat::bridge]
pub mod ffi {
    use vortex::error::vortex_err;
    use vortex::file::VortexFile;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::runtime::BlockingRuntime;

    use crate::RUNTIME;
    use crate::array::ffi::VxArray;
    use crate::error::ffi::VortexFfiError;
    use crate::session::ffi::VxSession;

    /// A handle to a Vortex file, encapsulating the footer and the logic for instantiating a
    /// reader.
    ///
    /// Mirrors the `vx_file` opaque type from the C ABI (`arc_wrapper!` over `VortexFile`).
    #[diplomat::opaque]
    pub struct VxFile(pub(crate) VortexFile);

    impl VxFile {
        /// Write an array to a Vortex file at `path` using the given session's write options.
        ///
        /// This replaces the C ABI `vx_file_write_array`. On failure it returns an error rather
        /// than writing into an `error_out` out-parameter, and the path is a UTF-8 `&str` rather
        /// than a null-terminated `*const c_char`.
        pub fn write_array(
            session: &VxSession,
            path: &str,
            array: &VxArray,
        ) -> Result<(), Box<VortexFfiError>> {
            let options = session.inner().write_options();
            RUNTIME
                .block_on(async move {
                    options
                        .write(
                            &mut async_fs::File::create(path)
                                .await
                                .map_err(|e| vortex_err!("failed to create file: {e}"))?,
                            array.inner().to_array_stream(),
                        )
                        .await?;
                    Ok(())
                })
                .map_err(Into::into)
        }
    }
}
