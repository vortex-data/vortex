// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::arc_wrapper;
use crate::array::vx_array;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::session::vx_session;
use crate::string::vx_view;

arc_wrapper!(
    /// A handle to a Vortex file encapsulating the footer and logic for instantiating a reader.
    VortexFile,
    vx_file
);

#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_file_write_array(
    session: *const vx_session,
    path: vx_view,
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) {
    let session = vx_session::as_ref(session);
    let options = session.write_options();
    let array = vx_array::as_ref(array);
    try_or_default(error_out, || {
        let path = unsafe { path.as_str() }?;

        RUNTIME.block_on(async move {
            options
                .write(
                    &mut async_fs::File::create(path).await?,
                    array.to_array_stream(),
                )
                .await?;
            Ok(())
        })
    });
}
