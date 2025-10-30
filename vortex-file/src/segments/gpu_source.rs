// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::sync::Arc;

use cudarc::cufile::{Cufile, FileHandle};
use cudarc::driver::CudaStream;
use vortex_error::{VortexExpect, vortex_err};
use vortex_flatbuffers::footer::SegmentSpec;
use vortex_layout::segments::SegmentId;

pub struct FileGpuSegmentSource {
    segments: Arc<[SegmentSpec]>,
    stream: Arc<CudaStream>,
    cu_file: Arc<Cufile>,
    file_handle: FileHandle,
}

impl FileGpuSegmentSource {
    fn new(segments: Arc<[SegmentSpec]>, stream: Arc<CudaStream>, file: File) -> Self {
        let cu_file = Cufile::new()
            .map_err(|e| vortex_err!("cu file {e}"))
            .vortex_expect("Failed to create cufile");

        let file_handle = cu_file
            .register(file)
            .map_err(|e| vortex_err!("cu file register {e}"))
            .vortex_unwrap();

        FileGpuSegmentSource {
            segments,
            stream,
            cu_file,
            file_handle,
        }
    }
}

impl GpuSegmentSource for FileGpuSegmentSource {
    fn request(&self, id: SegmentId) -> GpuSegmentFuture {
        let spec = *self
            .segments
            .get(*id as usize)
            .vortex_expect("missing segment id");

        let cu_slice = self
            .stream
            .alloc_zeros::<u8>(spec.length())
            .map_err(|e| vortex_err!("cu slice {e}"))
            .vortex_expect("Failed to allocate cu slice");

        // this is optional? and has strange perf characteristics.
        // cu_file
        //     .buf_register(&cu_slice)
        //     .map_err(|e| vortex_err!("cu file {e}"))
        //     .vortex_unwrap();

        async move {
            let written = self
                .file_handle
                .sync_read(
                    i64::try_from(spec.offset()).vortex_expect("must fit"),
                    &cu_slice,
                )
                .map_err(|e| vortex_err!("cu file {e}"));
            assert_eq!(written, cu_slice.len());
            cu_slice
        }
    }
}
