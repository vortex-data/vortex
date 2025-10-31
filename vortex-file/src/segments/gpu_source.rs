// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::sync::Arc;

use cudarc::cufile::{Cufile, FileHandle};
use cudarc::driver::CudaStream;
use futures::FutureExt;
use vortex_error::{VortexExpect, VortexUnwrap, vortex_err};
use vortex_layout::segments::{GpuSegmentFuture, GpuSegmentSource, SegmentId};

use crate::SegmentSpec;

pub struct FileGpuSegmentSource {
    segments: Arc<[SegmentSpec]>,
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    cu_file: Arc<Cufile>,
    file_handle: Arc<FileHandle>,
}

impl FileGpuSegmentSource {
    pub fn new(segments: Arc<[SegmentSpec]>, stream: Arc<CudaStream>, file: File) -> Self {
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
            file_handle: Arc::new(file_handle),
        }
    }
}

impl GpuSegmentSource for FileGpuSegmentSource {
    fn request(&self, id: SegmentId) -> GpuSegmentFuture {
        let spec = self
            .segments
            .get(*id as usize)
            .vortex_expect("missing segment id")
            .clone();

        let mut cu_slice = unsafe { self.stream.alloc::<u8>(spec.length as usize) }
            .map_err(|e| vortex_err!("cu slice {e}"))
            .vortex_expect("Failed to allocate cu slice");

        // this is optional? and has strange perf characteristics.
        // self.cu_file
        //     .buf_register(&cu_slice)
        //     .map_err(|e| vortex_err!("cu file {e}"))
        //     .vortex_unwrap();
        let offset = i64::try_from(spec.offset).vortex_expect("must fit");

        let file_handle = self.file_handle.clone();
        let stream = self.stream.clone();
        async move {
            // println!("try read");
            file_handle.sync_read(offset, &mut cu_slice);
            let read = stream
                .memcpy_ftod(&file_handle, offset, &mut cu_slice)
                .ok()
                .vortex_expect("memcpy_ftod");
            // println!("did read");

            // read.synchronize()
            //     .map_err(|e| vortex_err!("sync write {e}"))
            //     .vortex_unwrap();
            // println!("did sync");
            Ok(cu_slice)
        }
        .boxed()
    }
}
