// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA-aware Vortex file opening.

use std::path::Path;
use std::sync::Arc;

use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::file::VortexFile;
use vortex::file::VortexOpenOptions;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::DynLayout;
use vortex::layout::layouts::zoned::LegacyStats;
use vortex::layout::layouts::zoned::Zoned;
use vortex::layout::segments::SegmentFuture;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSource;

use crate::CudaSessionExt;
use crate::PooledFileReadAt;
use crate::PooledFileReadAtOptions;
use crate::layout::register_cuda_layout;

/// Extension trait for opening CUDA-readable files from [`VortexOpenOptions`].
pub trait CudaOpenOptionsExt {
    /// Configure the opener to keep control-plane segments on the host and read data-plane
    /// segments into CUDA device memory.
    fn with_cuda(self) -> CudaOpenOptions;
}

impl CudaOpenOptionsExt for VortexOpenOptions {
    fn with_cuda(self) -> CudaOpenOptions {
        CudaOpenOptions {
            inner: self,
            read_at_options: PooledFileReadAtOptions::default(),
        }
    }
}

/// CUDA-aware Vortex file open options.
///
/// Construct this adapter with [`CudaOpenOptionsExt::with_cuda`]. Standard file options should be
/// configured before calling `with_cuda`.
pub struct CudaOpenOptions {
    inner: VortexOpenOptions,
    read_at_options: PooledFileReadAtOptions,
}

impl CudaOpenOptions {
    /// Configure how the pooled data-plane reader opens and reads the local file.
    ///
    /// Footer and zone-map reads continue to use the standard buffered host reader.
    pub fn with_read_at_options(mut self, options: PooledFileReadAtOptions) -> Self {
        self.read_at_options = options;
        self
    }

    /// Open a local Vortex file for CUDA execution.
    ///
    /// The footer and zone-map segments are read through the ordinary host path. All other file
    /// segments are staged through the session-scoped pinned buffer pool and transferred to the
    /// device.
    pub async fn open_path(self, path: impl AsRef<Path>) -> VortexResult<VortexFile> {
        let path = path.as_ref().to_owned();
        register_cuda_layout(self.inner.session());

        // A host cache cannot front the data source: cached host buffers would violate the CUDA
        // source's device-placement contract. It remains enabled on the control-plane source.
        let data_options = self.inner.clone().without_segment_cache();
        let control_file = self.inner.open_path(&path).await?;
        let footer = control_file.footer().clone();

        let session = control_file.session().clone();
        let cuda_session = session.cuda_session();
        let stream = cuda_session.stream()?;
        let pool = Arc::clone(cuda_session.pinned_buffer_pool());
        drop(cuda_session);

        let reader = Arc::new(PooledFileReadAt::open_with_options(
            &path,
            session.handle(),
            pool,
            stream,
            self.read_at_options,
        )?);
        let data_file = data_options
            .with_footer(footer.clone())
            .open(reader)
            .await?;

        let control_segments =
            control_plane_segments(footer.layout().as_ref(), footer.segment_map().len())?;
        let routed = Arc::new(RoutingSegmentSource {
            control_segments: control_segments.into(),
            control: control_file.segment_source(),
            data: data_file.segment_source(),
        });

        Ok(data_file.with_segment_source(routed))
    }
}

struct RoutingSegmentSource {
    control_segments: Arc<[bool]>,
    control: Arc<dyn SegmentSource>,
    data: Arc<dyn SegmentSource>,
}

impl SegmentSource for RoutingSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let Ok(idx) = usize::try_from(*id) else {
            return self.data.request(id);
        };
        if self.control_segments.get(idx).copied().unwrap_or(false) {
            self.control.request(id)
        } else {
            self.data.request(id)
        }
    }
}

fn control_plane_segments(layout: &dyn DynLayout, segment_count: usize) -> VortexResult<Vec<bool>> {
    let mut control_segments = vec![false; segment_count];
    mark_zone_map_segments(layout, &mut control_segments)?;
    Ok(control_segments)
}

fn mark_zone_map_segments(
    layout: &dyn DynLayout,
    control_segments: &mut [bool],
) -> VortexResult<()> {
    if layout.is::<Zoned>() || layout.is::<LegacyStats>() {
        let (Some(data), Some(zones)) = (layout.slot(0)?, layout.slot(1)?) else {
            vortex_bail!("zone map layout is missing its data or zones child");
        };
        mark_layout_segments(zones.as_ref(), control_segments)?;
        mark_zone_map_segments(data.as_ref(), control_segments)?;
        return Ok(());
    }

    for child in layout.children()? {
        mark_zone_map_segments(child.as_ref(), control_segments)?;
    }
    Ok(())
}

fn mark_layout_segments(layout: &dyn DynLayout, control_segments: &mut [bool]) -> VortexResult<()> {
    for id in layout.segment_ids() {
        let idx = usize::try_from(*id)?;
        let Some(is_control) = control_segments.get_mut(idx) else {
            vortex_bail!("layout references out-of-range segment {id}");
        };
        *is_control = true;
    }

    for child in layout.children()? {
        mark_layout_segments(child.as_ref(), control_segments)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use futures::FutureExt;
    use parking_lot::Mutex;
    use vortex::array::buffer::BufferHandle;
    use vortex::array::dtype::DType;
    use vortex::array::dtype::Nullability::NonNullable;
    use vortex::array::dtype::PType;
    use vortex::array::dtype::StructFields;
    use vortex::buffer::ByteBuffer;
    use vortex::layout::layouts::flat::FlatLayout;
    use vortex::layout::layouts::flat::FlatLayoutExt;
    use vortex::layout::layouts::zoned::ZonedLayout;
    use vortex::layout::layouts::zoned::ZonedLayoutExt;
    use vortex::session::registry::ReadContext;

    use super::*;

    #[derive(Clone)]
    struct RecordingSource {
        requests: Arc<Mutex<Vec<SegmentId>>>,
        value: u8,
    }

    impl RecordingSource {
        fn new(value: u8) -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                value,
            }
        }
    }

    impl SegmentSource for RecordingSource {
        fn request(&self, id: SegmentId) -> SegmentFuture {
            self.requests.lock().push(id);
            let value = self.value;
            async move { Ok(BufferHandle::new_host(ByteBuffer::from(vec![value]))) }.boxed()
        }
    }

    #[tokio::test]
    async fn routes_control_and_data_segments() {
        let control = RecordingSource::new(1);
        let data = RecordingSource::new(2);
        let source = RoutingSegmentSource {
            control_segments: Arc::from([false, true, false]),
            control: Arc::new(control.clone()),
            data: Arc::new(data.clone()),
        };

        let control_buffer = source
            .request(SegmentId::from(1))
            .await
            .unwrap()
            .unwrap_host();
        let data_buffer = source
            .request(SegmentId::from(2))
            .await
            .unwrap()
            .unwrap_host();

        assert_eq!(control_buffer.as_ref(), &[1]);
        assert_eq!(data_buffer.as_ref(), &[2]);
        assert_eq!(*control.requests.lock(), [SegmentId::from(1)]);
        assert_eq!(*data.requests.lock(), [SegmentId::from(2)]);
    }

    #[test]
    fn marks_only_zone_map_subtree_segments_as_control_plane() -> VortexResult<()> {
        let read_ctx = ReadContext::new([]);
        let data_dtype = DType::Primitive(PType::I32, NonNullable);
        let data =
            FlatLayout::new(16, data_dtype, SegmentId::from(0), read_ctx.clone()).into_layout();
        let zones = FlatLayout::new(
            1,
            DType::Struct(StructFields::empty(), NonNullable),
            SegmentId::from(1),
            read_ctx,
        )
        .into_layout();
        let layout =
            ZonedLayout::try_new(data, zones, NonZeroUsize::new(16).unwrap(), Arc::new([]))?
                .into_layout();

        let control_segments = control_plane_segments(layout.as_ref(), 2)?;

        assert_eq!(control_segments, [false, true]);
        Ok(())
    }
}
