use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use futures_util::FutureExt;
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::ContextRef;
use vortex_dtype::{DType, FieldPath};
use vortex_error::{vortex_err, VortexResult};
use vortex_layout::scan::unified::UnifiedDriverFuture;
use vortex_layout::scan::{ScanBuilder, ScanDriver};

use crate::footer::FileLayout;
use crate::open::FileType;
use crate::segments::SegmentCache;

pub struct VortexFile<F: FileType> {
    pub(crate) read: F::Read,
    pub(crate) options: F::Options,
    pub(crate) ctx: ContextRef,
    pub(crate) file_layout: FileLayout,
    pub(crate) segment_cache: Arc<dyn SegmentCache>,
    pub(crate) _marker: PhantomData<F>,
}

impl<F: FileType> VortexFile<F> {
    pub fn row_count(&self) -> u64 {
        self.file_layout.row_count()
    }

    pub fn dtype(&self) -> &DType {
        self.file_layout.dtype()
    }

    pub fn file_layout(&self) -> &FileLayout {
        &self.file_layout
    }

    pub fn file_stats(&self) -> &[StatsSet] {
        self.file_layout.statistics()
    }

    pub fn statistics(
        &self,
        field_paths: Arc<[FieldPath]>,
        stats: Arc<[Stat]>,
    ) -> VortexResult<impl Future<Output = VortexResult<Vec<StatsSet>>> + 'static + use<'_, F>>
    {
        let driver = F::scan_driver(
            self.read.clone(),
            self.options.clone(),
            self.file_layout.clone(),
            self.segment_cache.clone(),
        );

        // TODO(ngates): this section should disappear when we store file-level statistics.
        //  That's why it's a little odd that we have to manually setup a driver here.

        // Create a single LayoutReader that is reused for the entire scan.
        let reader = self
            .file_layout
            .root_layout()
            .reader(driver.segment_reader(), self.ctx.clone())?;

        let (send, recv) = oneshot::channel::<VortexResult<Vec<StatsSet>>>();

        let result_future = recv.map(|result| match result {
            Ok(result) => result,
            Err(_) => Err(vortex_err!("Failed to receive result, send dropped")),
        });
        let driver_stream = driver.drive_future(async move {
            let field_paths = field_paths.clone();
            let stats = stats.clone();
            reader
                .clone()
                .evaluate_stats(field_paths, stats)
                .map(|result| match send.send(result) {
                    Ok(()) => Ok(()),
                    Err(_) => Err(vortex_err!("Failed to send result, recv dropped")),
                })
                .await
        });

        Ok(UnifiedDriverFuture {
            exec_future: result_future,
            io_stream: driver_stream,
        })
    }

    pub fn scan(&self) -> ScanBuilder<F::ScanDriver> {
        let driver = F::scan_driver(
            self.read.clone(),
            self.options.clone(),
            self.file_layout.clone(),
            self.segment_cache.clone(),
        );
        ScanBuilder::new(
            driver,
            self.file_layout.root_layout().clone(),
            self.ctx.clone(),
        )
    }
}
