use std::marker::PhantomData;
use std::sync::Arc;

use futures::channel::oneshot;
use futures_util::{FutureExt, TryFutureExt};
use vortex_array::stats::{Stat, StatsSet};
use vortex_array::ContextRef;
use vortex_dtype::{DType, FieldPath};
use vortex_error::{vortex_err, VortexResult};
use vortex_layout::scan::unified::UnifiedDriverFuture;
use vortex_layout::scan::{Scan, ScanDriver};

use crate::footer::FileLayout;
use crate::open::VortexFileOpener;
use crate::segments::SegmentCache;

/// Trait for a Vortex file.
///
/// This exists so different Vortex file types can configure their scan operations differently.
pub struct VortexFile<F: VortexFileOpener> {
    pub(crate) read: F::Read,
    pub(crate) options: F::Options,
    pub(crate) ctx: ContextRef,
    pub(crate) file_layout: FileLayout,
    pub(crate) segment_cache: Arc<dyn SegmentCache>,
    pub(crate) _marker: PhantomData<F>,
}

impl<F: VortexFileOpener> VortexFile<F> {
    pub fn row_count(&self) -> u64 {
        self.file_layout.row_count()
    }

    pub fn dtype(&self) -> &DType {
        self.file_layout.dtype()
    }

    pub fn file_layout(&self) -> &FileLayout {
        &self.file_layout
    }

    pub async fn statistics(
        &self,
        field_paths: Arc<[FieldPath]>,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Vec<StatsSet>> {
        let driver = F::scan_driver(
            self.read.clone(),
            self.options.clone(),
            self.file_layout.clone(),
            self.segment_cache.clone(),
        );

        // Create a single LayoutReader that is reused for the entire scan.
        let reader = self
            .file_layout
            .root_layout()
            .reader(driver.segment_reader(), self.ctx.clone())?;

        let (send, recv) = oneshot::channel::<VortexResult<Vec<StatsSet>>>();

        let task_future = async move {
            reader
                .clone()
                .evaluate_stats(field_paths, stats)
                .map(move |result| match send.send(result) {
                    Ok(()) => Ok(()),
                    Err(_) => Err(vortex_err!("Failed to send result, recv dropped")),
                })
                .await
        };

        let driver_stream = driver.drive_future(task_future);

        UnifiedDriverFuture {
            exec_future: recv.map_err(|_| vortex_err!("Failed to receive result, send dropped")),
            io_stream: driver_stream,
        }
        .await?
    }

    pub fn scan(&self) -> Scan<F::ScanDriver> {
        let driver = F::scan_driver(
            self.read.clone(),
            self.options.clone(),
            self.file_layout.clone(),
            self.segment_cache.clone(),
        );
        Scan::new(
            driver,
            self.file_layout.root_layout().clone(),
            self.ctx.clone(),
        )
    }
}
