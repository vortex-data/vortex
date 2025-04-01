use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{BoxStream, LocalBoxStream};
use vortex_array::stats::StatsSet;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_io::PerformanceHint;
use vortex_layout::segments::SegmentId;
use vortex_metrics::VortexMetrics;

use crate::ScanBuilder;
use crate::footer::Footer;
use crate::scan::segments::CoalescedSegmentRequest;
use crate::segments::SegmentCache;

#[derive(Clone)]
pub struct VortexFile {
    /// The footer of the Vortex file.
    pub(crate) footer: Footer,
    /// Cache containing any segments that were incidentally read as part of the initial read.
    pub(crate) segment_cache: Arc<dyn SegmentCache>,
    pub(crate) io_driver: Option<Arc<dyn IoDriver>>,
    /// Metrics tied to the file.
    pub(crate) metrics: VortexMetrics,
}

impl VortexFile {
    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    pub fn row_count(&self) -> u64 {
        self.footer.row_count()
    }

    pub fn dtype(&self) -> &DType {
        self.footer.dtype()
    }

    pub fn file_stats(&self) -> Option<&Arc<[StatsSet]>> {
        self.footer.statistics()
    }

    pub fn metrics(&self) -> &VortexMetrics {
        &self.metrics
    }

    pub fn scan(&self) -> ScanBuilder {
        ScanBuilder::new(self.clone())
    }
}

pub trait IoDriver: 'static + Send + Sync {
    fn drive(
        &self,
        requests: LocalBoxStream<'static, VortexResult<CoalescedSegmentRequest>>,
    ) -> LocalBoxStream<'static, VortexResult<()>>;
}
