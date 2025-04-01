use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{BoxStream, LocalBoxStream};
use vortex_array::stats::StatsSet;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexError, VortexResult};
use vortex_io::PerformanceHint;
use vortex_layout::segments::{SegmentId, SegmentSource};
use vortex_metrics::VortexMetrics;

use crate::ScanBuilder;
use crate::footer::Footer;
use crate::segments::SegmentCache;

#[derive(Clone)]
pub struct VortexFile {
    /// The footer of the Vortex file.
    pub(crate) footer: Footer,
    /// A source for reading segments from the file.
    pub(crate) driver: Arc<dyn FileDriver>,
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

pub trait FileDriver {
    fn spawn(&self) -> (Arc<dyn SegmentSource>, BoxStream<'static, VortexError>);
}
