// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::collections::VecDeque;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_layout::LayoutReaderRef;
use vortex_session::VortexSession;

use crate::ScanBuilder;
use crate::api::DataSource;
use crate::api::DataSourceScan;
use crate::api::DataSourceScanRef;
use crate::api::Estimate;
use crate::api::ScanRequest;
use crate::api::Split;
use crate::api::SplitRef;

/// An implementation of a [`DataSource`] that reads data from a [`LayoutReaderRef`].
pub struct LayoutReaderDataSource {
    reader: LayoutReaderRef,
    session: VortexSession,
}

impl LayoutReaderDataSource {
    /// Creates a new [`LayoutReaderDataSource`].
    pub fn new(reader: LayoutReaderRef, session: VortexSession) -> Self {
        Self { reader, session }
    }
}

#[async_trait]
impl DataSource for LayoutReaderDataSource {
    fn dtype(&self) -> &DType {
        self.reader.dtype()
    }

    fn row_count_estimate(&self) -> Estimate<u64> {
        Estimate::Exact(self.reader.row_count())
    }

    async fn scan(&self, scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
        let mut builder = ScanBuilder::new(self.session.clone(), self.reader.clone());

        if let Some(projection) = scan_request.projection {
            builder = builder.with_projection(projection);
        }

        if let Some(filter) = scan_request.filter {
            builder = builder.with_filter(filter);
        }

        if let Some(limit) = scan_request.limit {
            // TODO(ngates): ScanBuilder limit should be u64
            let limit = usize::try_from(limit)?;
            builder = builder.with_limit(limit);
        }

        let scan = builder.prepare()?;
        let dtype = scan.dtype().clone();
        let splits = scan.execute(None)?;

        Ok(Box::new(LayoutReaderScan {
            dtype,
            splits: VecDeque::from_iter(splits),
        }))
    }

    fn serialize_split(&self, _split: &dyn Split) -> VortexResult<Vec<u8>> {
        vortex_bail!("LayoutReader splits are not yet serializable");
    }

    fn deserialize_split(&self, _split: &[u8]) -> VortexResult<SplitRef> {
        vortex_bail!("LayoutReader splits are not yet serializable");
    }
}

struct LayoutReaderScan {
    dtype: DType,
    splits: VecDeque<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>,
}

#[async_trait]
impl DataSourceScan for LayoutReaderScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn remaining_splits_estimate(&self) -> Estimate<usize> {
        Estimate::Exact(self.splits.len())
    }

    async fn next_splits(&mut self, max_splits: usize) -> VortexResult<Vec<SplitRef>> {
        let n = std::cmp::min(max_splits, self.splits.len());

        let mut splits = Vec::with_capacity(n);
        for _ in 0..n {
            let fut = self
                .splits
                .pop_front()
                .vortex_expect("Checked length above ensures we have enough splits");
            splits.push(Box::new(LayoutReaderSplit {
                dtype: self.dtype.clone(),
                fut,
            }) as SplitRef);
        }

        Ok(splits)
    }
}

struct LayoutReaderSplit {
    dtype: DType,
    fut: BoxFuture<'static, VortexResult<Option<ArrayRef>>>,
}

#[async_trait]
impl Split for LayoutReaderSplit {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
        Ok(Box::pin(ArrayStreamAdapter::new(
            self.dtype,
            stream::once(self.fut).filter_map(|a| async move { a.transpose() }),
        )))
    }

    fn row_count_estimate(&self) -> Estimate<u64> {
        Estimate::Unknown
    }

    fn byte_size_estimate(&self) -> Estimate<u64> {
        Estimate::Unknown
    }
}
