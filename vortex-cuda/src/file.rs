// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA-aware Vortex file opening.

use std::path::Path;
use std::sync::Arc;

use itertools::Itertools;
use roaring::RoaringTreemap;
use vortex::array::ArrayRef;
use vortex::array::expr::Expression;
use vortex::array::expr::lit;
use vortex::array::scalar_fn::fns::binary::Binary;
use vortex::array::scalar_fn::fns::literal::Literal;
use vortex::array::scalar_fn::fns::operators::Operator;
use vortex::dtype::DType;
use vortex::dtype::FieldMask;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::file::VortexFile;
use vortex::file::VortexOpenOptions;
use vortex::file::v2::FileStatsLayoutReader;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::DynLayout;
use vortex::layout::LayoutReaderContext;
use vortex::layout::LayoutReaderRef;
use vortex::layout::layouts::zoned::LegacyStats;
use vortex::layout::layouts::zoned::Zoned;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::layout::segments::SegmentFuture;
use vortex::layout::segments::SegmentId;
use vortex::layout::segments::SegmentSource;
use vortex::mask::AllOr;
use vortex::mask::Mask;
use vortex::scan::selection::Selection;

use crate::CudaSessionExt;
use crate::DictionaryExport;
use crate::PooledFileReadAt;
use crate::PooledFileReadAtOptions;
use crate::layout::CUDA_SCAN_DICTIONARY_EXPORT;
use crate::layout::register_cuda_layout;

/// Extension trait for scanning CUDA-readable files with a scan-local dictionary policy.
pub trait CudaFileScanExt {
    /// Build a scan for an Arrow Device export using `dictionary_export`.
    ///
    /// With [`DictionaryExport::Decode`], CUDA-flat readers eagerly execute numeric field packs
    /// before joining field inputs. Raw primitive projections stay lazy for auxiliary readers.
    /// Other dtypes remain lazy until export. Preserve scans and ordinary [`VortexFile::scan`]
    /// calls keep dictionary encodings and do not execute eagerly.
    ///
    /// This policy is local to the reader tree: the shared session and cached ordinary readers
    /// are unchanged. It does not initialize CUDA; eager execution requires a CUDA session
    /// already installed on the file. Use the same dictionary policy when exporting the scan.
    fn scan_cuda(&self, dictionary_export: DictionaryExport)
    -> VortexResult<ScanBuilder<ArrayRef>>;
}

impl CudaFileScanExt for VortexFile {
    fn scan_cuda(
        &self,
        dictionary_export: DictionaryExport,
    ) -> VortexResult<ScanBuilder<ArrayRef>> {
        Ok(ScanBuilder::new(
            self.session().clone(),
            cuda_reader(self, dictionary_export)?,
        ))
    }
}

fn cuda_reader(
    file: &VortexFile,
    dictionary_export: DictionaryExport,
) -> VortexResult<LayoutReaderRef> {
    let ctx =
        LayoutReaderContext::new().with(*CUDA_SCAN_DICTIONARY_EXPORT, Arc::new(dictionary_export));
    let reader = file.footer().layout().new_reader(
        "".into(),
        file.segment_source(),
        file.session(),
        &ctx,
    )?;
    Ok(if let Some(stats) = file.footer().statistics().cloned() {
        Arc::new(FileStatsLayoutReader::new(
            reader,
            stats,
            file.session().clone(),
        ))
    } else {
        reader
    })
}

/// Like [`CudaFileScanExt::scan_cuda`], skipping zones whose statistics prove `filter` false.
///
/// Only zone and file statistics are consulted, before any data is read, so pruned zones cost
/// no I/O or upload. Kept zones are returned whole and may contain rows that fail `filter`:
/// callers must still filter rows exactly. Comparison literals are cast to the type of the other
/// operand, so dates may be given as their integer storage.
pub async fn scan_cuda_pruned(
    file: &VortexFile,
    dictionary_export: DictionaryExport,
    filter: &Expression,
) -> VortexResult<ScanBuilder<ArrayRef>> {
    let reader = cuda_reader(file, dictionary_export)?;
    let scan = ScanBuilder::new(file.session().clone(), Arc::clone(&reader));
    let filter = coerce_comparison_literals(filter.clone(), file.dtype())?;
    let filter = filter.bind(file.dtype())?.optimize()?;
    let rows = reader.row_count();
    // Evaluate split by split: zones align with layout splits, so each split resolves to an
    // all-pruned or all-kept mask without per-row buffers.
    let splits = SplitBy::Layout.splits(reader.as_ref(), &(0..rows), &[FieldMask::All])?;
    let evaluations = splits
        .iter()
        .tuple_windows()
        .map(|(&start, &end)| {
            let mask = Mask::new_true(usize::try_from(end - start)?);
            Ok((
                start,
                reader.pruning_evaluation(&(start..end), &filter, mask)?,
            ))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let mut pruned = RoaringTreemap::new();
    for (start, evaluation) in evaluations {
        let kept = evaluation.await?;
        let end = start + kept.len() as u64;
        match kept.slices() {
            AllOr::All => {}
            AllOr::None => {
                pruned.insert_range(start..end);
            }
            AllOr::Some(slices) => {
                let mut next = start;
                for &(slice_start, slice_end) in slices {
                    pruned.insert_range(next..start + slice_start as u64);
                    next = start + slice_end as u64;
                }
                pruned.insert_range(next..end);
            }
        }
    }
    if pruned.is_empty() {
        return Ok(scan);
    }
    Ok(scan.with_selection(Selection::ExcludeRoaring(pruned)))
}

/// Cast each comparison literal to the type of the other operand.
fn coerce_comparison_literals(expr: Expression, scope: &DType) -> VortexResult<Expression> {
    let children = expr
        .children()
        .iter()
        .map(|child| coerce_comparison_literals(child.clone(), scope))
        .collect::<VortexResult<Vec<_>>>()?;
    let expr = if children.is_empty() {
        expr
    } else {
        expr.with_children(children)?
    };
    if !expr.as_opt::<Binary>().is_some_and(|op| {
        matches!(
            op,
            Operator::Eq
                | Operator::NotEq
                | Operator::Gt
                | Operator::Gte
                | Operator::Lt
                | Operator::Lte
        )
    }) {
        return Ok(expr);
    }
    let cast = |literal: &Expression, other: &Expression| -> VortexResult<Option<Expression>> {
        let (Some(scalar), None) = (literal.as_opt::<Literal>(), other.as_opt::<Literal>()) else {
            return Ok(None);
        };
        let target = other.return_dtype(scope)?;
        if scalar.dtype().eq_ignore_nullability(&target) {
            return Ok(None);
        }
        Ok(Some(lit(scalar.cast(&target)?)))
    };
    let (lhs, rhs) = (expr.child(0).clone(), expr.child(1).clone());
    match (cast(&lhs, &rhs)?, cast(&rhs, &lhs)?) {
        (Some(lhs), _) => expr.with_children([lhs, rhs]),
        (_, Some(rhs)) => expr.with_children([lhs, rhs]),
        (None, None) => Ok(expr),
    }
}

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
        let stream = cuda_session.copy_stream()?;
        let pool = Arc::clone(cuda_session.pinned_buffer_pool());
        let reader = PooledFileReadAt::open_with_options(
            &path,
            session.handle(),
            pool,
            stream,
            self.read_at_options,
        )?;
        // Direct I/O bypasses the page cache, so there are no cached pages to copy from.
        let reader = if self.read_at_options.is_direct_io() {
            reader
        } else {
            reader.with_mapped_copy(&path, &cuda_session)?
        };
        drop(cuda_session);
        let reader = Arc::new(reader);
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
    use vortex::layout::layouts::zoned::ZonedLayout;
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

#[cfg(test)]
mod pruning_tests {
    use futures::TryStreamExt;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::TemporalArray;
    use vortex::array::expr::and;
    use vortex::array::expr::col;
    use vortex::array::expr::gt_eq;
    use vortex::array::expr::lt;
    use vortex::array::extension::datetime::TimeUnit;
    use vortex::array::scalar::Scalar;
    use vortex::buffer::ByteBufferMut;
    use vortex::error::VortexExpect;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::io::runtime::current::CurrentThreadRuntime;
    use vortex::layout::scan::split_by::SplitBy;
    use vortex::session::VortexSession;

    use super::*;
    use crate::layout::cuda_write_strategy;

    const ZONE_ROWS: usize = 1000;

    /// Scans a file of `x = 0..10_000` written in 1,000-row zones and returns the batch lengths
    /// and first values, so the test can tell exactly which zones were read.
    fn scan_zones(filter: Expression) -> VortexResult<Vec<(usize, i32)>> {
        let x = PrimitiveArray::from_iter(0i32..10_000).into_array();
        scan_column_zones(x, filter)?
            .into_iter()
            .map(|(len, first)| Ok((len, i32::try_from(&first)?)))
            .collect()
    }

    fn scan_column_zones(x: ArrayRef, filter: Expression) -> VortexResult<Vec<(usize, Scalar)>> {
        let runtime = CurrentThreadRuntime::new();
        let session = VortexSession::default().with_handle(runtime.handle());
        register_cuda_layout(&session);
        runtime.block_on(async {
            let input = StructArray::from_fields(&[("x", x)])?.into_array();
            let mut buffer = ByteBufferMut::empty();
            session
                .write_options()
                .with_strategy(cuda_write_strategy(&session, ZONE_ROWS))
                .write(&mut buffer, input.to_array_stream())
                .await?;
            let file = session.open_options().open_buffer(buffer.freeze())?;
            let batches: Vec<ArrayRef> = scan_cuda_pruned(&file, DictionaryExport::Decode, &filter)
                .await?
                .with_split_by(SplitBy::Layout)
                .into_array_stream()?
                .try_collect()
                .await?;
            let mut ctx = session.create_execution_ctx();
            batches
                .into_iter()
                .map(|batch| {
                    let first = batch
                        .execute_scalar(0, &mut ctx)?
                        .as_struct()
                        .field_by_idx(0)
                        .vortex_expect("x");
                    Ok((batch.len(), first))
                })
                .collect::<VortexResult<Vec<_>>>()
        })
    }

    /// Only zones overlapping the filter range are read; an `i64` literal is cast to the `i32`
    /// column for pruning.
    #[test]
    fn pruned_scan_reads_only_overlapping_zones() -> VortexResult<()> {
        let filter = and(gt_eq(col("x"), lit(2500i64)), lt(col("x"), lit(4500i64)));
        assert_eq!(
            scan_zones(filter)?,
            [(1000, 2000), (1000, 3000), (1000, 4000)]
        );
        Ok(())
    }

    /// Dates prune on their day-count storage; the `i32` literals are cast to the date type.
    #[test]
    fn pruned_scan_prunes_date_zones() -> VortexResult<()> {
        let days = PrimitiveArray::from_iter(0i32..10_000).into_array();
        let dates = TemporalArray::new_date(days, TimeUnit::Days).into_array();
        let filter = and(gt_eq(col("x"), lit(2500i32)), lt(col("x"), lit(4500i32)));
        let lengths: Vec<_> = scan_column_zones(dates, filter)?
            .into_iter()
            .map(|(len, _)| len)
            .collect();
        assert_eq!(lengths, [1000, 1000, 1000]);
        Ok(())
    }

    #[test]
    fn pruned_scan_keeps_everything_when_stats_cannot_prune() -> VortexResult<()> {
        let zones = scan_zones(gt_eq(col("x"), lit(0i32)))?;
        assert_eq!(zones.len(), 10);
        assert!(zones.iter().all(|&(len, _)| len == ZONE_ROWS));
        Ok(())
    }

    #[test]
    fn pruned_scan_reads_nothing_when_every_zone_is_pruned() -> VortexResult<()> {
        assert!(scan_zones(lt(col("x"), lit(0i32)))?.is_empty());
        Ok(())
    }
}
