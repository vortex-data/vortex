// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter and project over natural splits, one morsel per split.
//!
//! Deliberate shortcut: data reads do not go through the protocol. The morsel evaluates through
//! the existing `LayoutReader` and `FileSegmentSource`, blocking on their futures. The runtime
//! that drives them is created by the planner on its worker, after `start()`, so no runtime is
//! captured by the `Send` pending work that constructs the stage.

use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use vortex_array::MaskFuture;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_file::VortexFile;
use vortex_file::segments::FileSegmentSource;
use vortex_file::segments::RequestMetrics;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::single::SingleThreadRuntime;
use vortex_layout::LayoutReader;
use vortex_mask::Mask;
use vortex_metrics::DefaultMetricsRegistry;
use vortex_session::VortexSession;

use crate::io::IoConsumer;
use crate::io::IoRequestId;
use crate::io::IoResult;
use crate::morsel::Morsel;
use crate::morsel::MorselOutput;
use crate::planner::Planner;
use crate::planner::PlannerOutput;
use crate::planner::State;
use crate::planner::WorkScope;
use crate::stages::OpenedFile;

/// Opens the file's layout on the first compute, then emits one [`SplitMorsel`] per natural
/// split in file order and finishes.
pub struct FilterProject {
    opened: Option<OpenedFile>,
    filter: Option<Expression>,
    projection: Expression,
    session: VortexSession,
    prepared: Option<Prepared>,
    next_split: usize,
}

struct Prepared {
    /// Drives the segment source's read driver and the morsels' evaluation futures. Owned here
    /// and shared with the morsels, so it lives as long as any of them.
    runtime: Rc<SingleThreadRuntime>,
    reader: Arc<dyn LayoutReader>,
    filter: Option<BoundExpression>,
    projection: BoundExpression,
    splits: Vec<Range<u64>>,
}

impl FilterProject {
    /// Creates the stage; the layout is opened and expressions bound on the first compute.
    pub fn new(
        opened: OpenedFile,
        filter: Option<Expression>,
        projection: Expression,
        session: VortexSession,
    ) -> Self {
        Self {
            opened: Some(opened),
            filter,
            projection,
            session,
            prepared: None,
            next_split: 0,
        }
    }

    fn prepare(&mut self, opened: OpenedFile) -> VortexResult<()> {
        let dtype = opened.footer.dtype().clone();
        let runtime = Rc::new(SingleThreadRuntime::default());
        let source = FileSegmentSource::open(
            opened.footer.segment_specs_with_metadata(),
            opened.read,
            runtime.handle(),
            RequestMetrics::new(&DefaultMetricsRegistry::default(), vec![]),
        );
        let file = VortexFile::new(opened.footer, Arc::new(source), self.session.clone());
        self.prepared = Some(Prepared {
            runtime,
            reader: file.layout_reader()?,
            filter: self
                .filter
                .as_ref()
                .map(|filter| filter.bind(&dtype))
                .transpose()?,
            projection: self.projection.bind(&dtype)?,
            splits: file.splits()?,
        });
        Ok(())
    }
}

impl IoConsumer for FilterProject {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for FilterProject {
    fn state(&self) -> State {
        match &self.prepared {
            None if self.opened.is_some() => State::NeedsCompute,
            None => State::Done,
            Some(prepared) if self.next_split < prepared.splits.len() => State::NeedsCompute,
            Some(_) => State::Done,
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        if let Some(opened) = self.opened.take() {
            self.prepare(opened)?;
            return Ok(PlannerOutput::Continue);
        }
        let Some(prepared) = &self.prepared else {
            vortex_bail!("FilterProject: compute called after Done");
        };
        let Some(range) = prepared.splits.get(self.next_split).cloned() else {
            return Ok(PlannerOutput::Done);
        };
        self.next_split += 1;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: range.clone(),
        };
        Ok(PlannerOutput::Morsel(
            scope,
            Box::new(SplitMorsel {
                reader: Arc::clone(&prepared.reader),
                range,
                filter: prepared.filter.clone(),
                projection: prepared.projection.clone(),
                runtime: Rc::clone(&prepared.runtime),
                done: false,
            }),
        ))
    }
}

/// Evaluates the filter and projection for one split in a single compute, blocking on the
/// layout reader's futures. A split with no matching rows finishes without a batch.
pub struct SplitMorsel {
    reader: Arc<dyn LayoutReader>,
    range: Range<u64>,
    filter: Option<BoundExpression>,
    projection: BoundExpression,
    runtime: Rc<SingleThreadRuntime>,
    done: bool,
}

impl IoConsumer for SplitMorsel {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Morsel for SplitMorsel {
    fn state(&self) -> State {
        if self.done {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        if self.done {
            vortex_bail!("SplitMorsel: compute called after Done");
        }
        self.done = true;
        let len = usize::try_from(self.range.end - self.range.start)?;
        let mut mask = MaskFuture::ready(Mask::new_true(len));
        if let Some(filter) = &self.filter {
            mask = self.reader.filter_evaluation(&self.range, filter, mask)?;
        }
        let array = self.runtime.block_on(self.reader.projection_evaluation(
            &self.range,
            &self.projection,
            mask,
        )?)?;
        if array.is_empty() {
            Ok(MorselOutput::Done)
        } else {
            Ok(MorselOutput::Batch(array))
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::checked_add;
    use vortex_array::expr::col;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;

    use super::*;
    use crate::tests::fixtures::SESSION;
    use crate::tests::fixtures::concat;
    use crate::tests::fixtures::open_buffer;
    use crate::tests::fixtures::reference_scan;
    use crate::tests::fixtures::write_chunked_test_file;
    use crate::tests::scripted::ctx;

    fn chunks() -> Vec<ArrayRef> {
        vec![
            buffer![1u32, 2, 3, 4].into_array(),
            buffer![5u32, 6, 7, 8].into_array(),
        ]
    }

    fn two_chunk_file() -> VortexResult<ByteBuffer> {
        write_chunked_test_file(&[("numbers", chunks())])
    }

    fn stage(
        buffer: &ByteBuffer,
        filter: Option<Expression>,
        projection: Expression,
    ) -> VortexResult<FilterProject> {
        let footer = open_buffer(buffer)?.footer().clone();
        Ok(FilterProject::new(
            OpenedFile {
                read: Arc::new(buffer.clone()),
                size: buffer.len() as u64,
                footer,
            },
            filter,
            projection,
            SESSION.clone(),
        ))
    }

    /// Drives the planner to `Done`, collecting every emitted morsel with its scope.
    fn morsels(planner: &mut FilterProject) -> VortexResult<Vec<(WorkScope, Box<dyn Morsel>)>> {
        let mut out = Vec::new();
        loop {
            match planner.state() {
                State::Done => return Ok(out),
                State::NeedsIO(batch) => vortex_bail!("unexpected IO {batch:?}"),
                State::NeedsCompute => match planner.compute()? {
                    PlannerOutput::Done => return Ok(out),
                    PlannerOutput::Continue => continue,
                    PlannerOutput::Morsel(scope, morsel) => out.push((scope, morsel)),
                    other => vortex_bail!("unexpected output {other:?}"),
                },
            }
        }
    }

    /// Runs one morsel to completion and returns its batches.
    fn batches(mut morsel: Box<dyn Morsel>) -> VortexResult<Vec<ArrayRef>> {
        let mut out = Vec::new();
        loop {
            match morsel.state() {
                State::Done => return Ok(out),
                State::NeedsIO(batch) => vortex_bail!("unexpected IO {batch:?}"),
                State::NeedsCompute => match morsel.compute()? {
                    MorselOutput::Done => return Ok(out),
                    MorselOutput::Continue => continue,
                    MorselOutput::Batch(array) => out.push(array),
                    MorselOutput::NeedsIO(batch) => vortex_bail!("unexpected IO {batch:?}"),
                },
            }
        }
    }

    #[test]
    fn one_morsel_per_split_in_order() -> VortexResult<()> {
        let buffer = two_chunk_file()?;
        let mut planner = stage(&buffer, None, root())?;
        let scopes: Vec<_> = morsels(&mut planner)?
            .into_iter()
            .map(|(scope, _)| scope.rows)
            .collect();
        assert_eq!(scopes, vec![0..4, 4..8]);
        Ok(())
    }

    #[test]
    fn root_projection_returns_each_chunk() -> VortexResult<()> {
        let buffer = two_chunk_file()?;
        let mut planner = stage(&buffer, None, root())?;
        for ((_, morsel), chunk) in morsels(&mut planner)?.into_iter().zip(chunks()) {
            let out = batches(morsel)?;
            assert_eq!(out.len(), 1);
            let expected = StructArray::from_fields(&[("numbers", chunk)])?;
            assert_arrays_eq!(out[0], expected, &mut ctx());
        }
        Ok(())
    }

    #[test]
    fn filter_across_the_chunk_boundary() -> VortexResult<()> {
        let buffer = two_chunk_file()?;
        let mut planner = stage(&buffer, Some(gt(col("numbers"), lit(6u32))), root())?;
        let mut morsels = morsels(&mut planner)?.into_iter();
        let (_, first) = morsels.next().ok_or_else(|| vortex_err("first"))?;
        assert!(batches(first)?.is_empty());
        let (_, second) = morsels.next().ok_or_else(|| vortex_err("second"))?;
        let out = batches(second)?;
        assert_eq!(out.len(), 1);
        let expected = StructArray::from_fields(&[("numbers", buffer![7u32, 8].into_array())])?;
        assert_arrays_eq!(out[0], expected, &mut ctx());
        Ok(())
    }

    fn vortex_err(which: &str) -> vortex_error::VortexError {
        vortex_error::vortex_err!("missing {which} morsel")
    }

    #[test]
    fn filter_matching_nothing_yields_no_batches() -> VortexResult<()> {
        let buffer = two_chunk_file()?;
        let mut planner = stage(&buffer, Some(gt(col("numbers"), lit(100u32))), root())?;
        let morsels = morsels(&mut planner)?;
        assert_eq!(morsels.len(), 2);
        for (_, morsel) in morsels {
            assert!(batches(morsel)?.is_empty());
        }
        Ok(())
    }

    #[test]
    fn expression_projection_matches_the_existing_scan() -> VortexResult<()> {
        let buffer = two_chunk_file()?;
        let filter = gt(col("numbers"), lit(2u32));
        let projection = checked_add(col("numbers"), lit(1u32));
        let mut planner = stage(&buffer, Some(filter.clone()), projection.clone())?;
        let mut actual = Vec::new();
        for (_, morsel) in morsels(&mut planner)? {
            actual.extend(batches(morsel)?);
        }
        let expected = reference_scan(&buffer, Some(filter), projection)?;
        assert_eq!(actual.len(), 2);
        let dtype = expected
            .first()
            .map(|array| array.dtype().clone())
            .ok_or_else(|| vortex_err("reference"))?;
        assert_arrays_eq!(
            concat(actual, &dtype)?,
            concat(expected, &dtype)?,
            &mut ctx()
        );
        Ok(())
    }

    #[test]
    fn unbound_projection_column_fails_on_first_compute() -> VortexResult<()> {
        let buffer = two_chunk_file()?;
        let mut planner = stage(&buffer, None, col("missing"))?;
        let err = planner.compute().err().map(|e| e.to_string());
        assert!(
            err.as_deref().is_some_and(|m| m.contains("missing")),
            "{err:?}"
        );
        Ok(())
    }
}
