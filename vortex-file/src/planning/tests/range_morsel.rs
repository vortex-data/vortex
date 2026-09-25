// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Diagnostic output: one row naming the surviving file row range.

use std::ops::Range;

use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use vortex_io::request::IoConsumer;
use vortex_io::request::IoRequestId;
use vortex_io::request::IoResult;
use vortex_scan::planning::morsel::Morsel;
use vortex_scan::planning::morsel::MorselOutput;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::PlannerOutput;
use vortex_scan::planning::planner::State;
use vortex_scan::planning::planner::WorkScope;
use crate::planning::OpenedFile;

/// Emits one struct row `{ start: u64, end: u64 }` for a half-open candidate range, then
/// finishes. It reports the surviving range, not matching rows, and never requests IO.
pub struct RangeMorsel {
    range: Range<u64>,
    done: bool,
}

impl RangeMorsel {
    /// Creates the morsel; an empty range is rejected because a batch is never empty.
    pub fn new(range: Range<u64>) -> VortexResult<Self> {
        if range.is_empty() {
            vortex_bail!("RangeMorsel: range {range:?} is empty");
        }
        Ok(Self { range, done: false })
    }
}

impl IoConsumer for RangeMorsel {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Morsel for RangeMorsel {
    fn state(&self) -> State {
        if self.done {
            State::Done
        } else {
            State::NeedsCompute
        }
    }

    fn compute(&mut self) -> VortexResult<MorselOutput> {
        if self.done {
            vortex_bail!("RangeMorsel: compute called after Done");
        }
        self.done = true;
        let row = StructArray::from_fields(&[
            ("start", buffer![self.range.start].into_array()),
            ("end", buffer![self.range.end].into_array()),
        ])?;
        Ok(MorselOutput::Batch(row.into_array()))
    }
}

/// A one-shot planner that emits a [`RangeMorsel`] for the whole file and finishes.
pub struct RangeOnly {
    row_count: Option<u64>,
}

impl RangeOnly {
    /// Creates the planner for `opened`.
    pub fn new(opened: OpenedFile) -> Self {
        Self {
            row_count: Some(opened.footer.row_count()),
        }
    }
}

impl IoConsumer for RangeOnly {
    fn set_io_result(&mut self, _request: IoRequestId, _result: IoResult) {}
}

impl Planner for RangeOnly {
    fn state(&self) -> State {
        if self.row_count.is_some() {
            State::NeedsCompute
        } else {
            State::Done
        }
    }

    fn compute(&mut self) -> VortexResult<PlannerOutput> {
        let Some(row_count) = self.row_count.take() else {
            vortex_bail!("RangeOnly: compute called after Done");
        };
        let scope = WorkScope {
            file_ordinal: 0,
            rows: 0..row_count,
        };
        Ok(PlannerOutput::Morsel(
            scope,
            Box::new(RangeMorsel::new(0..row_count)?),
        ))
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;

    use super::*;
    use crate::planning::tests::fixtures::ctx;

    #[test]
    fn emits_one_range_row_then_finishes() -> VortexResult<()> {
        let mut morsel = RangeMorsel::new(3..10)?;
        assert_eq!(morsel.state(), State::NeedsCompute);
        let MorselOutput::Batch(batch) = morsel.compute()? else {
            vortex_bail!("expected a batch");
        };
        let u64_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        assert_eq!(
            batch.dtype(),
            &DType::Struct(
                StructFields::new(["start", "end"].into(), vec![u64_dtype.clone(), u64_dtype]),
                Nullability::NonNullable,
            )
        );
        let expected = StructArray::from_fields(&[
            ("start", buffer![3u64].into_array()),
            ("end", buffer![10u64].into_array()),
        ])?;
        assert_arrays_eq!(batch, expected, &mut ctx());
        assert_eq!(morsel.state(), State::Done);
        Ok(())
    }

    #[test]
    fn rejects_an_empty_range() {
        let err = RangeMorsel::new(5..5).err().map(|e| e.to_string());
        assert!(
            err.as_deref().is_some_and(|m| m.contains("is empty")),
            "{err:?}"
        );
    }
}
