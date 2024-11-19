use std::collections::BTreeSet;
use std::mem;

use vortex_array::array::StructArray;
use vortex_array::stats::ArrayStatistics;
use vortex_array::validity::Validity;
use vortex_array::{ArrayData, IntoArrayData};
use vortex_dtype::FieldNames;
use vortex_error::{vortex_err, VortexExpect, VortexResult};
use vortex_expr::ExprRef;

use super::IsPrunedRead;
use crate::read::mask::RowMask;
use crate::read::{BatchRead, LayoutReader};

/// Read multiple layouts by combining them into one struct array
///
/// Result can be optionally reduced with an expression, i.e. to produce a bitmask for other columns
#[derive(Debug)]
pub struct ColumnBatchReader {
    names: FieldNames,
    children: Vec<Box<dyn LayoutReader>>,
    arrays: Vec<Option<ArrayData>>,
    pruned: Vec<Option<bool>>,
    expr: Option<ExprRef>,
    // TODO(robert): This is a hack/optimization that tells us if we're reducing results with AND or not
    shortcircuit_siblings: bool,
}

impl ColumnBatchReader {
    pub fn new(
        names: FieldNames,
        children: Vec<Box<dyn LayoutReader>>,
        expr: Option<ExprRef>,
        shortcircuit_siblings: bool,
    ) -> Self {
        assert_eq!(
            names.len(),
            children.len(),
            "Names and children must be of same length"
        );
        let arrays = vec![None; children.len()];
        let pruned = vec![None; children.len()];
        Self {
            names,
            children,
            arrays,
            pruned,
            expr,
            shortcircuit_siblings,
        }
    }
}

impl LayoutReader for ColumnBatchReader {
    fn add_splits(&self, row_offset: usize, splits: &mut BTreeSet<usize>) -> VortexResult<()> {
        for child in &self.children {
            child.add_splits(row_offset, splits)?
        }
        Ok(())
    }

    fn read_selection(&mut self, selection: &RowMask) -> VortexResult<Option<BatchRead>> {
        let mut messages = Vec::new();
        for (i, child_array) in self
            .arrays
            .iter_mut()
            .enumerate()
            .filter(|(_, a)| a.is_none())
        {
            match self.children[i].read_selection(selection)? {
                Some(rr) => match rr {
                    BatchRead::ReadMore(message) => {
                        messages.extend(message);
                    }
                    BatchRead::Batch(arr) => {
                        if self.shortcircuit_siblings
                            && arr.statistics().compute_true_count().vortex_expect(
                                "must be a bool array if shortcircuit_siblings is set to true",
                            ) == 0
                        {
                            return Ok(None);
                        }
                        *child_array = Some(arr)
                    }
                },
                None => {
                    debug_assert!(
                        self.arrays.iter().all(Option::is_none),
                        "Expected layout {}({i}) to produce an array but it was empty",
                        self.names[i]
                    );
                    return Ok(None);
                }
            }
        }

        if messages.is_empty() {
            let child_arrays = mem::replace(&mut self.arrays, vec![None; self.children.len()])
                .into_iter()
                .enumerate()
                .map(|(i, a)| a.ok_or_else(|| vortex_err!("Missing child array at index {i}")))
                .collect::<VortexResult<Vec<_>>>()?;
            let len = child_arrays
                .first()
                .map(|l| l.len())
                .unwrap_or(selection.len());
            let array =
                StructArray::try_new(self.names.clone(), child_arrays, len, Validity::NonNullable)?
                    .into_array();
            self.expr
                .as_ref()
                .map(|e| e.evaluate(&array))
                .unwrap_or_else(|| Ok(array))
                .map(BatchRead::Batch)
                .map(Some)
        } else {
            Ok(Some(BatchRead::ReadMore(messages)))
        }
    }

    fn is_pruned(&mut self, begin: usize, end: usize) -> VortexResult<IsPrunedRead> {
        let mut messages = Vec::new();
        for (i, child_is_pruned) in self
            .pruned
            .iter_mut()
            .enumerate()
            .filter(|(_, a)| a.is_none())
        {
            match self.children[i].is_pruned(begin, end)? {
                IsPrunedRead::ReadMore(message) => messages.extend(message),
                IsPrunedRead::IsPruned(is_pruned) => *child_is_pruned = Some(is_pruned),
            }
        }

        if messages.is_empty() {
            let any_child_is_pruned = self
                .pruned
                .iter()
                .any(|x| x.vortex_expect("all pruned-ness should be available"));
            self.pruned = vec![None; self.pruned.len()]; // FIXME(DK): very not thread safe
            Ok(IsPrunedRead::IsPruned(any_child_is_pruned))
        } else {
            Ok(IsPrunedRead::ReadMore(messages))
        }
    }
}
