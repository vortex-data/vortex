// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scattering present rows over fill rows.

use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_sparse::Sparse;

use super::join_inlet;
use crate::layouts::sparse::INDICES_DTYPE;
use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;

const INDICES: usize = 0;
const VALUES: usize = 1;

/// Joins the indices and values of the blocks a range overlaps, once both inlets have closed,
/// and emits the rows of the range its mask selects: the present ones from the values, the rest
/// the fill. Both inlets should be unbounded.
pub(crate) struct SparseSource {
    dtype: DType,
    fill: Scalar,
    /// The first row of the range, in the plan's row domain.
    start: u64,
    /// The rows of the range to emit.
    mask: Mask,
    done: bool,
}

impl SparseSource {
    pub(crate) fn new(dtype: DType, fill: Scalar, start: u64, mask: Mask) -> Self {
        Self {
            dtype,
            fill,
            start,
            mask,
            done: false,
        }
    }
}

impl Source for SparseSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if self.done {
            return Ok(SourceReady::Ready);
        }
        for inlet in [INDICES, VALUES] {
            if !cx.inlet(inlet).closed() {
                return Ok(SourceReady::Blocked(SourceBlocked::Inlet(inlet)));
            }
        }
        Ok(SourceReady::Ready)
    }
}

impl Operator for SparseSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        if self.done {
            return Ok(Step::End);
        }
        self.done = true;
        let indices = join_inlet(cx, INDICES, &INDICES_DTYPE)?;
        let values = join_inlet(cx, VALUES, &self.dtype)?;
        let indices = indices.execute::<PrimitiveArray>(cx.exec())?;
        let (keep, positions) = select(indices.as_slice::<u64>(), self.start, &self.mask)?;

        let len = self.mask.true_count();
        if keep.is_empty() {
            return Ok(Step::Final(
                ConstantArray::new(self.fill.clone(), len).into_array(),
            ));
        }
        if self.fill.is_null() {
            // A null fill is a take of the values with null indices for the absent rows, which
            // takes push into nested children instead of expanding the rows one by one.
            let mut take = vec![0u64; len];
            let mut present = vec![false; len];
            for (&value, &position) in keep.iter().zip(&positions) {
                let position = usize::try_from(position)?;
                take[position] = value as u64;
                present[position] = true;
            }
            let take = PrimitiveArray::new(Buffer::from(take), Validity::from_iter(present));
            return Ok(Step::Final(values.take(take.into_array())?));
        }
        let values = values.filter(Mask::from_indices(values.len(), keep))?;
        let positions = PrimitiveArray::new(Buffer::from(positions), Validity::NonNullable);
        let array = Sparse::try_new(
            positions.into_array(),
            values,
            len,
            self.fill.clone(),
            cx.exec(),
        )?;
        Ok(Step::Final(array.into_array()))
    }
}

/// Of the sorted present-row `indices`, those inside the range starting at `start` that `mask`
/// selects: their positions within `indices`, and their positions among the selected rows.
fn select(indices: &[u64], start: u64, mask: &Mask) -> VortexResult<(Vec<usize>, Vec<u64>)> {
    let end = start + mask.len() as u64;
    let first = indices.partition_point(|&index| index < start);
    let last = indices.partition_point(|&index| index < end);
    let in_range = &indices[first..last];
    let mut keep = Vec::with_capacity(in_range.len());
    let mut positions = Vec::with_capacity(in_range.len());
    match mask.indices() {
        AllOr::All => {
            keep.extend(first..last);
            positions.extend(in_range.iter().map(|&index| index - start));
        }
        AllOr::None => {}
        AllOr::Some(selected) => {
            // Both are sorted, so one pass ranks each present row among the selected rows.
            let mut rank = 0;
            for (offset, &index) in in_range.iter().enumerate() {
                let local = usize::try_from(index - start)?;
                rank += selected[rank..].partition_point(|&row| row < local);
                if selected.get(rank) == Some(&local) {
                    keep.push(first + offset);
                    positions.push(rank as u64);
                }
            }
        }
    }
    Ok((keep, positions))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::select;

    #[rstest]
    #[case::all_selected(Mask::new_true(6), vec![1, 2, 3], vec![1, 2, 4])]
    #[case::none_selected(Mask::new_false(6), vec![], vec![])]
    #[case::some_selected(Mask::from_indices(6, [0, 2, 3, 4]), vec![2, 3], vec![1, 3])]
    fn selects_present_rows_in_range(
        #[case] mask: Mask,
        #[case] keep: Vec<usize>,
        #[case] positions: Vec<u64>,
    ) -> VortexResult<()> {
        // Present rows 3, 11, 12 and 14 of a range starting at row 10: only 11, 12 and 14 lie
        // inside it.
        let (actual_keep, actual_positions) = select(&[3, 11, 12, 14, 20], 10, &mask)?;
        assert_eq!(actual_keep, keep);
        assert_eq!(actual_positions, positions);
        Ok(())
    }
}
