// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reading a few rows of an uncompressed fixed-width segment as byte ranges of its values.
//!
//! A segment that stores primitive values, or fixed-size lists of them, as one uncompressed
//! buffer keeps row `i` at a computed byte offset. A scan of a few of its rows then reads only
//! those bytes instead of the whole segment.
//!
//! When the layout holds the segment's array tree, the tree gives the offset of the buffer. Else
//! the reads guess that the buffer starts the segment, and also read the end of the segment,
//! which holds the tree. They are issued together, so the guess costs no round trip. When the
//! tree shows that the guess is wrong, the scan reads the whole segment.
//!
//! The reads guess only when the segment is known to be at least as long as its uncompressed
//! values. A compressed segment is shorter, so it is read whole at once, and a guessed range
//! never ends after its segment.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VTable as _;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::SerializedArray;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

use crate::plan::physical::SegmentScanPlan;
use crate::segments::SegmentBytes;

/// A segment smaller than this is read whole: one small read costs less than several.
const MIN_SEGMENT_BYTES: u64 = 64 << 10;

/// The end of a segment read to find its array tree, when the layout does not hold it.
const TREE_TAIL_BYTES: u64 = 1 << 10;

/// Selected rows closer than this many bytes are read as one range.
const MERGE_GAP_BYTES: u64 = 4 << 10;

/// The most ranges one segment scan reads. A selection that needs more reads the segment whole.
const MAX_RANGES: usize = 64;

/// The rows of a segment, when each is a fixed number of bytes: a primitive value, or a
/// fixed-size list of them.
#[derive(Clone, Copy, Debug)]
struct RowShape {
    ptype: PType,
    /// The nullability of the primitive values.
    values: Nullability,
    /// The size and nullability of each list, when the rows are fixed-size lists.
    list: Option<(u32, Nullability)>,
}

impl RowShape {
    fn of(dtype: &DType) -> Option<Self> {
        match dtype {
            DType::Primitive(ptype, nullability) => Some(Self {
                ptype: *ptype,
                values: *nullability,
                list: None,
            }),
            DType::FixedSizeList(element, size, nullability) => match element.as_ref() {
                DType::Primitive(ptype, values) if *size > 0 => Some(Self {
                    ptype: *ptype,
                    values: *values,
                    list: Some((*size, *nullability)),
                }),
                _ => None,
            },
            _ => None,
        }
    }

    /// The values in one row.
    fn values_per_row(&self) -> u64 {
        self.list.map_or(1, |(size, _)| u64::from(size))
    }

    fn row_bytes(&self) -> u64 {
        self.values_per_row() * self.ptype.byte_width() as u64
    }
}

/// Reads of some rows of a segment as byte ranges of the buffer that holds its values.
#[derive(Debug)]
pub(crate) struct PointReads {
    shape: RowShape,
    /// The rows of the segment.
    row_count: u64,
    /// Where the values buffer starts in the segment.
    buffer_start: u64,
    /// The values buffer starts the segment by guess, so the end of the segment, which holds
    /// its array tree, must show it.
    guessed: bool,
    /// The runs of segment rows read, in order, one range each.
    runs: Vec<Range<u64>>,
    /// The rows to emit, of the rows that the runs hold.
    mask: Mask,
}

impl PointReads {
    /// The point reads of the rows `slice` and `filter` select from the segment of `plan`, when
    /// they read much less than the whole segment. `segment_length` is the byte length of the
    /// segment, when the scan knows it.
    pub(crate) fn plan(
        plan: &SegmentScanPlan,
        segment_length: Option<u64>,
        slice: Option<&Range<usize>>,
        filter: Option<&Mask>,
    ) -> VortexResult<Option<Self>> {
        let Some(shape) = RowShape::of(plan.dtype()) else {
            return Ok(None);
        };

        let row_count = plan.row_count();
        let row_bytes = shape.row_bytes();
        let segment_bytes = row_count.saturating_mul(row_bytes);
        if segment_bytes < MIN_SEGMENT_BYTES {
            return Ok(None);
        }

        let (buffer_start, guessed) = match plan.array_tree() {
            Some(tree) => {
                let tree = SerializedArray::from_array_tree(tree.clone())?;
                match values_start(&tree, &shape, row_count, plan.array_ctx()) {
                    Some(start) => (start, false),
                    None => return Ok(None),
                }
            }
            // Uncompressed values need a segment at least as long as they are.
            None if segment_length.is_some_and(|length| length >= segment_bytes) => (0, true),
            None => return Ok(None),
        };

        let Some((runs, mask)) = selected_runs(slice, filter, row_count, row_bytes)? else {
            return Ok(None);
        };

        let run_rows = runs.iter().map(|run| run.end - run.start).sum::<u64>();
        let tail = if guessed { TREE_TAIL_BYTES } else { 0 };
        let read_bytes = run_rows.saturating_mul(row_bytes).saturating_add(tail);
        if runs.len() > MAX_RANGES || read_bytes.saturating_mul(2) > segment_bytes {
            return Ok(None);
        }

        Ok(Some(Self {
            shape,
            row_count,
            buffer_start,
            guessed,
            runs,
            mask,
        }))
    }

    /// Whether the first read is the end of the segment, to confirm where the values start.
    pub(crate) fn guessed(&self) -> bool {
        self.guessed
    }

    /// The bytes to read, in order: the end of the segment when the start of its buffer is a
    /// guess, then the range of each run.
    pub(crate) fn reads(&self) -> impl Iterator<Item = SegmentBytes> + '_ {
        let tail = self.guessed.then_some(SegmentBytes::Last(TREE_TAIL_BYTES));
        let row_bytes = self.shape.row_bytes();
        let ranges = self.runs.iter().map(move |run| {
            SegmentBytes::Range(
                self.buffer_start + run.start * row_bytes..self.buffer_start + run.end * row_bytes,
            )
        });
        tail.into_iter().chain(ranges)
    }

    /// Whether `tail`, the end of the segment, shows that the values buffer starts the segment,
    /// as the reads guessed. Reads whose start is known need no check.
    pub(crate) fn confirms(&self, tail: Option<&BufferHandle>, ctx: &ReadContext) -> bool {
        if !self.guessed {
            return true;
        }

        let Some(tail) = tail.and_then(|tail| tail.as_host_opt()) else {
            return false;
        };
        let Some((head, length)) = tail.as_slice().split_last_chunk::<4>() else {
            return false;
        };
        let Some(start) = head.len().checked_sub(u32::from_le_bytes(*length) as usize) else {
            return false;
        };
        let Ok(tree) = SerializedArray::from_array_tree(ByteBuffer::copy_from(&head[start..]))
        else {
            return false;
        };
        values_start(&tree, &self.shape, self.row_count, ctx) == Some(0)
    }

    /// The emitted rows, from the bytes of each run in order.
    pub(crate) fn assemble(&self, runs: Vec<BufferHandle>) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(runs.len(), self.runs.len());

        let row_bytes = self.shape.row_bytes();
        let rows = self.mask.len();
        let width = self.shape.ptype.byte_width();
        let capacity = rows
            .checked_mul(usize::try_from(row_bytes)?)
            .ok_or_else(|| vortex_err!("Point reads of {rows} rows exceed usize"))?;
        let mut bytes = ByteBufferMut::with_capacity_aligned(capacity, Alignment::new(width));
        for (run, buffer) in self.runs.iter().zip(runs) {
            let buffer = buffer.try_to_host_sync()?;
            vortex_ensure_eq!(buffer.len() as u64, (run.end - run.start) * row_bytes);
            bytes.extend_from_slice(buffer.as_slice());
        }

        let values = PrimitiveArray::from_byte_buffer(
            bytes.freeze(),
            self.shape.ptype,
            Validity::from(self.shape.values),
        )
        .into_array();
        let array = match self.shape.list {
            Some((size, nullability)) => {
                FixedSizeListArray::try_new(values, size, Validity::from(nullability), rows)?
                    .into_array()
            }
            None => values,
        };

        if self.mask.all_true() {
            return Ok(array);
        }
        array.filter(self.mask.clone())
    }
}

/// The runs of segment rows that `slice` and `filter` select, with selected rows close together
/// merged into one run, and the mask of the selected rows over the rows of the runs. `None`
/// when the selection needs more than [`MAX_RANGES`] runs.
fn selected_runs(
    slice: Option<&Range<usize>>,
    filter: Option<&Mask>,
    row_count: u64,
    row_bytes: u64,
) -> VortexResult<Option<(Vec<Range<u64>>, Mask)>> {
    let start = slice.map_or(0, |slice| slice.start as u64);
    let len = slice.map_or(row_count, |slice| (slice.end - slice.start) as u64);
    let selected: Vec<Range<u64>> = match filter.map(Mask::slices) {
        None | Some(AllOr::All) => std::iter::once(start..start + len).collect(),
        Some(AllOr::None) => Vec::new(),
        Some(AllOr::Some(slices)) => slices
            .iter()
            .map(|&(begin, end)| start + begin as u64..start + end as u64)
            .collect(),
    };

    let gap_rows = MERGE_GAP_BYTES / row_bytes.max(1);
    let mut runs: Vec<Range<u64>> = Vec::new();
    for range in &selected {
        match runs.last_mut() {
            Some(run) if range.start - run.end <= gap_rows => run.end = range.end,
            _ => {
                if runs.len() == MAX_RANGES {
                    return Ok(None);
                }
                runs.push(range.clone());
            }
        }
    }

    // The selected rows, as positions in the rows of the runs.
    let mut slices = Vec::with_capacity(selected.len());
    let mut before = 0u64;
    let mut ranges = selected.iter().peekable();
    for run in &runs {
        while let Some(range) = ranges.next_if(|range| range.end <= run.end) {
            let begin = usize::try_from(before + range.start - run.start)?;
            let end = usize::try_from(before + range.end - run.start)?;
            slices.push((begin, end));
        }
        before += run.end - run.start;
    }

    Ok(Some((
        runs,
        Mask::from_slices(usize::try_from(before)?, slices),
    )))
}

/// Where the values buffer of `tree` starts in its segment, when the tree stores `row_count`
/// rows of `shape` as one uncompressed buffer of values and nothing else.
fn values_start(
    tree: &SerializedArray,
    shape: &RowShape,
    row_count: u64,
    ctx: &ReadContext,
) -> Option<u64> {
    let is = |node: &SerializedArray, id| ctx.resolve(node.encoding_id()) == Some(id);

    let values = match shape.list {
        Some(_) => {
            if !is(tree, FixedSizeList.id()) || tree.nchildren() != 1 {
                return None;
            }
            tree.child(0)
        }
        None => tree.clone(),
    };
    if !is(&values, Primitive.id()) || values.nchildren() != 0 || !values.metadata().is_empty() {
        return None;
    }

    let ranges = tree.buffer_ranges().ok()?;
    let [range] = ranges.as_slice() else {
        return None;
    };
    let bytes = row_count.checked_mul(shape.row_bytes())?;
    (range.len() as u64 == bytes).then_some(range.start as u64)
}
