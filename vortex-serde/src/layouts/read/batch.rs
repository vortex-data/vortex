use std::mem;

use vortex::array::StructArray;
use vortex::validity::Validity;
use vortex::{Array, IntoArray};
use vortex_dtype::FieldNames;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};

use crate::layouts::read::selection::RowSelector;
use crate::layouts::read::{LayoutReader, ReadResult};
use crate::layouts::{Message, RangeResult};

#[derive(Debug)]
pub struct ColumnBatchReader {
    names: FieldNames,
    children: Vec<Box<dyn LayoutReader>>,
    read_ranges: Vec<Option<RowSelector>>,
    arrays: Vec<Option<Array>>,
}

impl ColumnBatchReader {
    pub fn new(names: FieldNames, children: Vec<Box<dyn LayoutReader>>) -> Self {
        let arrays = vec![None; children.len()];
        let read_ranges = vec![None; children.len()];
        Self {
            names,
            children,
            read_ranges,
            arrays,
        }
    }

    pub fn next_range(&mut self) -> VortexResult<RangeResult> {
        let mut messages = Vec::new();
        for (i, child_selector) in self
            .read_ranges
            .iter_mut()
            .enumerate()
            .filter(|(_, a)| a.is_none())
        {
            match self.children[i].next_range()? {
                RangeResult::ReadMore(m) => messages.extend(m),
                RangeResult::Rows(s) => match s {
                    None => return Ok(RangeResult::Rows(None)),
                    Some(rs) => {
                        if rs.is_empty() {
                            return self.advance(rs.end()).map(RangeResult::ReadMore);
                        }
                        *child_selector = Some(rs);
                    }
                },
            }
        }

        if messages.is_empty() {
            let ranges = mem::replace(&mut self.read_ranges, vec![None; self.children.len()]);
            let mut ranges_iter = ranges.iter().enumerate();
            let mut final_range: Option<RowSelector> =
                ranges_iter.next().and_then(|(_, ri)| ri.clone());
            for (i, range) in ranges_iter {
                let Some(column_range) = range else {
                    vortex_bail!("Finished reading all columns but column {i} didn't produce range")
                };
                final_range = final_range.and_then(|fr| {
                    let intersection = fr.intersect(column_range);
                    if intersection.is_empty() {
                        None
                    } else {
                        Some(intersection)
                    }
                })
            }

            if let Some(fr) = final_range.as_ref() {
                self.read_ranges = ranges
                    .into_iter()
                    .map(|rs| rs.and_then(|r| r.advance(fr.end())))
                    .collect();
            }
            Ok(RangeResult::Rows(final_range))
        } else {
            Ok(RangeResult::ReadMore(messages))
        }
    }

    pub fn read_next(&mut self, selection: RowSelector) -> VortexResult<Option<ReadResult>> {
        let mut messages = Vec::new();
        for (i, child_array) in self
            .arrays
            .iter_mut()
            .enumerate()
            .filter(|(_, a)| a.is_none())
        {
            match self.children[i].read_next(selection.clone())? {
                Some(rr) => match rr {
                    ReadResult::ReadMore(message) => {
                        messages.extend(message);
                    }
                    ReadResult::Batch(a) => *child_array = Some(a),
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
            Ok(Some(ReadResult::Batch(
                StructArray::try_new(self.names.clone(), child_arrays, len, Validity::NonNullable)?
                    .into_array(),
            )))
        } else {
            Ok(Some(ReadResult::ReadMore(messages)))
        }
    }

    pub fn advance(&mut self, up_to_row: usize) -> VortexResult<Vec<Message>> {
        self.arrays = vec![None; self.children.len()];
        self.read_ranges = mem::take(&mut self.read_ranges)
            .into_iter()
            .map(|s| s.and_then(|rs| rs.advance(up_to_row)))
            .collect();

        let mut messages = Vec::new();
        for c in self.children.iter_mut() {
            messages.extend(c.advance(up_to_row)?);
        }
        Ok(messages)
    }
}

#[derive(Debug)]
pub struct ColumnBatchFilter {
    children: Vec<Box<dyn LayoutReader>>,
    read_ranges: Vec<Option<RowSelector>>,
    filters: Vec<Option<RowSelector>>,
}

impl ColumnBatchFilter {
    pub fn new(children: Vec<Box<dyn LayoutReader>>) -> Self {
        let read_ranges = vec![None; children.len()];
        let filters = vec![None; children.len()];
        Self {
            children,
            read_ranges,
            filters,
        }
    }

    pub fn next_range(&mut self) -> VortexResult<RangeResult> {
        let mut messages = Vec::new();
        for (i, child_selector) in self
            .read_ranges
            .iter_mut()
            .enumerate()
            .filter(|(_, a)| a.is_none())
        {
            match self.children[i].next_range()? {
                RangeResult::ReadMore(m) => messages.extend(m),
                RangeResult::Rows(s) => match s {
                    None => return Ok(RangeResult::Rows(None)),
                    Some(rs) => {
                        if rs.is_empty() {
                            return self.advance(rs.end()).map(RangeResult::ReadMore);
                        }
                        *child_selector = Some(rs);
                    }
                },
            }
        }

        if messages.is_empty() {
            let ranges = mem::replace(&mut self.read_ranges, vec![None; self.children.len()]);
            let mut ranges_iter = ranges.iter().enumerate();
            let mut final_range: Option<RowSelector> =
                ranges_iter.next().and_then(|(_, ri)| ri.clone());
            for (i, range) in ranges_iter {
                let Some(column_range) = range else {
                    vortex_bail!("Finished reading all columns but column {i} didn't produce range")
                };
                final_range = final_range.and_then(|fr| {
                    let intersection = fr.intersect(column_range);
                    if intersection.is_empty() {
                        None
                    } else {
                        Some(intersection)
                    }
                })
            }

            if let Some(fr) = final_range.as_ref() {
                self.read_ranges = ranges
                    .into_iter()
                    .map(|rs| rs.and_then(|r| r.advance(fr.end())))
                    .collect();
            }
            Ok(RangeResult::Rows(final_range))
        } else {
            Ok(RangeResult::ReadMore(messages))
        }
    }

    pub fn read_next(&mut self, selection: RowSelector) -> VortexResult<Option<ReadResult>> {
        let mut messages = Vec::new();
        for (i, child_selector) in self
            .filters
            .iter_mut()
            .enumerate()
            .filter(|(_, a)| a.is_none())
        {
            match self.children[i].read_next(selection.clone())? {
                Some(rr) => match rr {
                    ReadResult::ReadMore(msgs) => messages.extend(msgs),
                    ReadResult::Selector(s) => {
                        if s.is_empty() {
                            return self.advance(s.end()).map(ReadResult::ReadMore).map(Some);
                        }
                        *child_selector = Some(s)
                    }
                    ReadResult::Batch(a) => unreachable!("Can only produce selectors"),
                },
                None => {
                    debug_assert!(
                        self.filters.iter().all(Option::is_none),
                        "Expected layout to produce an array but it was empty"
                    );
                    return Ok(None);
                }
            }
        }

        if messages.is_empty() {
            let selectors = mem::replace(&mut self.filters, vec![None; self.children.len()])
                .into_iter()
                .enumerate()
                .map(|(i, a)| a.ok_or_else(|| vortex_err!("Missing child array at index {i}")))
                .collect::<VortexResult<Vec<_>>>()?;
            let mut selector_iter = selectors.into_iter();
            // TODO(robert): Handle empty projections
            let mut current = selector_iter
                .next()
                .vortex_expect("Must have at least one child");
            for next_filter in selector_iter {
                if current.is_empty() {
                    return self.read_next(selection);
                }
                current = current.intersect(&next_filter);
            }
            Ok(Some(ReadResult::Selector(current)))
        } else {
            Ok(Some(ReadResult::ReadMore(messages)))
        }
    }

    pub fn advance(&mut self, up_to_row: usize) -> VortexResult<Vec<Message>> {
        self.filters = mem::take(&mut self.filters)
            .into_iter()
            .map(|s| s.and_then(|rs| rs.advance(up_to_row)))
            .collect();

        self.read_ranges = mem::take(&mut self.read_ranges)
            .into_iter()
            .map(|s| s.and_then(|rs| rs.advance(up_to_row)))
            .collect();

        let mut messages = Vec::new();
        for c in self.children.iter_mut() {
            messages.extend(c.advance(up_to_row)?);
        }
        Ok(messages)
    }
}
