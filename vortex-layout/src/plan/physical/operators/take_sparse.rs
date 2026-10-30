// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Read dictionary codes first, then request only the referenced regions of the values plan.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::join;
use crate::plan::physical::TakePlan;
use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;
use crate::plan::physical::pipeline::AccessMode;
use crate::plan::physical::pipeline::ReadId;
use crate::plan::physical::pipeline::Scan;
use crate::plan::physical::pipeline::Split;
use crate::plan::physical::pipeline::Turn;
use crate::plan::physical::pipeline::ValueCache;
use crate::segments::SegmentId;

const VALUE_BLOCK_ROWS: u64 = 65_536;

pub(crate) struct SparseTakeSource {
    plan: TakePlan,
    cache: ValueCache,
    row_offset: u64,
    active: Option<ValueScan>,
    output: Option<ArrayRef>,
}

struct ValueScan {
    scan: Scan,
    codes: ArrayRef,
    pieces: Vec<(usize, ArrayRef)>,
    pending: FxHashMap<SegmentId, VecDeque<ReadId>>,
    selected: usize,
    reads: usize,
}

impl SparseTakeSource {
    pub(crate) fn new(plan: TakePlan, cache: ValueCache, row_offset: u64) -> Self {
        Self {
            plan,
            cache,
            row_offset,
            active: None,
            output: None,
        }
    }

    fn start(&mut self, codes: ArrayRef, cx: &mut Cx<'_>) -> VortexResult<()> {
        let values = self.plan.values()?;
        let codes = codes
            .cast(DType::Primitive(PType::U64, codes.dtype().nullability()))?
            .execute::<PrimitiveArray>(cx.exec())?;
        let valid = codes.validity()?.execute_mask(codes.len(), cx.exec())?;
        let original = codes.as_slice::<u64>();
        let mut selected = Vec::new();
        let mut select = |row: usize| -> VortexResult<()> {
            vortex_ensure!(
                original[row] < values.row_count(),
                "Dictionary code out of bounds"
            );
            selected.push(original[row]);
            Ok(())
        };
        match valid.indices() {
            AllOr::All => (0..codes.len()).try_for_each(&mut select)?,
            AllOr::Some(rows) => rows.iter().copied().try_for_each(&mut select)?,
            AllOr::None => {}
        }
        selected.sort_unstable();
        selected.dedup();
        if selected.is_empty() {
            self.output = Some(if codes.is_empty() {
                Canonical::empty(self.plan.dtype()).into_array()
            } else {
                ConstantArray::new(Scalar::null(self.plan.dtype().clone()), codes.len())
                    .into_array()
            });
            return Ok(());
        }
        let mut remapped = vec![0u64; codes.len()];
        let mut remap = |row: usize| {
            remapped[row] = selected
                .binary_search(&original[row])
                .ok()
                .vortex_expect("selected codes include every valid code")
                as u64;
        };
        match valid.indices() {
            AllOr::All => (0..codes.len()).for_each(&mut remap),
            AllOr::Some(rows) => rows.iter().copied().for_each(&mut remap),
            AllOr::None => {}
        }
        let remapped = PrimitiveArray::new(
            Buffer::from(remapped),
            Validity::from_mask(valid, codes.dtype().nullability()),
        )
        .into_array();
        let splits = selected
            .chunk_by(|left, right| left / VALUE_BLOCK_ROWS == right / VALUE_BLOCK_ROWS)
            .map(|group| {
                let first = group[0];
                let end = group[group.len() - 1] + 1;
                Ok(Split {
                    rows: first..end,
                    mask: Mask::from_indices(
                        usize::try_from(end - first)?,
                        group.iter().map(|&row| {
                            usize::try_from(row - first)
                                .vortex_expect("value block indices fit in usize")
                        }),
                    ),
                })
            })
            .collect::<VortexResult<Vec<_>>>()?;
        self.active = Some(ValueScan {
            scan: Scan::try_new(cx.session().clone(), values, splits)?
                .with_value_cache(self.cache.clone())
                .with_access_mode(AccessMode::RandomAccess)
                .with_row_offset(self.row_offset),
            codes: remapped,
            pieces: Vec::new(),
            pending: FxHashMap::default(),
            selected: selected.len(),
            reads: 0,
        });
        Ok(())
    }
}

impl Source for SparseTakeSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if self.output.is_some() {
            return Ok(SourceReady::Ready);
        }
        if self.active.is_none() {
            match cx.inlet(0).take() {
                Some(codes) => self.start(codes, cx)?,
                None if cx.inlet(0).closed() => return Ok(SourceReady::Ready),
                None => return Ok(SourceReady::Blocked(SourceBlocked::Inlet(0))),
            }
        }
        if self.output.is_some() {
            return Ok(SourceReady::Ready);
        }
        let active = self.active.as_mut().vortex_expect("value scan started");
        while let Some((segment, bytes)) = cx.take_segment() {
            let read = active
                .pending
                .get_mut(&segment)
                .and_then(VecDeque::pop_front)
                .ok_or_else(|| vortex_err!("Unrequested dictionary values segment {segment}"))?;
            active.scan.deliver(read, bytes)?;
        }
        loop {
            match active.scan.step()? {
                Turn::Read(read) => {
                    active
                        .pending
                        .entry(read.segment_id)
                        .or_default()
                        .push_back(read.id);
                    active.reads += 1;
                    cx.request_read(read.segment_id);
                }
                Turn::Output(split, array) => active.pieces.push((split, array)),
                Turn::Waiting => return Ok(SourceReady::Blocked(SourceBlocked::Io)),
                Turn::Done => {
                    let mut active = self.active.take().vortex_expect("value scan exists");
                    active.pieces.sort_by_key(|(split, _)| *split);
                    let values = join(
                        self.plan.values()?.dtype(),
                        active.pieces.into_iter().map(|(_, array)| array).collect(),
                    )?;
                    vortex_ensure_eq!(values.len(), active.selected);
                    tracing::debug!(
                        selected_values = active.selected,
                        dictionary_values = self.plan.values()?.row_count(),
                        value_reads = active.reads,
                        "v2 selected dictionary reads"
                    );
                    self.output = Some(DictArray::try_new(active.codes, values)?.into_array());
                    return Ok(SourceReady::Ready);
                }
            }
        }
    }
}

impl Operator for SparseTakeSource {
    fn compute(&mut self, _input: Input, _cx: &mut Cx<'_>) -> VortexResult<Step> {
        Ok(match self.output.take() {
            Some(array) => Step::Final(array),
            None => Step::End,
        })
    }
}
