// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Looking codes up in dictionary values.

use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::match_each_integer_ptype;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::UNBOUNDED;
use super::join_inlet;
use crate::arrays::shared::Shared;
use crate::arrays::shared::SharedArrayExt;
use crate::plan::physical::TakePlan;
use crate::plan::physical::operator::Cx;
use crate::plan::physical::operator::Input;
use crate::plan::physical::operator::Operator;
use crate::plan::physical::operator::Source;
use crate::plan::physical::operator::SourceBlocked;
use crate::plan::physical::operator::SourceReady;
use crate::plan::physical::operator::Step;
use crate::plan::physical::pipeline::DEFAULT_CAPACITY;
use crate::plan::physical::pipeline::ValueCache;

/// Wraps each batch of codes as a dictionary over the values, once the values inlet has ended.
pub(crate) struct TakeSource {
    plan: TakePlan,
    /// The values, once their inlet has ended.
    values: Option<ArrayRef>,
    /// Where values joined from the inlet are kept for the split's later scans.
    cache: Option<ValueCache>,
}

const CODES: usize = 0;
const VALUES: usize = 1;

impl TakeSource {
    /// A take over `values`, or over what its values inlet produces when `None`, keeping those
    /// in `cache`.
    pub(crate) fn new(plan: TakePlan, values: Option<ArrayRef>, cache: Option<ValueCache>) -> Self {
        Self {
            plan,
            values,
            cache,
        }
    }

    /// The batches inlet `inlet` holds: the values are taken whole, so they never block.
    pub(crate) fn capacity(inlet: usize) -> usize {
        if inlet == VALUES {
            UNBOUNDED
        } else {
            DEFAULT_CAPACITY
        }
    }
}

impl Source for TakeSource {
    fn ready(&mut self, cx: &mut Cx<'_>) -> VortexResult<SourceReady> {
        if self.values.is_none() && !cx.inlet(VALUES).closed() {
            return Ok(SourceReady::Blocked(SourceBlocked::Inlet(VALUES)));
        }
        let codes = cx.inlet(CODES);
        Ok(if codes.is_empty() && !codes.closed() {
            SourceReady::Blocked(SourceBlocked::Inlet(CODES))
        } else {
            SourceReady::Ready
        })
    }
}

impl Operator for TakeSource {
    fn compute(&mut self, _input: Input, cx: &mut Cx<'_>) -> VortexResult<Step> {
        let values = match &self.values {
            Some(values) => values.clone(),
            None => {
                let values_plan = self.plan.values()?;
                let values =
                    join_inlet(cx, VALUES, values_plan.dtype())?.optimize_ctx(cx.session())?;
                let values = shared(values);
                let values = match &self.cache {
                    Some(cache) => cache.insert(&values_plan, values),
                    None => values,
                };
                self.values = Some(values.clone());
                values
            }
        };
        Ok(match cx.inlet(CODES).take() {
            Some(codes) => {
                // Decode each referenced entry once, keeping repeated row codes as a
                // dictionary. Small or dense takes still share the whole decoded dictionary.
                if (values.dtype().is_utf8() || values.dtype().is_binary())
                    && values.len() >= 1024
                    && codes.len().saturating_mul(4) < values.len()
                    && let Some(shared) = values.as_opt::<Shared>()
                {
                    Step::Final(compact_take(codes, shared.current_array_ref(), cx.exec())?)
                } else {
                    Step::Final(DictArray::try_new(codes, values)?.into_array())
                }
            }
            None => Step::End,
        })
    }
}

fn compact_take(
    codes: ArrayRef,
    values: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        codes.dtype().is_int(),
        "expected type: int but instead got {}",
        codes.dtype()
    );
    let codes = codes.execute::<PrimitiveArray>(ctx)?;
    let validity = codes.validity()?;
    let valid = validity.execute_mask(codes.len(), ctx)?;
    let values_len = values.len();
    let mut mapping = FxHashMap::with_capacity_and_hasher(codes.len(), Default::default());
    let mut selected = Vec::<u64>::with_capacity(codes.len());
    let mut remapped = Vec::with_capacity(codes.len());
    match_each_integer_ptype!(codes.ptype(), |P| {
        for (&code, is_valid) in codes.as_slice::<P>().iter().zip(valid.iter()) {
            if !is_valid {
                remapped.push(0u64);
                continue;
            }
            let index = usize::try_from(code)?;
            vortex_ensure!(index < values_len, "Dictionary code out of bounds");
            let code = match mapping.entry(index) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let code = u64::try_from(selected.len())?;
                    selected.push(u64::try_from(index)?);
                    *entry.insert(code)
                }
            };
            remapped.push(code);
        }
    });
    let values = values
        .take(Buffer::from(selected).into_array())?
        .execute::<Canonical>(ctx)?
        .into_array();
    let codes = PrimitiveArray::new(Buffer::from(remapped), validity).into_array();
    Ok(DictArray::try_new(codes, values)?.into_array())
}

/// `array` as a [`SharedArray`](crate::arrays::shared::SharedArray), so however many dictionaries use it, it is canonicalized once.
fn shared(array: ArrayRef) -> ArrayRef {
    if array.is::<Shared>() {
        array
    } else {
        Shared::new(array).into_array()
    }
}
