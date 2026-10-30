// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Looking codes up in dictionary values.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::DictArray;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexResult;

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
use crate::plan::physical::pipeline::AccessMode;
use crate::plan::physical::pipeline::DEFAULT_CAPACITY;
use crate::plan::physical::pipeline::PredicateCache;
use crate::plan::physical::pipeline::ValueCache;
use crate::plan::physical::pipeline::sparse_selection;

/// Wraps each batch of codes as a dictionary over the values, once the values inlet has ended.
pub(crate) struct TakeSource {
    plan: TakePlan,
    /// The values, once their inlet has ended.
    values: Option<ArrayRef>,
    /// Where values joined from the inlet are kept for the split's later scans.
    cache: Option<ValueCache>,
    access_mode: AccessMode,
    predicate: Option<(BoundExpression, Arc<PredicateCache>)>,
}

const CODES: usize = 0;
const VALUES: usize = 1;

impl TakeSource {
    /// A take over `values`, or over what its values inlet produces when `None`, keeping those
    /// in `cache`.
    pub(crate) fn new(
        plan: TakePlan,
        values: Option<ArrayRef>,
        cache: Option<ValueCache>,
        access_mode: AccessMode,
    ) -> Self {
        Self {
            plan,
            values,
            cache,
            access_mode,
            predicate: None,
        }
    }

    /// Evaluate a static predicate only for dictionary entries the codes reference.
    pub(crate) fn with_predicate(
        mut self,
        expression: BoundExpression,
        cache: Arc<PredicateCache>,
    ) -> Self {
        self.predicate = Some((expression, cache));
        self
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
                let values = shared(join_inlet(cx, VALUES, values_plan.dtype())?);
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
                if let Some((expression, cache)) = &self.predicate {
                    let values = values.as_opt::<Shared>().map_or_else(
                        || values.clone(),
                        |shared| shared.current_array_ref().clone(),
                    );
                    return Ok(Step::Final(cache.evaluate(
                        values,
                        codes,
                        expression,
                        cx.exec(),
                    )?));
                }
                // Sparse lookups keep encoded take kernels visible, including numeric kernels
                // in random-access mode. Small and dense takes reuse whole-value conversions.
                let values = if (self.access_mode == AccessMode::RandomAccess
                    || values.dtype().is_utf8()
                    || values.dtype().is_binary())
                    && values.len() >= 1024
                    && sparse_selection(codes.len(), values.len())
                    && let Some(shared) = values.as_opt::<Shared>()
                {
                    shared.current_array_ref().clone()
                } else {
                    values
                };
                Step::Final(DictArray::try_new(codes, values)?.into_array())
            }
            None => Step::End,
        })
    }
}

/// `array` as a [`SharedArray`](crate::arrays::shared::SharedArray), so however many dictionaries use it, it is canonicalized once.
fn shared(array: ArrayRef) -> ArrayRef {
    if array.is::<Shared>() {
        array
    } else {
        Shared::new(array).into_array()
    }
}
