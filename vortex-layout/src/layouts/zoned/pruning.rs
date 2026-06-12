// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Read-time pruning support for zoned layouts.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use parking_lot::RwLock;
use tracing::trace;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::dynamic::DynamicExprUpdates;
use vortex_error::SharedVortexResult;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;

use crate::LazyReaderChildren;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::zone_map::ZoneMap;

type SharedZoneMap = Shared<BoxFuture<'static, SharedVortexResult<ZoneMap>>>;
pub(super) type SharedPruningResult =
    Shared<BoxFuture<'static, SharedVortexResult<Arc<PruningResult>>>>;
type PredicateCache = Arc<OnceLock<Option<Expression>>>;

pub(super) struct PruningState {
    zone_count: usize,
    row_count: u64,
    zone_len: u64,
    dtype: DType,
    lazy_children: Arc<LazyReaderChildren>,
    session: VortexSession,

    pruning_result: LazyLock<DashMap<Expression, Option<SharedPruningResult>>>,
    zone_map: OnceLock<SharedZoneMap>,
    pruning_predicates: LazyLock<Arc<DashMap<Expression, PredicateCache>>>,
}

impl PruningState {
    pub(super) fn new(
        layout: &ZonedLayout,
        lazy_children: Arc<LazyReaderChildren>,
        session: VortexSession,
    ) -> Self {
        Self {
            zone_count: layout.nzones(),
            row_count: layout.row_count(),
            zone_len: layout.zone_len() as u64,
            dtype: layout.dtype().clone(),
            lazy_children,
            session,
            pruning_result: Default::default(),
            zone_map: Default::default(),
            pruning_predicates: Default::default(),
        }
    }

    pub(super) fn pruning_mask_future(&self, expr: Expression) -> Option<SharedPruningResult> {
        if let Some(result) = self.pruning_result.get(&expr) {
            return result.value().clone();
        }

        self.pruning_result
            .entry(expr.clone())
            .or_insert_with(|| match self.pruning_predicate(expr.clone()) {
                None => {
                    trace!(%expr, "no pruning predicate");
                    None
                }
                Some(predicate) => {
                    trace!(%expr, ?predicate, "constructed pruning predicate");
                    let zone_map = self.zone_map();
                    let dynamic_updates = DynamicExprUpdates::new(&expr);
                    let session = self.session.clone();

                    Some(
                        async move {
                            let zone_map = zone_map.await?;
                            let initial_mask =
                                zone_map.prune(&predicate, &session).map_err(|err| {
                                    err.with_context(format!(
                                        "While evaluating pruning predicate {} (derived from {})",
                                        predicate, expr
                                    ))
                                })?;
                            Ok(Arc::new(PruningResult {
                                zone_map,
                                predicate,
                                dynamic_updates,
                                latest_result: RwLock::new((0, initial_mask)),
                                session,
                            }))
                        }
                        .boxed()
                        .shared(),
                    )
                }
            })
            .clone()
    }

    fn pruning_predicate(&self, expr: Expression) -> Option<Expression> {
        self.pruning_predicates
            .entry(expr.clone())
            .or_default()
            .get_or_init(move || match expr.falsify(&self.dtype, &self.session) {
                Ok(predicate) => predicate,
                Err(error) => {
                    trace!(%expr, %error, "failed to construct stats rewrite predicate");
                    None
                }
            })
            .clone()
    }

    fn zone_map(&self) -> SharedZoneMap {
        self.zone_map
            .get_or_init(move || {
                let zone_count = self.zone_count;
                let zones_eval = self
                    .lazy_children
                    .get(1)
                    .vortex_expect("failed to get zone child")
                    .projection_evaluation(
                        &(0..zone_count as u64),
                        &root(),
                        MaskFuture::new_true(zone_count),
                    )
                    .vortex_expect("Failed construct zone map evaluation");
                let session = self.session.clone();
                let zone_len = self.zone_len;
                let row_count = self.row_count;
                let dtype = self.dtype.clone();

                async move {
                    let mut ctx = session.create_execution_ctx();
                    let zones_array = zones_eval.await?.execute::<StructArray>(&mut ctx)?;
                    // SAFETY: zoned layout validation checked that this zones child was
                    // written from the same column dtype and stats-table schema.
                    Ok(unsafe { ZoneMap::new_unchecked(dtype, zones_array, zone_len, row_count) })
                }
                .map_err(Arc::new)
                .boxed()
                .shared()
            })
            .clone()
    }
}

pub(super) struct PruningResult {
    zone_map: ZoneMap,
    predicate: Expression,
    dynamic_updates: Option<DynamicExprUpdates>,
    latest_result: RwLock<(u64, Mask)>,
    session: VortexSession,
}

impl PruningResult {
    pub(super) fn mask(&self) -> VortexResult<Mask> {
        let Some(dynamic_updates) = &self.dynamic_updates else {
            return Ok(self.latest_result.read().1.clone());
        };

        let version = dynamic_updates.version();

        {
            let read_guard = self.latest_result.read();
            if read_guard.0 >= version {
                return Ok(read_guard.1.clone());
            }
        }

        let mut guard = self.latest_result.write();
        if guard.0 >= version {
            return Ok(guard.1.clone());
        }

        trace!(
            version,
            predicate = %self.predicate,
            "recomputing pruning mask"
        );

        let next_mask = self
            .zone_map
            .prune(&self.predicate, &self.session)
            .map_err(|err| {
                err.with_context(format!(
                    "While evaluating pruning predicate {}",
                    self.predicate
                ))
            })?;
        *guard = (version, next_mask.clone());

        Ok(next_mask)
    }
}
