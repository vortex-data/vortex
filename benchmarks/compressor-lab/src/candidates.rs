// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The candidates every chunk is compressed with, by name.
//!
//! - `production`: today's compressor, `BtrBlocksCompressorBuilder::from_session` (default
//!   encodings only).
//! - `sizemodel`: production with RunEnd and Sparse estimated from a size model.
//! - `forced/<scheme>`: `<scheme>` forced at the root, production below it.
//! - `exhaustive`: the [`oracle`](crate::oracle)'s Pareto front, stored as `exhaustive/<i>` from
//!   smallest to fastest.
//!
//! Every candidate produces a [`Plan`]. Its encoding is the production compressor following that
//! plan, so any encoding can be rebuilt from its plan alone, on any machine.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Context;
use anyhow::bail;
use parking_lot::Mutex;
use vortex::session::VortexSession;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::CompressionSessionExt;
use vortex_btrblocks::SchemeExt;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::plan::Plan;
use vortex_compressor::scheme::CompressionPolicy;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::scheme::SchemeId;
use vortex_compressor::scheme::SchemeSelection;
use vortex_compressor::scheme::SiteOutcome;
use vortex_compressor::scheme::SiteRequest;
use vortex_utils::aliases::hash_map::HashMap;

use crate::oracle::FrontPlan;
use crate::oracle::OracleParams;
use crate::oracle::ParetoOracle;
use crate::oracle::SearchStats;
use crate::oracle::SiteRow;
use crate::wrap::Mode;
use crate::wrap::Wrapped;

/// The integer schemes registered in the session, in registration order.
pub fn int_schemes(session: &VortexSession) -> Vec<&'static dyn Scheme> {
    let probe = Canonical::Primitive(PrimitiveArray::new(
        vortex::buffer::buffer![1i64, 2, 3],
        Validity::NonNullable,
    ));
    session
        .compression()
        .schemes()
        .iter()
        .copied()
        .filter(|s| s.matches(&probe))
        .collect()
}

/// The short name used in candidate names, e.g. `for` for `vortex.int.for`.
pub fn short_name(scheme: &dyn Scheme) -> &'static str {
    let name = scheme.scheme_name();
    name.rsplit('.').next().unwrap_or(name)
}

/// Production with every integer scheme replaced, keeping registration order, wrapping the ones
/// `wrap` picks.
pub fn replaced(
    session: &VortexSession,
    wrap: &dyn Fn(&'static dyn Scheme) -> Option<Mode>,
) -> BtrBlocksCompressor {
    let schemes = int_schemes(session);
    let ids: Vec<_> = schemes.iter().map(|s| s.id()).collect();
    let mut builder = BtrBlocksCompressorBuilder::from_session(session).exclude_schemes(ids);
    for scheme in &schemes {
        builder = builder.with_new_scheme(match wrap(*scheme) {
            Some(mode) => Wrapped::leak(*scheme, mode),
            None => *scheme,
        });
    }
    builder.build()
}

/// Builds the named candidates.
pub fn build(
    session: &VortexSession,
    names: &[String],
) -> anyhow::Result<BTreeMap<String, BtrBlocksCompressor>> {
    let schemes = int_schemes(session);
    names
        .iter()
        .map(|name| {
            let compressor = match name.as_str() {
                "production" => BtrBlocksCompressorBuilder::from_session(session).build(),
                "sizemodel" => replaced(session, &|_| Some(Mode::SizeModel)),
                other => {
                    let Some(short) = other.strip_prefix("forced/") else {
                        bail!("unknown candidate `{other}`");
                    };
                    let Some(scheme) = schemes.iter().find(|s| short_name(**s) == short) else {
                        bail!("no integer scheme named `{short}`");
                    };
                    BtrBlocksCompressorBuilder::from_session(session)
                        .with_new_scheme(Wrapped::leak(*scheme, Mode::ForceRoot))
                        .build()
                }
            };
            Ok((name.clone(), compressor))
        })
        .collect()
}

/// The name of the exhaustive oracle candidate in a plan.
pub const EXHAUSTIVE: &str = "exhaustive";

/// Every scheme registered in the session, for parsing plans.
pub fn all_schemes(session: &VortexSession) -> Vec<&'static dyn Scheme> {
    session.compression().schemes().to_vec()
}

/// Parses a stored plan against the session's schemes.
pub fn parse_plan(text: &str, session: &VortexSession) -> anyhow::Result<Plan> {
    Plan::parse(text, &all_schemes(session)).with_context(|| format!("parsing plan `{text}`"))
}

/// A candidate's plan, before it is compressed into a blob.
#[derive(Debug, Clone)]
pub struct ProducedPlan {
    /// The result name: the candidate name, or `exhaustive/<i>` for the oracle's front.
    pub name: String,
    /// The plan, as applied.
    pub plan: Plan,
    /// The oracle's predicted bytes and decode ns, for front plans.
    pub predicted: Option<(u64, u64)>,
}

/// What the exhaustive search reports beside its plans.
#[derive(Debug, Clone, Default)]
pub struct SearchOutput {
    /// The work done.
    pub stats: SearchStats,
    /// One row per (site, scheme) tried.
    pub sites: Vec<SiteRow>,
}

/// Builds the plans a candidate names.
///
/// `compressors` holds `production` and, if planned, `sizemodel`, as [`build`] makes them.
pub fn produce(
    name: &str,
    array: &ArrayRef,
    compressors: &BTreeMap<String, BtrBlocksCompressor>,
    session: &VortexSession,
    params: OracleParams,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<(Vec<ProducedPlan>, Option<SearchOutput>)> {
    let production = compressors
        .get("production")
        .context("the production compressor was not built")?;
    let single = |plan: Plan| {
        vec![ProducedPlan {
            name: name.to_string(),
            plan,
            predicted: None,
        }]
    };
    match name {
        "production" | "sizemodel" => {
            let compressor = compressors
                .get(name)
                .with_context(|| format!("candidate `{name}` was not built"))?;
            let (_, plan) = compressor.compress_recording_plan(array, ctx)?;
            Ok((single(plan), None))
        }
        EXHAUSTIVE => {
            let oracle = Arc::new(ParetoOracle::new(session.clone(), params));
            CascadingCompressor::clone(production)
                .with_policy(Arc::<ParetoOracle>::clone(&oracle))
                .compress(array, ctx)?;
            let stats = oracle.stats();
            let (front, sites) = oracle.front();
            let plans = front
                .into_iter()
                .enumerate()
                .map(
                    |(
                        i,
                        FrontPlan {
                            plan,
                            nbytes,
                            decode_ns,
                        },
                    )| ProducedPlan {
                        name: format!("{EXHAUSTIVE}/{i}"),
                        plan,
                        predicted: Some((nbytes, decode_ns)),
                    },
                )
                .collect();
            Ok((plans, Some(SearchOutput { stats, sites })))
        }
        other => {
            let Some(short) = other.strip_prefix("forced/") else {
                bail!("unknown candidate `{other}`");
            };
            let Some(scheme) = int_schemes(session)
                .into_iter()
                .find(|s| short_name(*s) == short)
            else {
                bail!("no integer scheme named `{short}`");
            };
            let policy = Arc::new(FollowAndRecord::new(Plan::Scheme {
                scheme: scheme.id(),
                children: Vec::new(),
            }));
            CascadingCompressor::clone(production)
                .with_policy(Arc::<FollowAndRecord>::clone(&policy))
                .compress(array, ctx)?;
            let plan = policy.plan();
            if !matches!(&plan, Plan::Scheme { scheme: applied, .. } if *applied == scheme.id()) {
                bail!("`{short}` does not apply to or shrink this chunk");
            }
            Ok((single(plan), None))
        }
    }
}

/// The scheme applied at each site, or `None` for canonical, by cascade history.
type Sites = HashMap<Vec<(SchemeId, usize)>, Option<SchemeId>>;

/// Follows a plan where it decides, estimates elsewhere, and records what was applied.
#[derive(Debug)]
struct FollowAndRecord {
    plan: Plan,
    sites: Mutex<Sites>,
}

impl FollowAndRecord {
    fn new(plan: Plan) -> Self {
        Self {
            plan,
            sites: Mutex::new(Sites::default()),
        }
    }

    fn plan(&self) -> Plan {
        recorded(&self.sites.lock(), &mut Vec::new())
    }
}

impl CompressionPolicy for FollowAndRecord {
    fn select_scheme(&self, request: &SiteRequest<'_>) -> SchemeSelection {
        match self.plan.at(request.ctx.cascade_history()) {
            Plan::Adaptive => SchemeSelection::Estimate,
            Plan::Canonical => SchemeSelection::Canonical,
            Plan::Scheme { scheme, .. } => SchemeSelection::Scheme(*scheme),
        }
    }

    fn observe(&self, outcome: &SiteOutcome<'_>) {
        self.sites
            .lock()
            .insert(outcome.ctx.cascade_history().to_vec(), outcome.scheme);
    }
}

fn recorded(sites: &Sites, path: &mut Vec<(SchemeId, usize)>) -> Plan {
    let Some(&applied) = sites.get(path.as_slice()) else {
        return Plan::Adaptive;
    };
    let Some(scheme) = applied else {
        return Plan::Canonical;
    };
    let len = sites
        .keys()
        .filter(|site| site.len() == path.len() + 1 && site.starts_with(path))
        .filter_map(|site| site.last())
        .filter(|(parent, _)| *parent == scheme)
        .map(|&(_, index)| index + 1)
        .max()
        .unwrap_or(0);
    let children = (0..len)
        .map(|index| {
            path.push((scheme, index));
            let child = recorded(sites, path);
            path.pop();
            child
        })
        .collect();
    Plan::Scheme { scheme, children }
}
