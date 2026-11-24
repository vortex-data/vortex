// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The exhaustive whole-cascade oracle: a Pareto front of plans on (bytes, decode time).
//!
//! [`ParetoOracle`] is a [`CompressionPolicy`] that asks the compressor to try every eligible
//! scheme at every site ([`SchemeSelection::Best`]) and observes every candidate it builds. After
//! one compression of a chunk it holds, for every site, the bytes and decode time of canonical and
//! of each scheme. [`ParetoOracle::front`] then combines them bottom-up into a front of whole
//! plans:
//!
//! - A candidate's own cost is its measured total minus the totals of the children the compressor
//!   kept below it, so `own + Σ child` is exact for bytes and approximate for decode time.
//! - Each candidate's front is the product of its children's fronts, thinned to the Pareto set
//!   with an ε margin on decode time and capped at `k` plans.
//! - A site's front is the union over its candidates, thinned again.
//!
//! The search compresses each site once per scheme, so its cost grows with the cascade. Two things
//! bound it. Sites whose array, depth and eligible schemes match a site already searched reuse its
//! result instead of searching again. After `max_compressions` candidates, the remaining sites
//! fall back to the compressor's estimates, and the result is marked capped.
//!
//! The front's plans are only candidates: the time stage measures each whole plan on a quiet
//! machine, and analysis picks the cheapest under any cost.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;
use std::hint::black_box;
use std::time::Instant;

use parking_lot::Mutex;
use serde::Deserialize;
use serde::Serialize;
use vortex::session::VortexSession;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::EqMode;
use vortex_array::VortexSessionExecute;
use vortex_compressor::plan::Plan;
use vortex_compressor::scheme::CompressionPolicy;
use vortex_compressor::scheme::SchemeId;
use vortex_compressor::scheme::SchemeSelection;
use vortex_compressor::scheme::SiteOutcome;
use vortex_compressor::scheme::SiteRequest;
use vortex_utils::aliases::hash_map::HashMap;

/// A cascade history: `(parent scheme, child index)` pairs from the root.
type Path = Vec<(SchemeId, usize)>;

/// Search settings.
#[derive(Debug, Clone, Copy)]
pub struct OracleParams {
    /// Plans kept per front.
    pub k: usize,
    /// Relative margin on decode time: a plan is kept only if it decodes faster than every
    /// smaller plan by more than this.
    pub epsilon: f64,
    /// Candidates compressed per chunk before the remaining sites fall back to estimates.
    pub max_compressions: u64,
    /// Decode repetitions per candidate; the fastest is kept.
    pub decode_reps: u32,
}

/// How much work one search did.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchStats {
    /// Sites the compressor asked about.
    pub sites: u64,
    /// Candidates compressed and measured.
    pub compressions: u64,
    /// Sites that reused an earlier identical site.
    pub memo_hits: u64,
    /// Sites left to the compressor's estimates because the budget ran out.
    pub capped_sites: u64,
    /// Wall time of the search, excluding building the front.
    pub search_ms: u64,
}

/// One candidate at one site, as the per-site table records it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SiteRow {
    /// The site, as `scheme.child/scheme.child`, empty for the root.
    pub path: String,
    /// The scheme tried, or `canonical`.
    pub scheme: String,
    /// Whether the compressor kept this candidate (smallest by bytes).
    pub won: bool,
    /// Bytes of the candidate as compressed, with the compressor's choice of children.
    pub nbytes: u64,
    /// Fastest decode of the candidate as compressed, ns.
    pub decode_ns: u64,
    /// Smallest bytes over the candidate's front: the best subtree below this scheme.
    pub best_nbytes: u64,
    /// Fastest decode over the candidate's front, ns.
    pub best_decode_ns: u64,
}

/// A plan on the front, with its predicted cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrontPlan {
    /// The plan.
    pub plan: Plan,
    /// Predicted bytes.
    pub nbytes: u64,
    /// Predicted decode time, ns.
    pub decode_ns: u64,
}

#[derive(Debug, Clone, Copy)]
struct Observed {
    scheme: Option<SchemeId>,
    nbytes: u64,
    decode_ns: u64,
}

#[derive(Debug, Default)]
struct State {
    candidates: HashMap<Path, Vec<Observed>>,
    winners: HashMap<Path, Observed>,
    alias: HashMap<Path, Path>,
    memo: HashMap<u64, Path>,
    stats: SearchStats,
    started: Option<Instant>,
}

/// The exhaustive Pareto search, as a compression policy. Use one per chunk.
#[derive(Debug)]
pub struct ParetoOracle {
    session: VortexSession,
    params: OracleParams,
    state: Mutex<State>,
}

impl ParetoOracle {
    /// A search with nothing observed yet.
    pub fn new(session: VortexSession, params: OracleParams) -> Self {
        Self {
            session,
            params,
            state: Mutex::new(State::default()),
        }
    }

    fn decode_ns(&self, array: &ArrayRef) -> u64 {
        let mut ctx = self.session.create_execution_ctx();
        let mut fastest = u64::MAX;
        for _ in 0..self.params.decode_reps.max(1) {
            let start = Instant::now();
            let Ok(decoded) = array.clone().execute::<Canonical>(&mut ctx) else {
                return u64::MAX;
            };
            fastest = fastest.min(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX));
            drop(black_box(decoded));
        }
        fastest
    }

    /// Returns the work done, with the search time measured up to now.
    pub fn stats(&self) -> SearchStats {
        let state = self.state.lock();
        let mut stats = state.stats.clone();
        if let Some(started) = state.started {
            stats.search_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        }
        stats
    }

    /// Builds the root front and the per-site table from what the compression observed.
    pub fn front(&self) -> (Vec<FrontPlan>, Vec<SiteRow>) {
        let state = self.state.lock();
        let mut builder = FrontBuilder {
            state: &state,
            params: self.params,
            done: HashMap::default(),
            rows: Vec::new(),
        };
        let front = builder.front(&Vec::new(), 0);
        let mut rows = builder.rows;
        rows.sort_by(|a, b| (&a.path, &a.scheme).cmp(&(&b.path, &b.scheme)));
        let front = front
            .into_iter()
            .map(|p| FrontPlan {
                plan: p.plan,
                nbytes: p.nbytes,
                decode_ns: p.decode_ns,
            })
            .collect();
        (front, rows)
    }
}

fn site_key(request: &SiteRequest<'_>) -> u64 {
    let mut hasher = DefaultHasher::new();
    request.array.array_hash(&mut hasher, EqMode::Value);
    request.ctx.cascade_history().len().hash(&mut hasher);
    request.eligible.hash(&mut hasher);
    hasher.finish()
}

impl CompressionPolicy for ParetoOracle {
    fn select_scheme(&self, request: &SiteRequest<'_>) -> SchemeSelection {
        let path = request.ctx.cascade_history().to_vec();
        let key = site_key(request);
        {
            let mut state = self.state.lock();
            state.started.get_or_insert_with(Instant::now);
            state.stats.sites += 1;
            if let Some(original) = state.memo.get(&key).cloned()
                && original != path
                && let Some(winner) = state.winners.get(&original).copied()
            {
                state.alias.insert(path, original);
                state.stats.memo_hits += 1;
                return match winner.scheme {
                    Some(scheme) => SchemeSelection::Scheme(scheme),
                    None => SchemeSelection::Canonical,
                };
            }
            if state.stats.compressions >= self.params.max_compressions {
                state.stats.capped_sites += 1;
                return SchemeSelection::Estimate;
            }
            state.memo.entry(key).or_insert_with(|| path.clone());
        }
        let canonical = Observed {
            scheme: None,
            nbytes: request.array.nbytes(),
            decode_ns: self.decode_ns(request.array),
        };
        self.state
            .lock()
            .candidates
            .entry(path)
            .or_default()
            .push(canonical);
        SchemeSelection::Best(Vec::new())
    }

    fn observe_candidate(&self, candidate: &SiteOutcome<'_>) {
        let observed = Observed {
            scheme: candidate.scheme,
            nbytes: candidate.array.nbytes(),
            decode_ns: self.decode_ns(candidate.array),
        };
        let mut state = self.state.lock();
        state.stats.compressions += 1;
        state
            .candidates
            .entry(candidate.ctx.cascade_history().to_vec())
            .or_default()
            .push(observed);
    }

    fn observe(&self, outcome: &SiteOutcome<'_>) {
        let path = outcome.ctx.cascade_history().to_vec();
        let known = {
            let state = self.state.lock();
            state.candidates.get(&path).and_then(|candidates| {
                candidates
                    .iter()
                    .find(|c| c.scheme == outcome.scheme && c.nbytes == outcome.array.nbytes())
                    .copied()
            })
        };
        let observed = known.unwrap_or_else(|| Observed {
            scheme: outcome.scheme,
            nbytes: outcome.array.nbytes(),
            decode_ns: self.decode_ns(outcome.array),
        });
        self.state.lock().winners.insert(path, observed);
    }
}

#[derive(Debug, Clone)]
struct Point {
    nbytes: u64,
    decode_ns: u64,
    plan: Plan,
}

struct FrontBuilder<'a> {
    state: &'a State,
    params: OracleParams,
    done: HashMap<Path, Vec<Point>>,
    rows: Vec<SiteRow>,
}

impl FrontBuilder<'_> {
    fn front(&mut self, path: &Path, depth: usize) -> Vec<Point> {
        if let Some(front) = self.done.get(path) {
            return front.clone();
        }
        let front = match self.state.alias.get(path) {
            Some(original) if depth < 64 => self.front(&original.clone(), depth + 1),
            _ => self.search_front(path, depth),
        };
        self.done.insert(path.clone(), front.clone());
        front
    }

    fn search_front(&mut self, path: &Path, depth: usize) -> Vec<Point> {
        let winner = self.state.winners.get(path).copied();
        let Some(candidates) = self.state.candidates.get(path) else {
            // Decided by estimates (budget) or not searched: a single point, as the compressor
            // left it.
            return winner
                .map(|w| {
                    vec![Point {
                        nbytes: w.nbytes,
                        decode_ns: w.decode_ns,
                        plan: self.winner_plan(path, depth),
                    }]
                })
                .unwrap_or_default();
        };
        let mut site = Vec::new();
        for candidate in candidates.clone() {
            let Some(scheme) = candidate.scheme else {
                site.push(Point {
                    nbytes: candidate.nbytes,
                    decode_ns: candidate.decode_ns,
                    plan: Plan::Canonical,
                });
                self.row(path, &candidate, winner, &site[site.len() - 1..]);
                continue;
            };
            let children = self.children(path, scheme);
            let mut own_bytes = candidate.nbytes;
            let mut own_decode = candidate.decode_ns;
            for child in children.values() {
                if let Some(w) = self.state.winners.get(child) {
                    own_bytes = own_bytes.saturating_sub(w.nbytes);
                    own_decode = own_decode.saturating_sub(w.decode_ns);
                }
            }
            let len = children.keys().max().map_or(0, |i| i + 1);
            let mut combos = vec![Point {
                nbytes: own_bytes,
                decode_ns: own_decode,
                plan: Plan::Scheme {
                    scheme,
                    children: vec![Plan::Adaptive; len],
                },
            }];
            for (&index, child) in &children {
                let child_front = self.front(child, depth + 1);
                if child_front.is_empty() {
                    continue;
                }
                let mut next = Vec::with_capacity(combos.len() * child_front.len());
                for combo in &combos {
                    for option in &child_front {
                        let mut plan = combo.plan.clone();
                        if let Plan::Scheme { children, .. } = &mut plan {
                            children[index] = option.plan.clone();
                        }
                        next.push(Point {
                            nbytes: combo.nbytes.saturating_add(option.nbytes),
                            decode_ns: combo.decode_ns.saturating_add(option.decode_ns),
                            plan,
                        });
                    }
                }
                combos = thin(next, self.params);
            }
            self.row(path, &candidate, winner, &combos);
            site.extend(combos);
        }
        thin(site, self.params)
    }

    /// The child sites observed below `scheme` at `path`, by child index.
    fn children(&self, path: &Path, scheme: SchemeId) -> BTreeMap<usize, Path> {
        let keys = self.state.winners.keys().chain(self.state.alias.keys());
        let mut children = BTreeMap::new();
        for key in keys {
            if key.len() == path.len() + 1
                && key.starts_with(path)
                && let Some(&(parent, index)) = key.last()
                && parent == scheme
            {
                children.insert(index, key.clone());
            }
        }
        children
    }

    /// The plan the compressor applied at `path` and below.
    fn winner_plan(&self, path: &Path, depth: usize) -> Plan {
        if depth > 64 {
            return Plan::Adaptive;
        }
        let path = self.state.alias.get(path).unwrap_or(path);
        match self.state.winners.get(path).and_then(|w| w.scheme) {
            None if self.state.winners.contains_key(path) => Plan::Canonical,
            None => Plan::Adaptive,
            Some(scheme) => {
                let children = self.children(path, scheme);
                let len = children.keys().max().map_or(0, |i| i + 1);
                let mut plans = vec![Plan::Adaptive; len];
                for (index, child) in children {
                    plans[index] = self.winner_plan(&child, depth + 1);
                }
                Plan::Scheme {
                    scheme,
                    children: plans,
                }
            }
        }
    }

    fn row(
        &mut self,
        path: &Path,
        candidate: &Observed,
        winner: Option<Observed>,
        front: &[Point],
    ) {
        self.rows.push(SiteRow {
            path: path_string(path),
            scheme: candidate
                .scheme
                .map_or_else(|| "canonical".to_string(), |s| s.to_string()),
            won: winner.is_some_and(|w| w.scheme == candidate.scheme),
            nbytes: candidate.nbytes,
            decode_ns: candidate.decode_ns,
            best_nbytes: front
                .iter()
                .map(|p| p.nbytes)
                .min()
                .unwrap_or(candidate.nbytes),
            best_decode_ns: front
                .iter()
                .map(|p| p.decode_ns)
                .min()
                .unwrap_or(candidate.decode_ns),
        });
    }
}

fn path_string(path: &Path) -> String {
    path.iter()
        .map(|(scheme, index)| format!("{scheme}.{index}"))
        .collect::<Vec<_>>()
        .join("/")
}

/// Keeps the Pareto set on (bytes, decode), with an ε margin on decode, and at most `k` plans
/// spread evenly from smallest to fastest.
fn thin(mut points: Vec<Point>, params: OracleParams) -> Vec<Point> {
    points.sort_by(|a, b| {
        (a.nbytes, a.decode_ns)
            .cmp(&(b.nbytes, b.decode_ns))
            .then_with(|| a.plan.to_string().cmp(&b.plan.to_string()))
    });
    let mut kept: Vec<Point> = Vec::new();
    let mut seen = BTreeSet::new();
    for point in points {
        let faster = kept.last().is_none_or(|last| {
            (point.decode_ns as f64) * (1.0 + params.epsilon) < last.decode_ns as f64
        });
        if faster && seen.insert(point.plan.to_string()) {
            kept.push(point);
        }
    }
    let k = params.k.max(1);
    if kept.len() <= k {
        return kept;
    }
    let last = kept.len() - 1;
    let picks: BTreeSet<usize> = (0..k)
        .map(|i| if k == 1 { 0 } else { i * last / (k - 1) })
        .collect();
    kept.into_iter()
        .enumerate()
        .filter(|(i, _)| picks.contains(i))
        .map(|(_, p)| p)
        .collect()
}

#[cfg(test)]
mod tests {
    use vortex::VortexSessionDefault;
    use vortex_btrblocks::SchemeExt;

    use super::*;

    fn point(nbytes: u64, decode_ns: u64, tag: usize) -> Point {
        Point {
            nbytes,
            decode_ns,
            plan: Plan::Scheme {
                scheme: crate::candidates::int_schemes(&VortexSession::default())[tag].id(),
                children: Vec::new(),
            },
        }
    }

    fn params(k: usize, epsilon: f64) -> OracleParams {
        OracleParams {
            k,
            epsilon,
            max_compressions: 0,
            decode_reps: 1,
        }
    }

    #[test]
    fn thin_keeps_the_pareto_set() {
        let points = vec![
            point(100, 50, 0),
            point(90, 60, 1),
            point(120, 40, 2),
            point(130, 45, 3), // dominated by (120, 40)
        ];
        let kept = thin(points, params(8, 0.0));
        let costs: Vec<_> = kept.iter().map(|p| (p.nbytes, p.decode_ns)).collect();
        assert_eq!(costs, vec![(90, 60), (100, 50), (120, 40)]);
    }

    #[test]
    fn thin_applies_epsilon_and_k() {
        let points = vec![point(90, 100, 0), point(100, 98, 1), point(110, 50, 2)];
        let kept = thin(points.clone(), params(8, 0.1));
        let costs: Vec<_> = kept.iter().map(|p| (p.nbytes, p.decode_ns)).collect();
        assert_eq!(costs, vec![(90, 100), (110, 50)]);
        assert_eq!(thin(points, params(1, 0.0)).len(), 1);
    }
}

#[cfg(test)]
mod replay_tests {
    use std::collections::BTreeMap;

    use vortex::VortexSessionDefault;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;

    use super::*;

    #[test]
    fn front_plans_replay_to_their_predicted_bytes() -> anyhow::Result<()> {
        let session = VortexSession::default();
        let mut ctx = session.create_execution_ctx();
        // Long runs of a few values: RunEnd, Dict and bit-packing all compete.
        let array =
            PrimitiveArray::from_iter((0..65_536i64).map(|i| (i / 300) % 7 * 1000)).into_array();
        let compressors: BTreeMap<_, _> =
            crate::candidates::build(&session, &["production".to_string()])?;
        let params = OracleParams {
            k: 8,
            epsilon: 0.0,
            max_compressions: 10_000,
            decode_reps: 1,
        };
        let (plans, output) = crate::candidates::produce(
            crate::candidates::EXHAUSTIVE,
            &array,
            &compressors,
            &session,
            params,
            &mut ctx,
        )?;
        let output = output.expect("the oracle reports its search");
        assert!(output.stats.compressions > 0);
        assert!(!output.sites.is_empty());
        assert!(!plans.is_empty());

        let production = &compressors["production"];
        let (_, production_plan) = production.compress_recording_plan(&array, &mut ctx)?;
        let production_bytes = production
            .compress_with_plan(&array, &production_plan, &mut ctx)?
            .nbytes();
        for produced in &plans {
            let text = produced.plan.to_string();
            let parsed = crate::candidates::parse_plan(&text, &session)?;
            assert_eq!(parsed, produced.plan);
            let replayed = production.compress_with_plan(&array, &parsed, &mut ctx)?;
            let (predicted, _) = produced.predicted.expect("front plans carry predictions");
            assert_eq!(replayed.nbytes(), predicted, "plan {text}");
        }
        let smallest = plans.iter().map(|p| p.predicted.unwrap().0).min().unwrap();
        assert!(smallest <= production_bytes);
        Ok(())
    }
}
