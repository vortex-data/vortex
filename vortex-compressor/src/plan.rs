// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Explicit compression plans: the scheme applied at every site of a cascade.
//!
//! A [`Plan`] makes a compression decision reproducible. [`RecordPlan`] is a
//! [`CompressionPolicy`] that records the plan the compressor applies, and [`FollowPlan`] is one
//! that replays a plan without estimating. Plans print as text and parse back against a list of
//! schemes, so they can be stored outside the process:
//!
//! ```text
//! vortex.int.runend(vortex.int.bitpacking, _)
//! ```
//!
//! `_` is [`Plan::Adaptive`] and `canonical` is [`Plan::Canonical`].

use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use crate::CascadingCompressor;
use crate::scheme::CompressionPolicy;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::scheme::SchemeSelection;
use crate::scheme::SiteOutcome;
use crate::scheme::SiteRequest;

/// The scheme applied at a compression site and, below it, at each of the scheme's children.
///
/// Plans apply to leaf arrays: booleans, primitives, decimals, strings and binary. The fields of
/// structs, lists and other nested arrays are compressed adaptively.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Plan {
    /// Let the compressor choose, as it does without a plan.
    Adaptive,
    /// Leave the array in its canonical encoding.
    Canonical,
    /// Compress the array with a scheme.
    Scheme {
        /// The scheme.
        scheme: SchemeId,
        /// The plans for the scheme's children, by child index. Children without an entry are
        /// compressed adaptively.
        children: Vec<Plan>,
    },
}

impl Plan {
    /// Returns the plan at the end of `path`, a cascade history of `(scheme, child index)` pairs,
    /// or [`Plan::Adaptive`] where this plan does not reach.
    pub fn at(&self, path: &[(SchemeId, usize)]) -> &Plan {
        let mut node = self;
        for &(parent, index) in path {
            match node {
                Plan::Scheme { scheme, children } if *scheme == parent => {
                    match children.get(index) {
                        Some(child) => node = child,
                        None => return &Plan::Adaptive,
                    }
                }
                _ => return &Plan::Adaptive,
            }
        }
        node
    }

    /// Parses a plan printed by its [`Display`](fmt::Display) implementation, resolving scheme
    /// names against `schemes`.
    ///
    /// # Errors
    ///
    /// Returns an error if the text is malformed or names a scheme not in `schemes`.
    pub fn parse(text: &str, schemes: &[&'static dyn Scheme]) -> VortexResult<Self> {
        let mut parser = Parser {
            rest: text,
            schemes,
        };
        let plan = parser.plan()?;
        parser.skip_space();
        if !parser.rest.is_empty() {
            vortex_bail!("unexpected `{}` after plan", parser.rest);
        }
        Ok(plan)
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Plan::Adaptive => f.write_str("_"),
            Plan::Canonical => f.write_str("canonical"),
            Plan::Scheme { scheme, children } => {
                write!(f, "{scheme}")?;
                if !children.is_empty() {
                    f.write_str("(")?;
                    for (index, child) in children.iter().enumerate() {
                        if index > 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{child}")?;
                    }
                    f.write_str(")")?;
                }
                Ok(())
            }
        }
    }
}

/// A recursive-descent parser over the text of a plan.
struct Parser<'a> {
    /// The text not parsed yet.
    rest: &'a str,
    /// The schemes that names resolve against.
    schemes: &'a [&'static dyn Scheme],
}

impl Parser<'_> {
    /// Skips leading whitespace.
    fn skip_space(&mut self) {
        self.rest = self.rest.trim_start();
    }

    /// Consumes `c` after any whitespace, returning whether it was there.
    fn eat(&mut self, c: char) -> bool {
        self.skip_space();
        match self.rest.strip_prefix(c) {
            Some(rest) => {
                self.rest = rest;
                true
            }
            None => false,
        }
    }

    /// Parses one plan.
    fn plan(&mut self) -> VortexResult<Plan> {
        self.skip_space();
        let end = self
            .rest
            .find(|c: char| c == '(' || c == ')' || c == ',' || c.is_whitespace())
            .unwrap_or(self.rest.len());
        let (name, rest) = self.rest.split_at(end);
        self.rest = rest;
        match name {
            "" => vortex_bail!("expected a plan at `{}`", self.rest),
            "_" => Ok(Plan::Adaptive),
            "canonical" => Ok(Plan::Canonical),
            name => {
                let scheme = self
                    .schemes
                    .iter()
                    .find(|s| s.scheme_name() == name)
                    .ok_or_else(|| vortex_err!("unknown scheme `{name}` in plan"))?
                    .id();
                let mut children = Vec::new();
                if self.eat('(') {
                    loop {
                        children.push(self.plan()?);
                        if self.eat(')') {
                            break;
                        }
                        if !self.eat(',') {
                            vortex_bail!("expected `,` or `)` at `{}`", self.rest);
                        }
                    }
                }
                Ok(Plan::Scheme { scheme, children })
            }
        }
    }
}

/// A policy that compresses every site as a [`Plan`] says, without estimating.
///
/// Where the plan is [`Plan::Adaptive`], or its scheme is not eligible, fails or does not make the
/// array smaller, the compressor estimates as usual.
#[derive(Debug)]
pub struct FollowPlan {
    /// The plan to follow.
    plan: Plan,
}

impl FollowPlan {
    /// A policy that follows `plan`.
    pub fn new(plan: Plan) -> Self {
        Self { plan }
    }
}

impl CompressionPolicy for FollowPlan {
    fn select_scheme(&self, request: &SiteRequest<'_>) -> SchemeSelection {
        match self.plan.at(request.ctx.cascade_history()) {
            Plan::Adaptive => SchemeSelection::Estimate,
            Plan::Canonical => SchemeSelection::Canonical,
            Plan::Scheme { scheme, .. } => SchemeSelection::Scheme(*scheme),
        }
    }
}

/// The scheme applied at each site, or `None` for canonical, by cascade history.
type Sites = HashMap<Vec<(SchemeId, usize)>, Option<SchemeId>>;

/// A policy that records the plan the compressor applies, and otherwise changes nothing.
#[derive(Debug, Default)]
pub struct RecordPlan {
    /// The scheme applied at each site, by cascade history.
    sites: Mutex<Sites>,
}

impl RecordPlan {
    /// A recorder with nothing recorded.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the plan applied to the array compressed with this recorder.
    ///
    /// Use a new recorder for each array: the sites of arrays compressed one after another
    /// overwrite each other.
    pub fn plan(&self) -> Plan {
        let sites = self.sites.lock();
        build(&sites, &mut Vec::new())
    }
}

/// Builds the plan recorded at `path` and below.
fn build(sites: &Sites, path: &mut Vec<(SchemeId, usize)>) -> Plan {
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
            let child = build(sites, path);
            path.pop();
            child
        })
        .collect();
    Plan::Scheme { scheme, children }
}

impl CompressionPolicy for RecordPlan {
    fn observe(&self, outcome: &SiteOutcome<'_>) {
        self.sites
            .lock()
            .insert(outcome.ctx.cascade_history().to_vec(), outcome.scheme);
    }
}

impl CascadingCompressor {
    /// Compresses an array as `plan` says, replacing any policy this compressor has.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress_with_plan(
        &self,
        array: &ArrayRef,
        plan: &Plan,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.clone()
            .with_policy(Arc::new(FollowPlan::new(plan.clone())))
            .compress(array, exec_ctx)
    }

    /// Compresses an array as [`compress`](Self::compress) does, and returns the plan it applied,
    /// replacing any policy this compressor has.
    ///
    /// Following the plan with [`compress_with_plan`](Self::compress_with_plan) reproduces the
    /// compressed array.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress_recording_plan(
        &self,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, Plan)> {
        let recorder = Arc::new(RecordPlan::new());
        let compressed = self
            .clone()
            .with_policy(Arc::<RecordPlan>::clone(&recorder))
            .compress(array, exec_ctx)?;
        Ok((compressed, recorder.plan()))
    }
}
