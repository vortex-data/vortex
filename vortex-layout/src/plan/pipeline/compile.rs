// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compiling plans into pipelines, and counting the readers of each segment so segments read
//! more than once are decoded once.

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use super::Operator;
use super::Source;
use super::port::PortId;
use super::port::Reader;
use super::scan::Core;
use crate::plan::PlanRef;

/// A pipeline under construction: a source and the stages after it, not yet given outlets, so a
/// parent plan can add stages before it becomes a pipeline.
pub struct Chain {
    pub(crate) source: Box<dyn Source>,
    pub(crate) stages: Vec<Box<dyn Operator>>,
    pub(crate) inlets: Vec<PortId>,
}

impl Chain {
    /// A chain of `source` alone.
    pub fn new(source: impl Source + 'static) -> Self {
        Self {
            source: Box::new(source),
            stages: Vec::new(),
            inlets: Vec::new(),
        }
    }

    /// This chain with `stage` after its last stage.
    pub fn with(mut self, stage: impl Operator + 'static) -> Self {
        self.stages.push(Box::new(stage));
        self
    }
}

impl Core {
    /// Compiles `plan` over `rows`, restricted to `mask`. See [`Compiler::compile`].
    pub(crate) fn compile(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Option<Chain>> {
        Compiler { core: self }.compile(plan, rows, mask)
    }
}

/// What a plan compiles its children through. See
/// [`PlanVTable::compile`](crate::plan::PlanVTable::compile).
pub struct Compiler<'a> {
    pub(crate) core: &'a mut Core,
}

impl Compiler<'_> {
    /// Compiles `plan` over `rows` of its domain, restricted to `mask`, into a chain producing
    /// the rows `mask` selects, in order, or `None` when that is no row.
    pub fn compile(
        &mut self,
        plan: &PlanRef,
        rows: Range<u64>,
        mask: &Mask,
    ) -> VortexResult<Option<Chain>> {
        if rows.start >= rows.end {
            return Ok(None);
        }
        plan.compile(rows, mask, self)
    }

    /// A chain reading `chains` through `source`: each chain becomes a pipeline writing a new
    /// port, inlet `i` of `source` reading chain `i`, with the capacity `source` asks for.
    pub fn join(&mut self, chains: Vec<Chain>, source: impl Source + 'static) -> Chain {
        let inlets = chains
            .into_iter()
            .enumerate()
            .map(|(index, chain)| {
                let port = self
                    .core
                    .arena
                    .create(source.capacity(index), None, Reader::Unclaimed);
                self.core.add_pipeline(chain, smallvec::smallvec![port]);
                port
            })
            .collect();
        Chain {
            source: Box::new(source),
            stages: Vec::new(),
            inlets,
        }
    }

    /// The global row index of the scanned plan's row zero.
    pub fn row_offset(&self) -> u64 {
        self.core.row_offset
    }

    /// The session used for decoding and evaluation.
    pub fn session(&self) -> &VortexSession {
        &self.core.session
    }
}

/// How a plan's rows map to the rows of the plan a scan's splits range over.
#[derive(Clone, Debug)]
pub enum Reach {
    /// Row `r` is scanned row `r + offset`.
    Offset(u64),
    /// Every row is read for these scanned rows, as a dictionary's values are for its codes.
    Fixed(Range<u64>),
}

impl Reach {
    /// The scanned rows `rows` are read for.
    pub fn root(&self, rows: &Range<u64>) -> Range<u64> {
        match self {
            Reach::Offset(offset) => rows.start + offset..rows.end + offset,
            Reach::Fixed(root) => root.clone(),
        }
    }

    /// The mapping of a child whose row zero is this plan's row `by`.
    pub fn shift(&self, by: u64) -> Self {
        match self {
            Reach::Offset(offset) => Reach::Offset(offset + by),
            Reach::Fixed(root) => Reach::Fixed(root.clone()),
        }
    }

    /// The mapping of a child read whole for `rows` of this plan, as dictionary values are.
    pub fn fixed(&self, rows: &Range<u64>) -> Self {
        Reach::Fixed(self.root(rows))
    }
}
