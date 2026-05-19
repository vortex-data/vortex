// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::{VortexResult, vortex_bail};

use crate::form::{Form, PType};
use crate::stage::{JitStage, Placement};

/// A typed, validated chain of stages ready for compilation.
#[derive(Debug)]
pub struct Pipeline {
    in_block: Vec<Arc<dyn JitStage>>,
    post_loop: Vec<Arc<dyn JitStage>>,
    in_ptype: PType,
    out_ptype: PType,
    block_size: usize,
}

impl Pipeline {
    pub fn new(in_ptype: PType, block_size: usize) -> Self {
        Self {
            in_block: Vec::new(),
            post_loop: Vec::new(),
            in_ptype,
            out_ptype: in_ptype,
            block_size,
        }
    }

    pub fn in_ptype(&self) -> PType {
        self.in_ptype
    }
    pub fn out_ptype(&self) -> PType {
        self.out_ptype
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn in_block_stages(&self) -> &[Arc<dyn JitStage>] {
        &self.in_block
    }
    pub fn post_loop_stages(&self) -> &[Arc<dyn JitStage>] {
        &self.post_loop
    }

    /// Append a stage. Validates form compatibility against the previous
    /// stage's output and updates the running output ptype.
    pub fn push(&mut self, stage: Arc<dyn JitStage>) -> VortexResult<()> {
        match stage.placement() {
            Placement::InBlock => self.push_in_block(stage),
            Placement::PostLoop => self.push_post_loop(stage),
        }
    }

    fn push_in_block(&mut self, stage: Arc<dyn JitStage>) -> VortexResult<()> {
        // Each in-block stage either:
        //   - has empty input (leaf), or
        //   - has one input whose Form is compatible with the previous output.
        if !self.post_loop.is_empty() {
            vortex_bail!("cannot add InBlock stage after PostLoop stages");
        }
        let input = stage.input();
        match (self.in_block.last(), input.len()) {
            (None, 0) => {}
            (None, _) => vortex_bail!(
                "first stage {} must be a leaf (empty input)",
                stage.tag()
            ),
            (Some(prev), 1) => {
                let prev_out = prev.output();
                if !prev_out.compatible(input[0]) {
                    vortex_bail!(
                        "incompatible forms between {} ({:?}) and {} ({:?})",
                        prev.tag(),
                        prev_out,
                        stage.tag(),
                        input[0]
                    );
                }
            }
            (Some(prev), 0) => vortex_bail!(
                "stage {} declares empty input but follows {}",
                stage.tag(),
                prev.tag()
            ),
            (Some(_), n) => vortex_bail!(
                "v0 supports single-input stages only; {} declared {} inputs",
                stage.tag(),
                n
            ),
        }
        if let Some(pt) = stage.output().ptype() {
            self.out_ptype = pt;
        }
        self.in_block.push(stage);
        Ok(())
    }

    fn push_post_loop(&mut self, stage: Arc<dyn JitStage>) -> VortexResult<()> {
        // PostLoop stages must be terminal-shaped: empty input, Form::None output.
        if !matches!(stage.output(), Form::None) {
            vortex_bail!(
                "PostLoop stage {} must declare output = Form::None",
                stage.tag()
            );
        }
        self.post_loop.push(stage);
        Ok(())
    }
}

/// Recursive description of an encoding tree.
///
/// v0 doesn't actually use the recursion (the test cases build `Pipeline`
/// directly), but the shape is here so the framework's lowering story stays
/// honest with the §9 design.
#[derive(Debug, Clone)]
pub struct DecodeNode {
    pub stage: Arc<dyn JitStage>,
    pub children: Vec<DecodeNode>,
}

impl DecodeNode {
    pub fn leaf(stage: Arc<dyn JitStage>) -> Self {
        Self {
            stage,
            children: Vec::new(),
        }
    }

    pub fn parent(stage: Arc<dyn JitStage>, children: Vec<DecodeNode>) -> Self {
        Self { stage, children }
    }

    /// Lower a single-child chain into a `Pipeline`. Multi-child tree lowering
    /// is left to encoding-specific `lower` impls (see §9 design discussion).
    pub fn lower_chain(self, in_ptype: PType, block_size: usize) -> VortexResult<Pipeline> {
        let mut p = Pipeline::new(in_ptype, block_size);
        let mut node = self;
        let mut stack = Vec::new();
        while !node.children.is_empty() {
            if node.children.len() != 1 {
                vortex_bail!("DecodeNode::lower_chain only supports linear chains");
            }
            stack.push(node.stage.clone());
            node = node.children.into_iter().next().unwrap();
        }
        // node is now the leaf; emit leaf, then unwind the stack from inside-out.
        p.push(node.stage)?;
        while let Some(s) = stack.pop() {
            p.push(s)?;
        }
        Ok(p)
    }
}
