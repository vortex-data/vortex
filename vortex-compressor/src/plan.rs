// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Explicit compression plans.
//!
//! A [`Plan`] fixes the scheme that the [`CascadingCompressor`] applies at each compression site
//! of a cascade, instead of letting the compressor estimate and choose. Plans make a compression
//! decision reproducible, let the searches in [`crate::search`] evaluate candidate cascades, and
//! let a plan trained on one sample be replayed on new data without sampling.
//!
//! [`CascadingCompressor`]: crate::CascadingCompressor

use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use rand::prelude::StdRng;

use crate::scheme::SchemeId;
use crate::search::CostModel;

/// An explicit compression decision for one array and its cascaded children.
///
/// Plans apply to leaf arrays: booleans, primitives, decimals, strings, binary and extension
/// arrays. The fields of structs, lists and other nested arrays are compressed adaptively.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Plan {
    /// Let the compressor choose the scheme for this array and its subtree.
    Adaptive,
    /// Leave the array in its canonical encoding.
    Canonical,
    /// The compressor's built-in constant encoding.
    ///
    /// The compressor detects constant arrays itself before it consults a plan, so this only
    /// appears in recorded plans. Following it on a non-constant array selects adaptively.
    Constant,
    /// Compress the array with a scheme.
    Scheme {
        /// The scheme to compress the array with.
        scheme: SchemeId,
        /// The plans for the scheme's cascaded children, indexed by the child index the scheme
        /// passes to [`CascadingCompressor::compress_child`]. Children without an entry are
        /// compressed adaptively.
        ///
        /// [`CascadingCompressor::compress_child`]: crate::CascadingCompressor::compress_child
        children: Arc<[Plan]>,
    },
}

impl Plan {
    /// Returns a plan that compresses with `scheme` and its children with `children`.
    pub fn scheme(scheme: SchemeId, children: impl Into<Arc<[Plan]>>) -> Self {
        Self::Scheme {
            scheme,
            children: children.into(),
        }
    }

    /// Returns the scheme this plan applies at its root, if any.
    pub fn scheme_id(&self) -> Option<SchemeId> {
        match self {
            Self::Scheme { scheme, .. } => Some(*scheme),
            Self::Adaptive | Self::Canonical | Self::Constant => None,
        }
    }

    /// Returns the plans for the root scheme's children, or an empty slice if the root is not a
    /// scheme.
    pub fn children(&self) -> &[Plan] {
        match self {
            Self::Scheme { children, .. } => children,
            Self::Adaptive | Self::Canonical | Self::Constant => &[],
        }
    }

    /// Returns a copy of this plan with the node at `path` replaced by `replacement`.
    ///
    /// `path` lists the child indices leading from the root to the node. Steps that do not exist
    /// in this plan leave it unchanged.
    pub(crate) fn replace_at(&self, path: &[usize], replacement: Plan) -> Plan {
        let Some((&index, rest)) = path.split_first() else {
            return replacement;
        };
        match self {
            Self::Scheme { scheme, children } if index < children.len() => {
                let mut children = children.to_vec();
                children[index] = children[index].replace_at(rest, replacement);
                Self::scheme(*scheme, children)
            }
            _ => self.clone(),
        }
    }

    /// Returns every node that was decided rather than left adaptive, in depth-first order.
    pub(crate) fn decided_nodes(&self) -> Vec<PlanNode<'_>> {
        let mut nodes = Vec::new();
        self.collect_decided_nodes(&mut Vec::new(), None, &mut nodes);
        nodes
    }

    /// Recursive helper for [`decided_nodes`](Self::decided_nodes).
    fn collect_decided_nodes<'a>(
        &'a self,
        path: &mut Vec<usize>,
        edge: Option<(SchemeId, usize)>,
        nodes: &mut Vec<PlanNode<'a>>,
    ) {
        if matches!(self, Self::Adaptive) {
            return;
        }
        nodes.push(PlanNode {
            path: path.clone(),
            edge,
            plan: self,
        });
        if let Self::Scheme { scheme, children } = self {
            for (index, child) in children.iter().enumerate() {
                path.push(index);
                child.collect_decided_nodes(path, Some((*scheme, index)), nodes);
                path.pop();
            }
        }
    }
}

impl fmt::Display for Plan {
    /// Formats the plan as `scheme(child0, child1, ...)`, with `*` for adaptive sites.
    ///
    /// A scheme whose children are all adaptive is written without parentheses.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adaptive => f.write_str("*"),
            Self::Canonical => f.write_str("canonical"),
            Self::Constant => f.write_str("constant"),
            Self::Scheme { scheme, children } => {
                write!(f, "{scheme}")?;
                if children.iter().all(|child| matches!(child, Self::Adaptive)) {
                    return Ok(());
                }
                f.write_str("(")?;
                for (index, child) in children.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{child}")?;
                }
                f.write_str(")")
            }
        }
    }
}

/// A decided node of a [`Plan`], with its position in the tree.
#[derive(Debug, Clone)]
pub(crate) struct PlanNode<'a> {
    /// The child indices leading from the root to this node.
    pub(crate) path: Vec<usize>,
    /// The parent scheme and child index this node hangs from, or `None` for the root.
    pub(crate) edge: Option<(SchemeId, usize)>,
    /// The subtree rooted at this node.
    pub(crate) plan: &'a Plan,
}

/// How the compressor chooses a scheme at a compression site.
#[derive(Debug, Clone)]
pub(crate) enum Selection {
    /// Choose by estimated compression ratio. This is the compressor's default.
    Estimate,
    /// Choose uniformly at random among canonical and the schemes that can compress the array.
    Random(Arc<Mutex<StdRng>>),
    /// Follow `plan`, and switch to `fallback` wherever the plan is adaptive or cannot be applied.
    Follow {
        /// The plan for the current compression site.
        plan: Plan,
        /// The selection used where the plan does not decide.
        fallback: Arc<Selection>,
    },
    /// Try canonical and every scheme, each with the cheapest plan for its children, and keep
    /// whichever is cheapest under `cost`.
    Exhaustive {
        /// The cost each candidate is ranked by.
        cost: CostModel,
        /// How many times each timing is repeated.
        iterations: usize,
    },
}

impl Selection {
    /// Returns a selection that follows `plan` and falls back to `fallback`.
    pub(crate) fn follow(plan: Plan, fallback: Selection) -> Self {
        Self::Follow {
            plan,
            fallback: Arc::new(fallback),
        }
    }

    /// Returns the selection for child `child_index` of scheme `parent`.
    pub(crate) fn descend(&self, parent: SchemeId, child_index: usize) -> Self {
        match self {
            Self::Follow {
                plan: Plan::Scheme { scheme, children },
                fallback,
            } if *scheme == parent => match children.get(child_index) {
                Some(child) => Self::Follow {
                    plan: child.clone(),
                    fallback: Arc::clone(fallback),
                },
                None => fallback.as_ref().clone(),
            },
            Self::Follow { fallback, .. } => fallback.as_ref().clone(),
            Self::Estimate | Self::Random(_) | Self::Exhaustive { .. } => self.clone(),
        }
    }
}

/// Collects the plans applied at the children of one compression site.
#[derive(Debug)]
pub(crate) struct PlanRecorder {
    /// The scheme whose children this recorder collects, or `None` for a top-level site.
    owner: Option<SchemeId>,
    /// `(child_index, plan)` pairs. A later record for the same child replaces the earlier one.
    slots: Mutex<Vec<(usize, Plan)>>,
}

impl PlanRecorder {
    /// Creates a recorder for the children of `owner`.
    pub(crate) fn new(owner: Option<SchemeId>) -> Self {
        Self {
            owner,
            slots: Mutex::new(Vec::new()),
        }
    }

    /// Records `plan` for child `child_index` of `parent`.
    ///
    /// Records from a different parent are ignored. They come from schemes that a scheme invokes
    /// directly rather than through the compressor.
    pub(crate) fn record(&self, parent: Option<SchemeId>, child_index: usize, plan: Plan) {
        if parent != self.owner {
            return;
        }
        let mut slots = self.slots.lock();
        match slots.iter_mut().find(|(index, _)| *index == child_index) {
            Some(slot) => slot.1 = plan,
            None => slots.push((child_index, plan)),
        }
    }

    /// Returns the plan recorded for `child_index`, if any.
    pub(crate) fn get(&self, child_index: usize) -> Option<Plan> {
        self.slots
            .lock()
            .iter()
            .find(|(index, _)| *index == child_index)
            .map(|(_, plan)| plan.clone())
    }

    /// Returns the recorded children as a dense list of at least `num_children` plans, with
    /// [`Plan::Adaptive`] for children that were never compressed.
    pub(crate) fn children(&self, num_children: usize) -> Arc<[Plan]> {
        let slots = self.slots.lock();
        let len = slots
            .iter()
            .map(|(index, _)| index + 1)
            .max()
            .unwrap_or(0)
            .max(num_children);
        let mut children = vec![Plan::Adaptive; len];
        for (index, plan) in slots.iter() {
            children[*index] = plan.clone();
        }
        children.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DICT: SchemeId = SchemeId { name: "test.dict" };
    const PACK: SchemeId = SchemeId { name: "test.pack" };

    fn dict_plan() -> Plan {
        Plan::scheme(
            DICT,
            vec![Plan::Canonical, Plan::scheme(PACK, Vec::<Plan>::new())],
        )
    }

    #[test]
    fn display() {
        assert_eq!(dict_plan().to_string(), "test.dict(canonical, test.pack)");
        assert_eq!(
            Plan::scheme(DICT, vec![Plan::Adaptive, Plan::Adaptive]).to_string(),
            "test.dict"
        );
        assert_eq!(
            Plan::scheme(DICT, vec![Plan::Adaptive, Plan::Constant]).to_string(),
            "test.dict(*, constant)"
        );
    }

    #[test]
    fn replace_and_enumerate_nodes() {
        let plan = dict_plan();
        let nodes = plan.decided_nodes();
        let paths: Vec<_> = nodes.iter().map(|node| node.path.clone()).collect();
        assert_eq!(paths, vec![vec![], vec![0], vec![1]]);
        assert_eq!(nodes[2].edge, Some((DICT, 1)));

        let replaced = plan.replace_at(&[1], Plan::Adaptive);
        assert_eq!(replaced.to_string(), "test.dict(canonical, *)");
        assert_eq!(replaced.decided_nodes().len(), 2);

        // Paths that leave the tree are ignored.
        assert_eq!(plan.replace_at(&[0, 3], Plan::Adaptive), plan);
        assert_eq!(plan.replace_at(&[], Plan::Canonical), Plan::Canonical);
    }

    #[test]
    fn follow_descends_into_matching_children() {
        let follow = Selection::follow(dict_plan(), Selection::Estimate);
        assert!(matches!(
            follow.descend(DICT, 0),
            Selection::Follow {
                plan: Plan::Canonical,
                ..
            }
        ));
        assert!(matches!(follow.descend(DICT, 5), Selection::Estimate));
        assert!(matches!(follow.descend(PACK, 0), Selection::Estimate));
    }

    #[test]
    fn recorder_ignores_other_parents() {
        let recorder = PlanRecorder::new(Some(DICT));
        recorder.record(Some(DICT), 1, Plan::Canonical);
        recorder.record(Some(PACK), 0, Plan::Constant);
        recorder.record(Some(DICT), 1, Plan::Constant);
        assert_eq!(
            recorder.children(2).as_ref(),
            &[Plan::Adaptive, Plan::Constant]
        );
        assert_eq!(recorder.get(0), None);
    }
}
