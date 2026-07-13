// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Exclusion rules that keep incompatible schemes out of a cascade chain.

use crate::scheme::SchemeId;

/// Selects which children of a cascading scheme a rule applies to.
#[derive(Debug, Clone, Copy)]
pub enum ChildSelection {
    /// Rule applies to all children.
    All,
    /// Rule applies to a single child.
    One(usize),
    /// Rule applies to multiple specific children.
    Many(&'static [usize]),
}

impl ChildSelection {
    /// Returns `true` if this selection includes the given child index.
    pub fn contains(&self, child_index: usize) -> bool {
        match self {
            ChildSelection::All => true,
            ChildSelection::One(idx) => *idx == child_index,
            ChildSelection::Many(indices) => indices.contains(&child_index),
        }
    }
}

/// Push rule: declared by a cascading scheme to exclude another scheme from the subtree
/// rooted at the specified children.
///
/// Use this when the declaring scheme (the ancestor) knows about the excluded scheme. For example,
/// `ZigZag` excludes `Dict` from all its children.
#[derive(Debug, Clone, Copy)]
pub struct DescendantExclusion {
    /// The scheme to exclude from descendants.
    pub excluded: SchemeId,
    /// Which children of the declaring scheme this rule applies to.
    pub children: ChildSelection,
}

/// Pull rule: declared by a scheme to exclude itself when the specified ancestor is in the
/// cascade chain.
///
/// Use this when the excluded scheme (the descendant) knows about the ancestor. For example,
/// `Sequence` excludes itself when `IntDict` is an ancestor on its codes child.
#[derive(Debug, Clone, Copy)]
pub struct AncestorExclusion {
    /// The ancestor scheme that makes the declaring scheme ineligible.
    pub ancestor: SchemeId,
    /// Which children of the ancestor this rule applies to.
    pub children: ChildSelection,
}
