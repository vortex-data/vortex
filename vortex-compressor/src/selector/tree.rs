// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A decision tree stored as a flat table of nodes.

/// One node of a [`TreeModel`].
#[derive(Debug, Clone, Copy)]
pub enum Node {
    /// Go to `left` when `features[feature]`, rounded to `f32`, is at most `threshold`, else to
    /// `right`.
    Split {
        /// The feature index.
        feature: u8,
        /// The split threshold.
        threshold: f64,
        /// The node index for values at or below the threshold.
        left: u16,
        /// The node index for values above the threshold.
        right: u16,
    },
    /// A leaf, by index into [`TreeModel::leaves`].
    Leaf(u16),
}

/// What a leaf of a [`TreeModel`] decides.
#[derive(Debug, Clone, Copy)]
pub struct Leaf {
    /// Scheme indices into [`TreeModel::schemes`], best first. The first is the pick.
    pub ranked: &'static [u8],
    /// Share of the leaf's training regret weight on rows where the pick was within the
    /// tolerance of the best scheme, from 0 to 1.
    pub confidence: f64,
}

/// A trained selector for one dtype family.
#[derive(Debug)]
pub struct TreeModel {
    /// The model's name and version.
    pub name: &'static str,
    /// A hash of the training table, for provenance.
    pub training_hash: &'static str,
    /// The schemes the model chooses between, by name.
    pub schemes: &'static [&'static str],
    /// The feature names the model was trained on, which must match the selector's.
    pub feature_names: &'static [&'static str],
    /// Each feature's minimum over the training rows.
    pub feature_min: &'static [f64],
    /// Each feature's maximum over the training rows.
    pub feature_max: &'static [f64],
    /// The nodes; the root is the first.
    pub nodes: &'static [Node],
    /// The leaves the nodes refer to.
    pub leaves: &'static [Leaf],
}

impl TreeModel {
    /// The leaf `features` fall in.
    pub fn predict(&self, features: &[f64]) -> &Leaf {
        let mut index = 0usize;
        loop {
            match self.nodes[index] {
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    // The trainer compares features as `f32`, so round the same way for
                    // identical splits.
                    #[expect(clippy::cast_possible_truncation, reason = "matches the trainer")]
                    let value = f64::from(features[usize::from(feature)] as f32);
                    index = if value <= threshold {
                        usize::from(left)
                    } else {
                        usize::from(right)
                    };
                }
                Node::Leaf(leaf) => return &self.leaves[usize::from(leaf)],
            }
        }
    }

    /// Whether every feature lies within the range seen in training.
    pub fn in_range(&self, features: &[f64]) -> bool {
        features
            .iter()
            .zip(self.feature_min.iter().zip(self.feature_max))
            .all(|(value, (min, max))| value >= min && value <= max)
    }
}
