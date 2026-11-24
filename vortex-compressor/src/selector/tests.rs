// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::INT_FEATURE_NAMES;
use super::INT_V1;
use super::Leaf;
use super::N_INT_FEATURES;
use super::Node;
use super::TreeModel;

static TREE: TreeModel = TreeModel {
    name: "test",
    training_hash: "",
    schemes: &["a", "b"],
    feature_names: &INT_FEATURE_NAMES,
    feature_min: &[0.0; N_INT_FEATURES],
    feature_max: &[10.0; N_INT_FEATURES],
    nodes: &[
        Node::Split {
            feature: 4,
            threshold: 5.0,
            left: 1,
            right: 2,
        },
        Node::Leaf(0),
        Node::Leaf(1),
    ],
    leaves: &[
        Leaf {
            ranked: &[0, 1],
            confidence: 1.0,
        },
        Leaf {
            ranked: &[1],
            confidence: 0.5,
        },
    ],
};

#[test]
fn test_predict_follows_splits() {
    let mut features = [1.0; N_INT_FEATURES];
    features[4] = 5.0;
    assert_eq!(TREE.predict(&features).ranked, &[0, 1]);
    features[4] = 5.5;
    assert_eq!(TREE.predict(&features).ranked, &[1]);
}

#[test]
fn test_in_range() {
    let mut features = [1.0; N_INT_FEATURES];
    assert!(TREE.in_range(&features));
    features[0] = 11.0;
    assert!(!TREE.in_range(&features));
}

#[test]
fn test_generated_model_matches_the_features() {
    assert_eq!(INT_V1.feature_names, INT_FEATURE_NAMES);
    assert_eq!(INT_V1.feature_min.len(), N_INT_FEATURES);
    assert_eq!(INT_V1.feature_max.len(), N_INT_FEATURES);
}
