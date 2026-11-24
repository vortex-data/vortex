// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The compression policy that applies a [`TreeModel`].

use vortex_array::ExecutionCtx;

use super::features::site_int_features;
use super::tree::TreeModel;
use crate::scheme::CompressionPolicy;
use crate::scheme::SchemeId;
use crate::scheme::SchemeSelection;
use crate::scheme::SiteRequest;
use crate::stats::ArrayAndStats;

/// A policy that picks integer schemes with a learned tree and estimates everything else.
///
/// At a confident leaf whose features are in range, the tree's pick is applied without
/// estimating. At an unsure leaf the compressor estimates over the leaf's top `fallback_k`
/// schemes, and out of range it estimates over all of them.
#[derive(Debug)]
pub struct LearnedPolicy {
    /// The integer model.
    model: &'static TreeModel,
    /// The leaf confidence at or above which the pick is applied without estimating.
    min_confidence: f64,
    /// How many of the leaf's ranked schemes to estimate over when unsure.
    fallback_k: usize,
}

impl LearnedPolicy {
    /// A policy applying `model`, with the defaults the model was tuned for.
    pub fn new(model: &'static TreeModel) -> Self {
        Self {
            model,
            min_confidence: 0.9,
            fallback_k: 3,
        }
    }

    /// Sets the leaf confidence at or above which the pick is applied without estimating.
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Sets how many ranked schemes to estimate over when unsure; 0 estimates over all.
    pub fn with_fallback_k(mut self, fallback_k: usize) -> Self {
        self.fallback_k = fallback_k;
        self
    }

    /// The eligible schemes in the leaf's ranked order.
    fn ranked_eligible(&self, ranked: &[u8], eligible: &[SchemeId]) -> Vec<SchemeId> {
        ranked
            .iter()
            .filter_map(|&i| {
                let name = self.model.schemes[usize::from(i)];
                eligible.iter().copied().find(|id| id.name == name)
            })
            .collect()
    }
}

impl CompressionPolicy for LearnedPolicy {
    fn select_scheme_with_stats(
        &self,
        request: &SiteRequest<'_>,
        data: &ArrayAndStats,
        exec_ctx: &mut ExecutionCtx,
    ) -> SchemeSelection {
        let Some(features) = site_int_features(request, data, exec_ctx) else {
            return SchemeSelection::Estimate;
        };
        let leaf = self.model.predict(&features);
        let mut ranked = self.ranked_eligible(leaf.ranked, request.eligible);
        if ranked.is_empty() {
            return SchemeSelection::Estimate;
        }
        // Outside the training range the leaf's ranking means nothing, so estimate over all.
        if !self.model.in_range(&features) {
            return SchemeSelection::Estimate;
        }
        if leaf.confidence >= self.min_confidence {
            return SchemeSelection::Scheme(ranked[0]);
        }
        if self.fallback_k == 0 {
            return SchemeSelection::Estimate;
        }
        ranked.truncate(self.fallback_k);
        SchemeSelection::EstimateAmong(ranked)
    }
}
