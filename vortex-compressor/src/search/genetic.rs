// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! NSGA-II search over compression plans.

use std::iter;
use std::sync::Arc;

use parking_lot::Mutex;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use super::Candidate;
use super::Measurement;
use super::Timings;
use super::dominates;
use super::search_input;
use crate::CascadingCompressor;
use crate::plan::Plan;
use crate::plan::Selection;
use crate::scheme::CompressorContext;

/// Evolves compression plans for one array with NSGA-II, minimizing compressed size,
/// compression time and decompression time together.
///
/// The result is the Pareto front: the plans that no other plan found beats on all three
/// objectives at once. Pick one with [`CostModel::cheapest`](super::CostModel::cheapest), or by
/// a constraint such as "the smallest plan that decompresses within a time budget".
///
/// # Genome and operators
///
/// Each individual is a [`Plan`] tree.
///
/// - Random plans grow by choosing uniformly among canonical and the schemes that can compress
///   each site's array, so every plan is valid for the training array.
/// - Mutation either regrows a random subtree or prunes it to canonical.
/// - Crossover swaps two subtrees that hang from the same parent scheme and child index, such as
///   the dictionary codes of two dictionary plans, so the swapped subtree compresses the same
///   kind of array.
/// - Each offspring is applied to the training array before it is measured. A planned scheme
///   that fails there is replaced by a random choice, and the individual keeps the plan that
///   was actually applied.
///
/// The initial population holds the plan the estimate-based compressor picks, the caller's
/// [`initial_plans`](Self::initial_plans) (for example from an
/// [`ExhaustiveSearch`](super::ExhaustiveSearch)), and random plans. Selection is elitist, so
/// the smallest plan found never leaves the population.
#[derive(Debug, Clone)]
pub struct GeneticSearch {
    /// Number of plans kept in each generation.
    pub population: usize,
    /// Number of generations to evolve.
    pub generations: usize,
    /// Probability that two parents swap subtrees.
    pub crossover_rate: f64,
    /// Probability that an offspring is mutated. Offspring identical to a parent are always
    /// mutated.
    pub mutation_rate: f64,
    /// How many times each timing is repeated. The fastest run is kept.
    pub iterations: usize,
    /// Seed for the random choices.
    pub seed: u64,
    /// Plans added to the initial population.
    pub initial_plans: Vec<Plan>,
}

impl Default for GeneticSearch {
    fn default() -> Self {
        Self {
            population: 24,
            generations: 16,
            crossover_rate: 0.9,
            mutation_rate: 0.4,
            iterations: 3,
            seed: 0,
            initial_plans: Vec::new(),
        }
    }
}

impl GeneticSearch {
    /// Adds `plans` to the initial population.
    pub fn with_initial_plans(mut self, plans: impl IntoIterator<Item = Plan>) -> Self {
        self.initial_plans.extend(plans);
        self
    }

    /// Returns the Pareto front of plans for `array`, ordered by compressed size.
    ///
    /// # Errors
    ///
    /// Returns an error if `array` is not a leaf array, or if compression or decompression
    /// fails.
    pub fn run(
        &self,
        compressor: &CascadingCompressor,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<Candidate>> {
        let array = search_input(array, exec_ctx)?;
        let size = self.population.max(2);
        let max_attempts = 4 * size;

        let mut evolution = Evolution {
            compressor,
            array: array.clone(),
            iterations: self.iterations,
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(self.seed))),
            measured: HashMap::default(),
        };

        let (_, estimated) = compressor.compress_recording_plan(&array, exec_ctx)?;
        let mut population = Vec::with_capacity(2 * size);
        for plan in iter::once(estimated).chain(self.initial_plans.iter().cloned()) {
            evolution.admit(&plan, &[], &mut population, exec_ctx)?;
        }
        let mut attempts = 0;
        while population.len() < size && attempts < max_attempts {
            attempts += 1;
            evolution.admit(&Plan::Adaptive, &[], &mut population, exec_ctx)?;
        }
        let mut population = select_survivors(population, size);

        for _ in 0..self.generations {
            let mut offspring = Vec::with_capacity(size);
            let mut attempts = 0;
            while offspring.len() < size && attempts < max_attempts {
                attempts += 1;
                let first = evolution.tournament(&population).plan.clone();
                let second = evolution.tournament(&population).plan.clone();
                let (left, right) = if evolution.chance(self.crossover_rate) {
                    evolution.crossover(&first, &second)
                } else {
                    (first.clone(), second.clone())
                };
                for child in [left, right] {
                    let child = if child == first
                        || child == second
                        || evolution.chance(self.mutation_rate)
                    {
                        evolution.mutate(&child)
                    } else {
                        child
                    };
                    evolution.admit(&child, &population, &mut offspring, exec_ctx)?;
                }
            }
            population.extend(offspring);
            population = select_survivors(population, size);
        }

        let mut front: Vec<Candidate> = population
            .into_iter()
            .filter(|individual| individual.rank == 0)
            .map(|individual| Candidate {
                plan: individual.plan,
                measurement: individual.measurement,
            })
            .collect();
        front.sort_by_key(|candidate| candidate.measurement.nbytes);
        Ok(front)
    }
}

/// A measured plan in the population.
struct Individual {
    /// The plan, as applied to the training array.
    plan: Plan,
    /// The plan's cost on the training array.
    measurement: Measurement,
    /// The index of the non-dominated front the plan belongs to, where 0 is the Pareto front.
    rank: usize,
    /// How isolated the plan is within its front. Larger values keep the front spread out.
    crowding: f64,
}

/// The state shared by one run of the search.
struct Evolution<'a> {
    /// The compressor whose schemes plans are built from.
    compressor: &'a CascadingCompressor,
    /// The canonical training array.
    array: ArrayRef,
    /// How many times each timing is repeated.
    iterations: usize,
    /// The random number generator, shared with random plan growth.
    rng: Arc<Mutex<StdRng>>,
    /// Measurements of the plans evaluated so far, so that each plan is timed once.
    measured: HashMap<Plan, Measurement>,
}

impl Evolution<'_> {
    /// Applies `plan` to the training array, growing its adaptive sites at random, measures the
    /// applied plan, and adds it to `admitted` unless `existing` or `admitted` already hold it.
    fn admit(
        &mut self,
        plan: &Plan,
        existing: &[Individual],
        admitted: &mut Vec<Individual>,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let site_ctx = CompressorContext::new();
        let (_, applied, _) = self.compressor.apply_plan(
            &self.array,
            plan,
            &site_ctx,
            Selection::Random(Arc::clone(&self.rng)),
            exec_ctx,
        )?;

        let is_known = |individual: &Individual| individual.plan == applied;
        if existing.iter().any(is_known) || admitted.iter().any(is_known) {
            return Ok(());
        }

        let measurement = match self.measured.get(&applied) {
            Some(measurement) => *measurement,
            None => {
                let evaluation = self.compressor.evaluate_plan(
                    &self.array,
                    &applied,
                    &site_ctx,
                    Timings::ALL,
                    self.iterations,
                    exec_ctx,
                )?;
                self.measured
                    .insert(applied.clone(), evaluation.measurement);
                evaluation.measurement
            }
        };

        admitted.push(Individual {
            plan: applied,
            measurement,
            rank: 0,
            crowding: 0.0,
        });
        Ok(())
    }

    /// Returns `true` with probability `p`.
    fn chance(&self, p: f64) -> bool {
        self.rng.lock().random_bool(p.clamp(0.0, 1.0))
    }

    /// Picks the better of two random individuals: lower rank first, then higher crowding.
    fn tournament<'p>(&self, population: &'p [Individual]) -> &'p Individual {
        let mut rng = self.rng.lock();
        let first = &population[rng.random_range(0..population.len())];
        let second = &population[rng.random_range(0..population.len())];
        if second.rank < first.rank
            || (second.rank == first.rank && second.crowding > first.crowding)
        {
            second
        } else {
            first
        }
    }

    /// Regrows a random subtree of `plan`, or prunes it to canonical.
    fn mutate(&self, plan: &Plan) -> Plan {
        let nodes = plan.decided_nodes();
        if nodes.is_empty() {
            return Plan::Adaptive;
        }
        let mut rng = self.rng.lock();
        let node = &nodes[rng.random_range(0..nodes.len())];
        let prune = !matches!(node.plan, Plan::Canonical) && rng.random_bool(0.25);
        let replacement = if prune {
            Plan::Canonical
        } else {
            Plan::Adaptive
        };
        plan.replace_at(&node.path, replacement)
    }

    /// Swaps two differing subtrees of `first` and `second` that hang from the same parent
    /// scheme and child index. Returns the parents unchanged if they have no such pair.
    fn crossover(&self, first: &Plan, second: &Plan) -> (Plan, Plan) {
        let first_nodes = first.decided_nodes();
        let second_nodes = second.decided_nodes();
        let pairs: Vec<_> = first_nodes
            .iter()
            .filter(|node| node.edge.is_some())
            .flat_map(|ours| {
                second_nodes
                    .iter()
                    .filter(move |theirs| theirs.edge == ours.edge && theirs.plan != ours.plan)
                    .map(move |theirs| (ours, theirs))
            })
            .collect();
        if pairs.is_empty() {
            return (first.clone(), second.clone());
        }

        let (ours, theirs) = pairs[self.rng.lock().random_range(0..pairs.len())];
        (
            first.replace_at(&ours.path, theirs.plan.clone()),
            second.replace_at(&theirs.path, ours.plan.clone()),
        )
    }
}

/// Keeps the best `size` individuals of `pool` by NSGA-II order, and sets their rank and
/// crowding distance.
///
/// Whole fronts are kept in order of rank. The first front that does not fit is truncated to
/// its most isolated members, so the kept front stays spread out.
fn select_survivors(pool: Vec<Individual>, size: usize) -> Vec<Individual> {
    let objectives: Vec<[f64; 3]> = pool
        .iter()
        .map(|individual| individual.measurement.objectives())
        .collect();
    let mut pool: Vec<Option<Individual>> = pool.into_iter().map(Some).collect();

    let mut survivors = Vec::with_capacity(size);
    for (rank, front) in non_dominated_fronts(&objectives).into_iter().enumerate() {
        if survivors.len() >= size {
            break;
        }
        let distances = crowding_distances(&objectives, &front);
        let mut members: Vec<(usize, f64)> = front.into_iter().zip(distances).collect();
        if survivors.len() + members.len() > size {
            members.sort_by(|a, b| b.1.total_cmp(&a.1));
            members.truncate(size - survivors.len());
        }
        for (index, crowding) in members {
            let mut individual = pool[index]
                .take()
                .vortex_expect("each individual belongs to exactly one front");
            individual.rank = rank;
            individual.crowding = crowding;
            survivors.push(individual);
        }
    }
    survivors
}

/// Sorts points into non-dominated fronts: front 0 holds the points no other point dominates,
/// front 1 the points only front 0 dominates, and so on.
fn non_dominated_fronts(objectives: &[[f64; 3]]) -> Vec<Vec<usize>> {
    let n = objectives.len();
    let mut dominated_count = vec![0usize; n];
    let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i != j && dominates(&objectives[i], &objectives[j]) {
                dominated[i].push(j);
                dominated_count[j] += 1;
            }
        }
    }

    let mut fronts = Vec::new();
    let mut current: Vec<usize> = (0..n).filter(|&i| dominated_count[i] == 0).collect();
    while !current.is_empty() {
        let mut next = Vec::new();
        for &i in &current {
            for &j in &dominated[i] {
                dominated_count[j] -= 1;
                if dominated_count[j] == 0 {
                    next.push(j);
                }
            }
        }
        fronts.push(current);
        current = next;
    }
    fronts
}

/// Returns the crowding distance of each member of `front`: the sum over objectives of the gap
/// between its neighbors, normalized by the objective's range. The extremes of each objective
/// get an infinite distance so they are always kept.
fn crowding_distances(objectives: &[[f64; 3]], front: &[usize]) -> Vec<f64> {
    let mut distances = vec![0.0; front.len()];
    if front.len() <= 2 {
        distances.fill(f64::INFINITY);
        return distances;
    }

    let last = front.len() - 1;
    for objective in 0..3 {
        let value = |position: usize| objectives[front[position]][objective];
        let mut order: Vec<usize> = (0..front.len()).collect();
        order.sort_by(|&a, &b| value(a).total_cmp(&value(b)));

        distances[order[0]] = f64::INFINITY;
        distances[order[last]] = f64::INFINITY;
        let range = value(order[last]) - value(order[0]);
        if range > 0.0 {
            for k in 1..last {
                distances[order[k]] += (value(order[k + 1]) - value(order[k - 1])) / range;
            }
        }
    }
    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fronts_are_ranked_by_dominance() {
        let objectives = [
            [1.0, 1.0, 1.0], // 0: dominates 2 and 3
            [0.5, 2.0, 1.0], // 1: trades size for speed with 0
            [2.0, 2.0, 2.0], // 2: dominated by 0
            [3.0, 3.0, 3.0], // 3: dominated by 0, 1 and 2
        ];
        assert_eq!(
            non_dominated_fronts(&objectives),
            vec![vec![0, 1], vec![2], vec![3]]
        );
    }

    #[test]
    fn crowding_keeps_extremes() {
        let objectives = [
            [0.0, 3.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        let distances = crowding_distances(&objectives, &[0, 1, 2, 3]);
        assert!(distances[0].is_infinite() && distances[3].is_infinite());
        assert!(distances[1].is_finite() && distances[1] > 0.0);
    }
}
