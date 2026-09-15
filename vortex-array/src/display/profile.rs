// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measured decompression cost for every subtree of an array.
//!
//! [`DecompressionProfile::measure`] canonicalizes each node of an encoding tree in isolation and
//! records how long it took. The result is keyed by node identity so that
//! [`ThroughputExtractor`](crate::display::ThroughputExtractor) can annotate the same tree that
//! [`TreeDisplay`](crate::display::TreeDisplay) already renders.
//!
//! # Fusion
//!
//! A node's *subtree* time is the wall time to drive that node to canonical form on its own. Its
//! *self* time is the subtree time minus the subtree times of its children, i.e. the work the node
//! performs beyond decoding what it is built from.
//!
//! Encodings do not always pay for their children. The executor rewrites `(parent, child)` pairs
//! through `execute_parent` kernels and `reduce_parent` rules, so a parent may reach canonical form
//! without ever canonicalizing its child. Such a node costs *less* than the sum of its children,
//! which this module reports as a [`NodeTiming::fusion_saving`] rather than a negative self time.
//! Measuring fusion this way needs no executor instrumentation: it falls out of comparing a node's
//! own cost against its children's.

use std::time::Duration;
use std::time::Instant;

use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray as _;
use crate::VortexSessionExecute as _;

/// How many times to canonicalize each node when building a [`DecompressionProfile`].
#[derive(Debug, Clone, Copy)]
pub struct ProfileOptions {
    /// Untimed runs performed before measurement, to warm caches and lazily-computed stats.
    pub warmup: usize,
    /// Timed runs. The reported time is the median of them. Values below one are treated as one,
    /// since a node with no timed run has no time to report.
    pub reps: usize,
}

impl Default for ProfileOptions {
    fn default() -> Self {
        Self { warmup: 1, reps: 5 }
    }
}

/// The measured decompression cost of a single node.
#[derive(Debug, Clone, Copy)]
pub struct NodeTiming {
    /// Median time to canonicalize this node, including everything below it.
    pub subtree: Duration,
    /// Sum of the [`Self::subtree`] times of this node's direct children.
    pub children: Duration,
    /// Compressed size of this subtree, as reported by [`ArrayRef::nbytes`].
    pub input_nbytes: u64,
    /// Size of the canonical array this subtree decodes into.
    pub output_nbytes: u64,
    /// Logical length of this node.
    pub rows: u64,
}

impl NodeTiming {
    /// Time this node spends beyond decoding its children.
    ///
    /// Zero when the node costs less than its children, which is reported by
    /// [`Self::fusion_saving`] instead.
    pub fn self_time(&self) -> Duration {
        self.subtree.saturating_sub(self.children)
    }

    /// How much cheaper this node is than decoding its children separately.
    ///
    /// `Some` only when the node fuses its children's decompression into its own work.
    pub fn fusion_saving(&self) -> Option<Duration> {
        let saving = self.children.saturating_sub(self.subtree);
        (!saving.is_zero()).then_some(saving)
    }

    /// Compressed bytes consumed per second while canonicalizing this subtree.
    pub fn input_bytes_per_sec(&self) -> f64 {
        per_sec(self.input_nbytes, self.subtree)
    }

    /// Canonical bytes produced per second while canonicalizing this subtree.
    pub fn output_bytes_per_sec(&self) -> f64 {
        per_sec(self.output_nbytes, self.subtree)
    }

    /// Rows produced per second while canonicalizing this subtree.
    pub fn rows_per_sec(&self) -> f64 {
        per_sec(self.rows, self.subtree)
    }
}

fn per_sec(amount: u64, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        return f64::INFINITY;
    }
    amount as f64 / secs
}

/// Decompression timings for every node of an encoding tree.
///
/// Build one with [`Self::measure`], then render it with
/// [`ArrayRef::display_tree_throughput`].
#[derive(Debug, Clone)]
pub struct DecompressionProfile {
    root: Duration,
    nodes: HashMap<usize, NodeTiming>,
}

impl DecompressionProfile {
    /// Measure the decompression cost of `array` and of every node beneath it.
    ///
    /// Each node is canonicalized `warmup + reps` times, so this performs `O(nodes * reps)`
    /// decompressions. It is a profiling entry point, never something a `Display` implementation
    /// should reach for.
    ///
    /// Nodes are identified by the array they hold, so a subtree reachable by more than one path
    /// is measured once per occurrence but recorded once. Its cost still counts towards each
    /// parent that reaches it.
    pub fn measure(
        array: &ArrayRef,
        session: &VortexSession,
        options: ProfileOptions,
    ) -> VortexResult<Self> {
        let mut profile = Self {
            root: Duration::ZERO,
            nodes: HashMap::default(),
        };
        profile.root = profile.measure_node(array, session, options)?;
        Ok(profile)
    }

    /// The timing recorded for `array`, if it was part of the measured tree.
    pub fn get(&self, array: &ArrayRef) -> Option<&NodeTiming> {
        self.nodes.get(&array.addr())
    }

    /// The time taken to canonicalize the whole tree, used as the denominator for percentages.
    pub fn root_time(&self) -> Duration {
        self.root
    }

    /// The number of measured nodes. Repeated occurrences of one shared node count once.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether no node was measured.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Measure `array` and its children, returning `array`'s subtree time.
    fn measure_node(
        &mut self,
        array: &ArrayRef,
        session: &VortexSession,
        options: ProfileOptions,
    ) -> VortexResult<Duration> {
        let mut children = Duration::ZERO;
        for child in array.children() {
            children += self.measure_node(&child, session, options)?;
        }

        let (subtree, output_nbytes) = time_canonicalize(array, session, options)?;
        self.nodes.insert(
            array.addr(),
            NodeTiming {
                subtree,
                children,
                input_nbytes: array.nbytes(),
                output_nbytes,
                rows: array.len() as u64,
            },
        );
        Ok(subtree)
    }
}

/// Canonicalize `array` repeatedly, returning the median elapsed time and the canonical size.
fn time_canonicalize(
    array: &ArrayRef,
    session: &VortexSession,
    options: ProfileOptions,
) -> VortexResult<(Duration, u64)> {
    let mut output_nbytes = 0;
    for _ in 0..options.warmup {
        let mut ctx = session.create_execution_ctx();
        output_nbytes = array
            .clone()
            .execute::<Canonical>(&mut ctx)?
            .into_array()
            .nbytes();
    }

    let reps = options.reps.max(1);
    let mut elapsed = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut ctx = session.create_execution_ctx();
        let array = array.clone();
        let start = Instant::now();
        let canonical = array.execute::<Canonical>(&mut ctx)?;
        elapsed.push(start.elapsed());
        output_nbytes = canonical.into_array().nbytes();
    }

    elapsed.sort_unstable();
    Ok((elapsed[elapsed.len() / 2], output_nbytes))
}

#[cfg(test)]
mod tests {
    // The profile is session-agnostic; these tests do not need a configured session.
    #![allow(clippy::disallowed_methods)]

    use vortex_buffer::buffer;
    use vortex_error::VortexExpect as _;

    use super::*;
    use crate::arrays::DictArray;
    use crate::display::ThroughputExtractor;
    use crate::legacy_session;

    /// Options that keep the tests fast: the timings themselves are not asserted on.
    fn fast() -> ProfileOptions {
        ProfileOptions { warmup: 0, reps: 1 }
    }

    fn dict_array() -> VortexResult<ArrayRef> {
        Ok(DictArray::try_new(
            buffer![0u32, 1, 0, 1, 2].into_array(),
            buffer![10i32, 20, 30].into_array(),
        )?
        .into_array())
    }

    #[test]
    fn measures_every_node() -> VortexResult<()> {
        let array = dict_array()?;
        let profile = DecompressionProfile::measure(&array, legacy_session(), fast())?;

        assert_eq!(profile.len(), 3, "root plus codes plus values");
        for node in [array.clone()].into_iter().chain(array.children()) {
            assert!(profile.get(&node).is_some(), "missing timing for {node}");
        }
        Ok(())
    }

    #[test]
    fn child_timings_roll_up_into_the_parent() -> VortexResult<()> {
        let array = dict_array()?;
        let profile = DecompressionProfile::measure(&array, legacy_session(), fast())?;

        let root = profile.get(&array).vortex_expect("root is measured");
        let children: Duration = array
            .children()
            .iter()
            .map(|child| {
                profile
                    .get(child)
                    .vortex_expect("child is measured")
                    .subtree
            })
            .sum();
        assert_eq!(root.children, children);
        assert_eq!(root.rows, array.len() as u64);
        assert_eq!(root.input_nbytes, array.nbytes());
        assert_eq!(profile.root_time(), root.subtree);

        // Self time and fusion saving are two directions of the same comparison, never both.
        assert!(root.self_time().is_zero() || root.fusion_saving().is_none());
        Ok(())
    }

    #[test]
    fn zero_reps_still_reports_a_time() -> VortexResult<()> {
        let array = buffer![0i32, 1, 2].into_array();
        let options = ProfileOptions { warmup: 0, reps: 0 };
        let profile = DecompressionProfile::measure(&array, legacy_session(), options)?;

        assert!(profile.get(&array).is_some());
        Ok(())
    }

    #[test]
    fn leaf_has_no_children_and_no_fusion() -> VortexResult<()> {
        let array = buffer![0i32, 1, 2].into_array();
        let profile = DecompressionProfile::measure(&array, legacy_session(), fast())?;

        let leaf = profile.get(&array).vortex_expect("root is measured");
        assert_eq!(leaf.children, Duration::ZERO);
        assert_eq!(leaf.self_time(), leaf.subtree);
        assert_eq!(leaf.fusion_saving(), None);
        Ok(())
    }

    #[test]
    fn every_rendered_node_carries_a_throughput_line() -> VortexResult<()> {
        let array = dict_array()?;
        let rendered = array
            .tree_display_builder()
            .with(crate::display::EncodingSummaryExtractor)
            .with(ThroughputExtractor::measure(
                &array,
                legacy_session(),
                fast(),
            )?)
            .to_string();

        assert_eq!(
            rendered.matches("throughput: ").count(),
            3,
            "one line per node in:\n{rendered}"
        );
        assert!(rendered.contains(" | in "), "{rendered}");
        assert!(rendered.contains(" | out "), "{rendered}");
        Ok(())
    }

    #[test]
    fn unmeasured_nodes_are_skipped() -> VortexResult<()> {
        // A profile of one array must not annotate an unrelated tree.
        let measured = buffer![0i32, 1, 2].into_array();
        let other = buffer![9i64, 8].into_array();
        let profile = DecompressionProfile::measure(&measured, legacy_session(), fast())?;

        assert!(profile.get(&other).is_none());
        let rendered = other
            .tree_display_builder()
            .with(crate::display::EncodingSummaryExtractor)
            .with(ThroughputExtractor::new(profile))
            .to_string();
        assert!(!rendered.contains("throughput"), "{rendered}");
        Ok(())
    }
}
