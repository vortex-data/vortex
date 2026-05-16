//! Spiral list-query benchmark — three-level parent/child/grandchild
//! `min(grandchild)` per parent, with a parent-side limit driving
//! back-propagation through both offset relations.
//!
//! Usage: `spiral [parents] [child_per_parent] [grandchild_per_child] [limit]`
//! Defaults: 100 / 100 / 100 / 1.
//!
//! Reports: wall-clock, grandchild rows decoded, child rows
//! decoded, and the back-prop reduction ratio
//! (baseline_grandchild_reads / limited_grandchild_reads).

use std::time::Instant;
use vortex_engine::examples;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parents: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let child_per_parent: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let grandchild_per_child: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let limit: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let total_grandchildren = parents * child_per_parent * grandchild_per_child;

    eprintln!(
        "spiral: parents={parents} child/parent={child_per_parent} grandchild/child={grandchild_per_child} limit={limit}"
    );
    eprintln!("  total grandchild rows: {total_grandchildren}");

    let t0 = Instant::now();
    let limited = examples::spiral_three_level(
        parents,
        child_per_parent,
        grandchild_per_child,
        limit,
    )
    .expect("limited spiral runs");
    let limited_ms = t0.elapsed();

    let t1 = Instant::now();
    let baseline = examples::spiral_three_level(
        parents,
        child_per_parent,
        grandchild_per_child,
        parents,
    )
    .expect("baseline spiral runs");
    let baseline_ms = t1.elapsed();

    let limited_g = limited.metrics.source_rows_read("grandchild_values");
    let baseline_g = baseline.metrics.source_rows_read("grandchild_values");
    let ratio = (baseline_g as f64) / (limited_g.max(1) as f64);

    eprintln!("  limited:  {limited_ms:?}  grandchild_reads={limited_g}");
    eprintln!("  baseline: {baseline_ms:?}  grandchild_reads={baseline_g}");
    eprintln!("  back-prop reduction: {ratio:.1}×");
    eprintln!("  output rows (limited): {:?}", limited.output_rows);
}
