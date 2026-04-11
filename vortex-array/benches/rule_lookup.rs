// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-child matching cost: current matches() scan vs HashMap lookup.
//!
//! Setup: N Primitive children, one Cast parent.
//!
//! reduce_parent (match only, nothing fires):
//!   Current per child: vtable dispatch → ArrayView → 3 × matches() → all miss
//!   Proposed per child: encoding_id() + HashMap.get()
//!
//! execute_parent (match + execute, CastExecuteAdaptor fires):
//!   Current per child: vtable dispatch → ArrayView → 4 × matches() → Cast hits → runs kernel
//!   Proposed per child: HashMap.get() → hit → call execute_parent

#![expect(clippy::unwrap_used)]

use std::collections::HashMap;
use std::hint::black_box;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;

fn main() {
    divan::main();
}

/// N Primitive children stored in a Vec (not inside a Chunked — avoids
/// confounding with ChunkedArray's own slots/offsets).
fn make_children(n: usize) -> Vec<ArrayRef> {
    (0..n)
        .map(|i| {
            PrimitiveArray::new(Buffer::from(vec![i as i32; 100]), Validity::NonNullable)
                .into_array()
        })
        .collect()
}

/// A Cast(Primitive) parent. The children above are what we iterate over.
fn make_cast_parent() -> ArrayRef {
    let a =
        PrimitiveArray::new(Buffer::from(vec![1i32; 100]), Validity::NonNullable).into_array();
    Cast.try_new_array(
        100,
        DType::Primitive(PType::I64, Nullability::NonNullable),
        [a],
    )
    .unwrap()
}

/// A Primitive "parent" (not a real expression parent — just to measure
/// the case where no child has any matching rules at all).
fn make_leaf_parent() -> ArrayRef {
    PrimitiveArray::new(Buffer::from(vec![1i32; 100]), Validity::NonNullable).into_array()
}

fn build_registry() -> HashMap<&'static str, HashMap<&'static str, bool>> {
    let mut map: HashMap<&'static str, HashMap<&'static str, bool>> = HashMap::new();
    for (p, c) in [
        ("vortex.cast", "vortex.primitive"),
        ("vortex.cast", "vortex.chunked"),
        ("vortex.between", "vortex.primitive"),
        ("vortex.binary", "vortex.primitive"),
        ("vortex.fill_null", "vortex.primitive"),
        ("vortex.take", "vortex.primitive"),
        ("vortex.masked", "vortex.primitive"),
        ("vortex.mask", "vortex.primitive"),
        ("vortex.slice", "vortex.primitive"),
        ("vortex.filter", "vortex.primitive"),
    ] {
        map.entry(p).or_default().insert(c, true);
    }
    map
}

const N: &[usize] = &[1, 10, 100];

// ============================================================================
// reduce_parent: MATCH ONLY (nothing fires)
//
// Primitive child, Cast parent.
// Primitive reduce rules: [Masked, Mask, Slice] — none match Cast.
// So this is purely wasted matching work.
// ============================================================================

/// Current: N × child.reduce_parent(parent, idx)
/// Each call does: vtable dispatch + ArrayView + 3 × matches() all miss.
#[divan::bench(args = N)]
fn reduce_match_only_current(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_cast_parent();
    bencher.bench(|| {
        for (i, child) in black_box(&children).iter().enumerate() {
            black_box(child.reduce_parent(black_box(&parent), i)).unwrap();
        }
    });
}

/// Proposed: N × (encoding_id + HashMap.get)
/// No vtable dispatch, no matches().
#[divan::bench(args = N)]
fn reduce_match_only_proposed(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_cast_parent();
    let registry = build_registry();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        let child_map = registry.get(black_box(pid.as_ref()));
        for child in black_box(&children).iter() {
            let cid = child.encoding_id();
            if let Some(cm) = child_map {
                black_box(cm.get(cid.as_ref()));
            }
        }
    });
}

/// Leaf parent: no rules for any child. Measures the overhead when the
/// parent isn't even an expression type.
#[divan::bench(args = N)]
fn reduce_match_only_current_leaf_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_leaf_parent();
    bencher.bench(|| {
        for (i, child) in black_box(&children).iter().enumerate() {
            black_box(child.reduce_parent(black_box(&parent), i)).unwrap();
        }
    });
}

/// Proposed with leaf parent: outer HashMap miss, skip all children.
#[divan::bench(args = N)]
fn reduce_match_only_proposed_leaf_parent(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_leaf_parent();
    let registry = build_registry();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        let child_map = registry.get(black_box(pid.as_ref()));
        if child_map.is_some() {
            for child in black_box(&children).iter() {
                let cid = child.encoding_id();
                black_box(child_map.unwrap().get(cid.as_ref()));
            }
        }
    });
}

// ============================================================================
// execute_parent: MATCH + EXECUTE (CastExecuteAdaptor fires)
//
// Primitive child, Cast parent.
// Primitive execute kernels: [Between, Cast, FillNull, Take].
// Between misses (1 matches()), Cast hits (1 matches() + runs kernel).
// So per child: 2 matches() calls + kernel execution.
//
// We iterate ALL N children (no early return) to measure total cost.
// ============================================================================

/// Current: N × child.execute_parent(parent, idx, ctx)
/// Each call: vtable dispatch + ArrayView + 2× matches() + kernel execution.
#[divan::bench(args = N)]
fn execute_match_and_run_current(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_cast_parent();
    bencher.bench(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        for (i, child) in black_box(&children).iter().enumerate() {
            black_box(child.execute_parent(black_box(&parent), i, &mut ctx)).unwrap();
        }
    });
}

/// Proposed: N × (HashMap.get → hit → call execute_parent)
/// The HashMap lookup replaces the matches() scan. We still call
/// execute_parent to measure the real kernel execution cost.
#[divan::bench(args = N)]
fn execute_match_and_run_proposed(bencher: divan::Bencher, n: usize) {
    let children = make_children(n);
    let parent = make_cast_parent();
    let registry = build_registry();
    let pid = parent.encoding_id();
    bencher.bench(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let child_map = registry.get(black_box(pid.as_ref()));
        for (i, child) in black_box(&children).iter().enumerate() {
            let cid = child.encoding_id();
            if let Some(cm) = child_map {
                if cm.get(cid.as_ref()).is_some() {
                    // Rule exists — call the real execute_parent
                    black_box(child.execute_parent(black_box(&parent), i, &mut ctx)).unwrap();
                }
            }
        }
    });
}
