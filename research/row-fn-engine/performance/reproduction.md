<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Reproduce the local experiment

Use a checkout at `96bd521eb0565555def2af7b8e97e96891728da6` and its pinned toolchain.
The manifest below used absolute paths into that checkout. Replace those paths with the local
checkout path before compiling. Run Cargo from the repository root to inherit its `.cargo` flags.

The experiment creates a separate package under `/tmp`. It does not add a workspace member or
change production source. The full source below is the source used for the reported final runs.

## Temporary manifest

Save this block as the temporary package's `Cargo.toml`.

```toml
[package]
name = "row-fn-overhead-probe"
version = "0.0.0"
edition = "2024"

[dependencies]
vortex-array = { path = "/Users/connor/spiral/vortex-data/vortex6/vortex-array", features = ["unstable_row_fns"] }
vortex-error = { path = "/Users/connor/spiral/vortex-data/vortex6/vortex-error" }
vortex-session = { path = "/Users/connor/spiral/vortex-data/vortex6/vortex-session" }
mimalloc = "=0.1.52"

[features]
count_allocations = []

[profile.release]
codegen-units = 1
lto = "off"
```

## Temporary source

Save this block as `src/main.rs` inside the temporary package.

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hint::black_box;
#[cfg(not(feature = "count_allocations"))]
use std::time::Instant;

#[cfg(not(feature = "count_allocations"))]
use mimalloc::MiMalloc;
use vortex_array::{ArrayRef, ExecutionCtx, IntoArray, VortexSessionExecute};
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::{EmptyOptions, ScalarFnId, VecExecutionArgs};
use vortex_array::scalar_fn::unstable::row::{OutputElement, RowFn, RowVisitor, execute_rows};
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

#[cfg(not(feature = "count_allocations"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(feature = "count_allocations")]
mod allocations {
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use mimalloc::MiMalloc;

    pub static ENABLED: AtomicBool = AtomicBool::new(false);
    pub static COUNT: AtomicUsize = AtomicUsize::new(0);
    pub static BYTES: AtomicUsize = AtomicUsize::new(0);
    pub struct CountingAllocator;

    // SAFETY: every allocation operation forwards the unchanged allocation contract to MiMalloc.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if ENABLED.load(Ordering::Relaxed) {
                COUNT.fetch_add(1, Ordering::Relaxed);
                BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            }
            // SAFETY: the allocator caller supplies the requirements for this operation.
            unsafe { MiMalloc.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            if ENABLED.load(Ordering::Relaxed) {
                COUNT.fetch_add(1, Ordering::Relaxed);
                BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            }
            // SAFETY: the allocator caller supplies the requirements for this operation.
            unsafe { MiMalloc.alloc_zeroed(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: the allocator caller supplies the requirements for this operation.
            unsafe { MiMalloc.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if ENABLED.load(Ordering::Relaxed) {
                COUNT.fetch_add(1, Ordering::Relaxed);
                BYTES.fetch_add(new_size, Ordering::Relaxed);
            }
            // SAFETY: the allocator caller supplies the requirements for this operation.
            unsafe { MiMalloc.realloc(ptr, layout, new_size) }
        }
    }
}

#[cfg(feature = "count_allocations")]
#[global_allocator]
static GLOBAL: allocations::CountingAllocator = allocations::CountingAllocator;

#[derive(Clone)]
struct AddOne;

impl RowFn for AddOne {
    type Options = EmptyOptions;
    const ARG_NAMES: &'static [&'static str] = &["value"];
    const INFALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("research.add_one");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(i64,), i64>(|(value,)| value.wrapping_add(1))
    }
}

#[inline(never)]
fn direct(input: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let input = input.clone().execute::<PrimitiveArray>(ctx)?.into_buffer::<i64>();
    let output: Vec<i64> = input.as_slice().iter().map(|value| value.wrapping_add(1)).collect();
    Ok(PrimitiveArray::new(output, Validity::NonNullable).into_array())
}

#[inline(never)]
fn direct_collector(input: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let input = input.clone().execute::<PrimitiveArray>(ctx)?.into_buffer::<i64>();
    Ok(i64::build_from(input.as_slice(), |value| value.wrapping_add(1)))
}

#[inline(never)]
fn framework(args: &VecExecutionArgs, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    execute_rows(&AddOne, &EmptyOptions, args, ctx)
}

#[inline(never)]
fn framework_fresh(input: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let args = VecExecutionArgs::new(vec![input.clone()], input.len());
    execute_rows(&AddOne, &EmptyOptions, &args, ctx)
}

fn call(mode: usize, input: &ArrayRef, args: &VecExecutionArgs, ctx: &mut ExecutionCtx) -> ArrayRef {
    match mode {
        0 => direct(black_box(input), black_box(ctx)).unwrap(),
        1 => framework(black_box(args), black_box(ctx)).unwrap(),
        2 => framework_fresh(black_box(input), black_box(ctx)).unwrap(),
        3 => direct(black_box(input), black_box(ctx)).unwrap(),
        4 => direct_collector(black_box(input), black_box(ctx)).unwrap(),
        _ => unreachable!(),
    }
}

#[cfg(not(feature = "count_allocations"))]
fn measure(mode: usize, repetitions: usize, input: &ArrayRef, args: &VecExecutionArgs, ctx: &mut ExecutionCtx) -> f64 {
    let start = Instant::now();
    for _ in 0..repetitions {
        drop(black_box(call(mode, input, args, ctx)));
    }
    start.elapsed().as_nanos() as f64 / repetitions as f64
}

fn main() {
    let session = vortex_array::array_session();
    let sizes = [0, 1, 8, 64, 1024, 16384, 262144];
    println!("kind,rows,sample,mode,repetitions,ns_per_call,allocations,requested_bytes");
    for rows in sizes {
        let input = PrimitiveArray::from_iter((0..rows).map(|index| index as i64)).into_array();
        let args = VecExecutionArgs::new(vec![input.clone()], rows);
        let mut ctx = session.create_execution_ctx();
        for mode in 0..5 {
            let result = call(mode, &input, &args, &mut ctx);
            assert_eq!(result.dtype(), input.dtype());
            let actual = result.execute::<PrimitiveArray>(&mut ctx).unwrap().into_buffer::<i64>();
            assert_eq!(actual.as_slice(), &(1..=rows as i64).collect::<Vec<_>>());
        }
        #[cfg(feature = "count_allocations")]
        {
            use std::sync::atomic::Ordering;
            for mode in [0, 1, 2, 4] {
                allocations::COUNT.store(0, Ordering::Relaxed);
                allocations::BYTES.store(0, Ordering::Relaxed);
                allocations::ENABLED.store(true, Ordering::Relaxed);
                drop(black_box(call(mode, &input, &args, &mut ctx)));
                allocations::ENABLED.store(false, Ordering::Relaxed);
                println!("allocation,{rows},0,{mode},1,0,{},{}", allocations::COUNT.load(Ordering::Relaxed), allocations::BYTES.load(Ordering::Relaxed));
            }
            continue;
        }
        #[cfg(not(feature = "count_allocations"))]
        {
            for mode in 0..5 {
                measure(mode, 1000, &input, &args, &mut ctx);
            }
            let mut repetitions = 1;
            while measure(0, repetitions, &input, &args, &mut ctx) * (repetitions as f64) < 2_000_000.0 {
                repetitions *= 2;
            }
            for sample in 0..32 {
                let order = if sample % 2 == 0 { [0, 4, 1, 2, 3] } else { [3, 2, 1, 4, 0] };
                for mode in order {
                    let time = measure(mode, repetitions, &input, &args, &mut ctx);
                    println!("timing,{rows},{sample},{mode},{repetitions},{time:.3},0,0");
                }
            }
        }
    }
}
```

## Commands

These are the commands used after writing the manifest and source. The first build permits Cargo
to prune the copied lockfile and add the temporary package. Every retained dependency must still
match the repository lockfile. Later builds use `--locked`.

```sh
cp Cargo.lock /tmp/row-fn-overhead-96bd521e/Cargo.lock
cargo build --release \
  --manifest-path /tmp/row-fn-overhead-96bd521e/Cargo.toml
cargo rustc --release --locked \
  --manifest-path /tmp/row-fn-overhead-96bd521e/Cargo.toml \
  -- --emit=llvm-ir,asm,link
cp /tmp/row-fn-overhead-96bd521e/target/release/row-fn-overhead-probe \
  /tmp/row-fn-overhead-96bd521e/timing-probe
```

Run the ordinary binary five times sequentially. Save each output as `run-0.csv` through `run-4.csv`.
The retained experiment used this runner:

```python
from pathlib import Path
import subprocess

root = Path("/tmp/row-fn-overhead-96bd521e")
for run in range(5):
    with (root / f"run-{run}.csv").open("w") as output:
        subprocess.run([str(root / "timing-probe")], stdout=output, check=True)
```

Build the allocation-counter variant only after timing completes. It writes allocation observations
and does not enter the timing branch.

```sh
cargo build --release --locked --features count_allocations \
  --manifest-path /tmp/row-fn-overhead-96bd521e/Cargo.toml
/tmp/row-fn-overhead-96bd521e/target/release/row-fn-overhead-probe \
  > /tmp/row-fn-overhead-96bd521e/allocations.csv
```

The allocation build replaces the ordinary binary at Cargo's output path. The `timing-probe` copy
preserves the exact ordinary binary used for timing.

## Aggregation

For each size, gather the 160 observations for one mode across the five runs. Report their median
as that mode's time. For a paired difference, subtract modes within the same run, size, and sample.
Take the median and quartiles of the 160 resulting differences. The report uses Python
`statistics.quantiles(values, n=4, method="exclusive")` for the interquartile range.

The per-process summaries apply the same operations to that process's 32 observations.
A paired ratio divides the two times inside each sample before aggregation. Do not divide displayed
medians and label that result a paired ratio.

## Retained local artifact identities

These artifacts are temporary, so the Markdown source, data, and compiler excerpts are the durable
record. Binary hashes identify this build but need not match a rebuild at a different absolute path.

| Artifact | SHA-256 |
| --- | --- |
| Measured `src/main.rs`. | `9c5fc99648a098b278cd073826f16beeb8d7e5bbd1aceaf0428bdb8f926f3ff0` |
| Pruned temporary `Cargo.lock`. | `4e7ea105fa1a5480e8698aa635ab540d7785e5f3a5fa08eb021cdeb23689b525` |
| Ordinary timing binary. | `1bb7b2a891f9b3b6905d974bc8b48bb9edacc33a67aa57b1825f2f5fccca4089` |
| Optimized LLVM IR. | `2fcb23e3e6b5457a1f7c77d5e26e1360cff12e17210d31149bedf2a2b4da9299` |
| Assembly. | `92618354e30366200991bf0e598b5fb75037fe0d7f73edc24e49396d5e6d4e0c` |

The final timing artifact stem is `row_fn_overhead_probe-5032bd499a2e3b16` under
`/tmp/row-fn-overhead-96bd521e/target/release/deps/`. An earlier pilot generated a different artifact
stem. The [compiler evidence](compiler-evidence.md) refers only to the final artifact.
