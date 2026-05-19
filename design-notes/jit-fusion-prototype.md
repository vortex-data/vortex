# JIT Compression-Kernel Fusion — Design Sketch

Status: **sketch / RFC**. Not implemented. Lives on the
`claude/jit-compression-kernel-design-g6wGm` branch as a thinking aid.

The goal is to evaluate using Cranelift as a runtime code generator that
specializes a *decompression chain* (ALP over FoR over Delta over BitPacked) into
a single fused kernel per chain fingerprint, then to think about what we would
have to build to turn that one prototype into a framework. The first half is the
sketch; the second half is the framework discussion; the appendix is the prior
art we have inside the repo today.

---

## 1. Why JIT, and why now

Compressed columns are usually decoded one encoding at a time. Each layer
allocates an intermediate buffer the size of the canonical output, runs a tight
SIMD loop, then hands the buffer to the next layer. For a chain of
`ALP(FoR(Delta(BitPacked)))` over a 1M-element column of `f64`, that's three
8 MiB intermediates and three trips through memory.

We already do one hand-rolled fusion in
`encodings/fastlanes/src/for/array/for_decompress.rs:51-78`:

```rust
if array.reference_scalar().dtype().is_unsigned_int()
    && let Some(bp) = array.encoded().as_opt::<BitPacked>()
{
    return fused_decompress::<T>(array, bp, ctx);
}
```

That path wins because it pulls the FoR `wrapping_add(reference)` into the same
1024-element block loop that fastlanes' bit-unpack uses, via a pluggable
`UnpackStrategy` trait. So the question "is fusion worth doing" is already
answered for the FoR/BitPacked pair. The next question — "can we generate the
analogous fusion for *any* chain a compressor produces, without writing a
quadratic number of `FooStrategy` types" — is what motivates a JIT.

A JIT also lets us bake encoding *parameters* as IR constants:

- BitPacked `bit_width` → unrolled shift/mask sequence
- FoR `reference` → `iadd_imm`
- ALP `(e, f)` → a single `f64` constant multiplier
- Delta `LANES` (16 for u64, 32 for u32) → unrolled lane reduction

Specialization on parameters is something monomorphization-by-generics can't do
when the parameter is only known at runtime (which is the common case — bit
widths and reference values come from the data).

---

## 2. The chain we are sketching

Concrete target: a `f64` column compressed as
`ALP(exponents) → FoR(reference) → Delta → BitPacked(width)`.

> Note on shape — the survey (`design-notes` appendix) found that Vortex does
> *not* currently produce this exact chain end-to-end. ALP's `encoded` child is
> a primitive `i32`/`i64`, not further compressed (`encodings/alp/src/alp/array.rs:202-218`).
> Delta's bases are a separate primitive child, not further compressed in the
> array structure itself. The chain we're sketching is what a future compressor
> *would* produce if we taught it to recurse. The JIT design has to handle that
> as a separate concern from the per-encoding codegen; see §6.

The decompression we want to emit, in pseudo-Rust:

```rust
for block in chunks_of(packed, 1024) {                       // BitPacked
    let mut lanes = unpack_block(bit_width, block);          //   const-folded shifts
    for lane in 0..16 {                                      // FoR
        for v in &mut lanes[lane] { *v += reference_i64; }
    }
    for lane in 0..16 {                                      // Delta
        let base = bases[block_idx * 16 + lane];
        for v in &mut lanes[lane] { *v = base.wrapping_add(*v); base = *v; }
    }
    untranspose(lanes, &mut out_i64_block);                  // FastLanes layout
    for v in &mut out_i64_block {                            // ALP decode
        *v_f64 = (*v_i64 as f64) * F10[f] * IF10[e];
    }
    out_f64.extend(out_i64_block);
}
apply_patches(&mut out_f64, exception_idx, exception_vals);  // ALP exceptions
apply_patches(&mut out_f64, bp_patch_idx, bp_patch_vals);    // BitPacked patches
```

Every line of that is a candidate for IR generation. The patches are explicitly
*not* inside the block loop — see §5 on patch ABI.

---

## 3. Cranelift IR sketch

The function we emit has a flat, C-style signature so the calling-side wrapper
can stay in safe Rust:

```rust
type FusedDecode = unsafe extern "C" fn(
    // BitPacked
    packed: *const u8,
    n_blocks: u32,
    // Delta
    bases: *const i64,
    // Output
    out: *mut f64,
    // Patches (applied after the loop, but pointers passed in)
    alp_idx: *const u64, alp_val: *const f64, n_alp: u32,
    bp_idx:  *const u64, bp_val:  *const i64, n_bp:  u32,
);
```

Encoding *constants* (`bit_width`, `reference`, `(e, f)`) are baked into the IR
at compile time and do not appear in the signature. That is the JIT win.

A minimal Cranelift builder for one block looks roughly like this. I am sketching
the structure rather than a working implementation — the IR opcodes are real
Cranelift IR, the surrounding Rust uses `cranelift-codegen` and
`cranelift-jit` conventions.

```rust
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Module, Linkage};

pub struct ChainCodegen {
    bit_width: u8,
    reference: i64,
    alp_e: u8,
    alp_f: u8,
}

impl ChainCodegen {
    fn emit_function(&self, m: &mut JITModule) -> cranelift_module::FuncId {
        let mut ctx = m.make_context();
        let mut sig = m.make_signature();

        // (packed, n_blocks, bases, out, alp_idx, alp_val, n_alp, bp_idx, bp_val, n_bp)
        for _ in 0..10 { sig.params.push(AbiParam::new(m.target_config().pointer_type())); }
        ctx.func.signature = sig;

        let id = m.declare_function("fused_decode", Linkage::Local, &ctx.func.signature).unwrap();
        let mut fbc = FunctionBuilderContext::new();
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fbc);

        let entry = fb.create_block();
        fb.append_block_params_for_function_params(entry);
        fb.switch_to_block(entry);
        fb.seal_block(entry);

        let packed   = fb.block_params(entry)[0];
        let n_blocks = fb.block_params(entry)[1];
        let bases    = fb.block_params(entry)[2];
        let out      = fb.block_params(entry)[3];

        // Block loop: for i in 0..n_blocks { emit_block(i, ...) }
        let loop_header = fb.create_block();
        let loop_body   = fb.create_block();
        let loop_exit   = fb.create_block();
        fb.append_block_param(loop_header, types::I32);

        let zero = fb.ins().iconst(types::I32, 0);
        fb.ins().jump(loop_header, &[zero]);

        fb.switch_to_block(loop_header);
        let i = fb.block_params(loop_header)[0];
        let cond = fb.ins().icmp(IntCC::UnsignedLessThan, i, n_blocks);
        fb.ins().brif(cond, loop_body, &[], loop_exit, &[]);

        fb.switch_to_block(loop_body);
        self.emit_one_block(&mut fb, packed, bases, out, i);
        let one = fb.ins().iconst(types::I32, 1);
        let next = fb.ins().iadd(i, one);
        fb.ins().jump(loop_header, &[next]);
        fb.seal_block(loop_body);
        fb.seal_block(loop_header);

        fb.switch_to_block(loop_exit);
        self.emit_patch_calls(&mut fb /*, args from entry */);
        fb.ins().return_(&[]);
        fb.seal_block(loop_exit);

        fb.finalize();
        m.define_function(id, &mut ctx).unwrap();
        m.clear_context(&mut ctx);
        id
    }

    fn emit_one_block(
        &self,
        fb: &mut FunctionBuilder,
        packed: Value, bases: Value, out: Value, block_idx: Value,
    ) {
        // -- BitPacked unpack: bit_width is a *constant* here, so we unroll.
        //    For each of the 1024 lanes, emit `band(ushr(word, lane_shift), mask)`.
        //    For width 11 in u64, 1024 lanes spans 1024*11/64 = 176 u64 words.
        //    A full unroll is ~1024 IR insts per block — fine.
        let lanes = self.emit_bitunpack(fb, packed, block_idx); // Vec<Value> length 1024

        // -- FoR: iadd_imm with the const reference. Cranelift constant-folds.
        let lanes = lanes.into_iter()
            .map(|v| fb.ins().iadd_imm(v, self.reference))
            .collect::<Vec<_>>();

        // -- Delta: per-lane prefix sum starting from bases[block_idx * 16 + lane].
        let lanes = self.emit_undelta(fb, lanes, bases, block_idx);

        // -- Untranspose: this is just an index permutation when we store.
        //    Emit 1024 `store` ops at the permuted offsets in `out`.
        //    (For non-unrolled, replace with an inner loop over a static permutation table.)
        // -- ALP: int -> f64 -> multiply by F10[f] * IF10[e] (both const-folded into one f64).
        let alp_scale = fb.ins().f64const(
            crate::alp::F10[self.alp_f as usize] * crate::alp::IF10[self.alp_e as usize],
        );
        let block_offset_bytes = fb.ins().imul_imm(block_idx, 1024 * 8);
        let block_out = fb.ins().iadd(out, block_offset_bytes);
        for (i, v_i64) in lanes.into_iter().enumerate() {
            let v_f = fb.ins().fcvt_from_sint(types::F64, v_i64);
            let v_f = fb.ins().fmul(v_f, alp_scale);
            // permuted offset = untranspose_index(i) * 8
            let off = untranspose_index(i) * 8;
            fb.ins().store(MemFlags::trusted(), v_f, block_out, off as i32);
        }
    }

    fn emit_patch_calls(&self, fb: &mut FunctionBuilder /*, args */) {
        // Call back into Rust `extern "C"` helpers.
        //   apply_alp_patches(out, alp_idx, alp_val, n_alp)
        //   apply_bp_patches(<intermediate buffer>, bp_idx, bp_val, n_bp)
        // We register these as JIT module symbols via `m.declare_function` +
        // `JITBuilder::symbol("apply_alp_patches", ptr_to_fn)`.
        // ...
    }
}
```

Things to flag about this sketch:

- **1024 IR instructions per block is fine for Cranelift's frontend, but the
  better path is to emit a small inner loop and let the optimizer hoist the
  bit-width constant.** Whether full unroll or inner-loop is faster is an
  empirical question we settle with the bench in §7.
- **Untranspose lives entirely in the address arithmetic.** This is the kind of
  thing that hand-written SIMD code has to do explicitly with `vshuf` ops;
  Cranelift can express it as `store` offsets and let the register allocator
  worry about it.
- **The patches loop is *not* JITted.** We call a Rust helper. See §5.
- **`MemFlags::trusted()` skips bounds checks.** The wrapper (§4) is responsible
  for guaranteeing `out` is large enough; that's the unsafe boundary.

---

## 4. Where the JIT plugs into Vortex

The survey of `vortex-array` found a kernel registry that is *exactly* the seam
we want:

- `ArrayKernels` is a session-scoped `SessionVar`
  (`vortex-array/src/optimizer/kernels.rs:117-120`).
- It maps `(parent_encoding_id, child_encoding_id)` to one or more
  `ExecuteParentFn` function pointers
  (`vortex-array/src/optimizer/kernels.rs`).
- The scheduler consults it before falling back to an encoding's static parent
  rules (`vortex-array/src/executor.rs:428-446`):

```rust
if let Some(kernels) = tmp_session.get_opt::<ArrayKernels>()
    && let Some(plugins) = kernels.find_execute_parent(parent.encoding_id(), child.encoding_id())
{
    for plugin in plugins.as_ref() {
        if let Some(result) = plugin(child, parent, slot_idx, ctx)? {
            return Ok(Some(result));
        }
    }
}
```

So the integration is: register a `JitKernels` session var of our own that
registers as many entries in `ArrayKernels` as we want to intercept. On a call,
we:

1. Walk the parent/child pair upward to recover the full visible chain (as deep
   as we want to fuse; the scheduler hands us one parent/child pair at a time,
   but the parent is still an `ArrayRef` and we can inspect its slots).
2. Compute a `ChainFingerprint` (see §6).
3. Look up the fingerprint in the compiled-kernel cache.
4. On miss, generate IR via §3, compile, install. On hit, call the cached
   function pointer.
5. Return `Some(canonical_result)` so the scheduler stops descending.

Notice the registry was *already* designed to support multiple plugins per
encoding pair and to let any plugin opt out by returning `None`. That fits the
JIT story perfectly: the JIT plugin can refuse to handle a chain it doesn't
have codegen for, and the static fallback takes over.

---

## 5. The hard bits — patches, nulls, and lane shapes

Three places where the IR sketch glosses over real complexity:

**Patches (ALP exceptions, BitPacked patch_values).** Patches are sparse
overrides applied after the dense decode. They have unpredictable indices and
can't usefully be inlined into the block loop. The current hand-fused path
already applies them as a separate scatter
(`encodings/fastlanes/src/for/array/for_decompress.rs:122-127`). The JIT should
do the same — emit a `call` to a Rust `extern "C"` helper after the loop. We
register the helper's address in the `JITBuilder` so Cranelift can resolve it.

There is a JIT-able variant: emit a sorted-merge of patch indices with the
block iteration, applying patches inline. That only wins above some patch
density (probably >5%). The cost-model decision is the same one
`bitpacking/compute/take.rs:43-45` already makes for unpack-then-take vs
selective-unpack. We can reuse that threshold.

**Null masks.** Today's decoders treat validity as a side channel. The JIT
should too: take a `*const u8` validity buffer (or `None`) and propagate it
verbatim. Validity manipulation is a different kernel; don't try to fuse it
into decompression.

**Lane shape.** FastLanes uses a 1024-element transposed layout where "lane"
means a logical SIMD slot, 16 for u64 / 32 for u32 / 64 for u16. The Delta and
BitPacked unpack already operate in this layout. ALP does not — it's a plain
linear array. So the untranspose has to happen between Delta and ALP, not after
ALP. Our IR sketch in §3 already does that, but it's worth flagging: the
*order* of encodings in a chain determines where the untranspose goes, and
that's an invariant the codegen has to enforce per-chain rather than
per-encoding.

---

## 6. From sketch to framework — open problems

If we were to commit to this beyond a prototype, here is the punch list. Each
of these is a discrete piece of work that the sketch does not solve.

1. **Per-encoding `CodegenDecompress` trait.** The framework needs a uniform
   way for an encoding to contribute a block of IR. Something like:

   ```rust
   trait CodegenDecompress {
       /// Constants this encoding contributes (bit_width, reference, ...).
       /// These get baked into the IR as constants and into the fingerprint.
       fn const_params(&self) -> ConstParams;

       /// Pointer/length args this encoding contributes to the function
       /// signature (packed buffer, base values, patches, ...).
       fn runtime_inputs(&self) -> Vec<RuntimeInput>;

       /// Given a builder positioned inside the block loop, with `child_lanes`
       /// holding the decoded output of the child encoding, emit IR that
       /// produces *this* encoding's output lanes.
       fn emit_block(
           &self,
           fb: &mut FunctionBuilder,
           ctx: &CodegenCtx,
           child_lanes: BlockLanes,
       ) -> BlockLanes;
   }
   ```

   The `UnpackStrategy` trait in fastlanes is the runtime analog of this. We're
   lifting it to the IR level.

2. **Chain fingerprint.** Identity of a kernel is `(chain of encoding ids,
   chain of const params, output ptype, op)`. Hash of that → cache key. Const
   *values* must be in the key (different bit widths produce different
   kernels). Pointer values and lengths must not be. Getting this boundary
   right is the single most important framework decision.

3. **Compile budget and cache eviction.** Cranelift compilation of one of these
   functions is ~ms-scale (we should measure). For one-shot queries that's a
   loss. Decisions:
   - LRU cache across `VortexSession`s? Or per-session only?
   - Persistent on-disk cache keyed by `(fingerprint, cranelift_version,
     target_triple)`? Cranelift can serialize compiled `cranelift-object`
     output, but the JIT path doesn't expose that — we'd need to use
     `cranelift-object` for AOT-like caching.
   - Compile threshold (only JIT chains with >N rows or seen >M times)?

4. **Cost model.** Some chains do not benefit from fusion: a very short column
   pays compile cost without recouping it; a chain dominated by one
   already-SIMD encoding (BitPacked alone) won't gain from JIT. We need a
   simple heuristic: at minimum `chain_length >= 2 && n_blocks > T`, where T
   covers compile amortization.

5. **Patch ABI.** Decide whether patches are post-loop scatter (default) or
   in-loop sorted-merge (high-density override). The decision must be part of
   the fingerprint so the cache returns the right kernel.

6. **Safety boundary.** The emitted function is `unsafe extern "C"`. The
   per-chain wrapper that we register in `ArrayKernels` does the
   alignment/length checks once, then calls the JITted function. Wrapper
   responsibilities:
   - Validate every pointer comes from a `Buffer<T>` with correct alignment
     (`vortex-buffer` already guarantees this).
   - Validate length is exactly `n_blocks * 1024 + tail` and refuse otherwise
     — or generate a separate tail kernel.
   - Validate dtype matches the fingerprint.
   - On any mismatch return `None` and let the static path take over.

7. **Test harness.** Each JIT kernel must be diffed against the
   canonical-path output on the same input. Proptest cases keyed by the same
   set of `(bit_width, reference, exponents, patch_density)` parameters. This
   is non-negotiable; an incorrect JIT silently corrupts data.

8. **perf integration.** Cranelift can emit `perf-jitdump` so `perf record`
   sees our kernels by name. Set this up early — once kernels start fusing,
   line-level profile diffs are how we'll understand wins and regressions.

9. **DType matrix.** The cross product is `PType × encoding_chain × op`. We
   need codegen for u8/u16/u32/u64/i8/.../f32/f64, plus likely
   nullable-vs-non-nullable, times each chain shape. Mitigation: most
   encodings have only 2-3 type families they support (BitPacked is integers
   only; ALP is f32/f64 only). The framework should plumb the type through
   `CodegenDecompress::emit_block` so encodings can refuse types they don't
   support and fall back.

10. **Target compatibility.** Cranelift supports x86_64 and aarch64; that
    covers our CI. WASM build of Vortex is a separate question — Cranelift can
    *target* wasm but we probably don't ship a JIT in the browser. The framework
    should be feature-gated (`features = ["jit"]`) and a no-op when off.

11. **Operations beyond decompress.** The same framework should generate
    `filter(mask) → canonical` and `take(indices) → canonical`. These have
    different output sizes and different optimal block strategies (selective
    decode under low selectivity). The `op` is part of the fingerprint; the
    `CodegenDecompress` trait probably becomes `CodegenKernel` parameterized
    by `KernelOp`.

12. **Falling back gracefully.** If codegen fails (unsupported chain,
    cranelift verifier error, compile timeout), we return `None` from the
    plugin and the scheduler uses the existing decode path. No degradation in
    correctness, just no speedup.

The ordering I'd suggest if we actually built this: 1, 2, 3 (cache only,
no eviction), 6, 7 — that's the minimum viable framework. Everything else is
optimization.

---

## 7. Divan benchmark template

Following `docs/developer-guide/benchmarking.md` and modeled on
`vortex/benches/single_encoding_throughput.rs`.

The bench should compare four configurations on the same input:

1. **Baseline — staged decompress.** Call `array.execute::<Canonical>(&mut ctx)`
   on the unmodified array, letting the current scheduler walk the chain. This
   is what users get today.
2. **Hand-fused.** A Rust function that mirrors what the JIT would emit, written
   by hand. Tells us the ceiling.
3. **JIT — cold.** First call: includes Cranelift compile time. Tells us the
   amortization threshold.
4. **JIT — warm.** Second call onward: the kernel is in the cache. Tells us the
   steady-state win.

We use `BytesCount` + `ItemsCount` per the survey's findings. Parameterize over
bit width and exception density — these are the two axes that change codegen
behavior.

```rust
// vortex-jit/benches/fused_alp_delta_ffor_bp.rs

#![allow(missing_docs)]

use divan::Bencher;
use divan::counter::{BytesCount, ItemsCount};
use mimalloc::MiMalloc;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::sync::LazyLock;

use vortex_array::session::ArraySession;
use vortex_array::{Canonical, IntoArray};
use vortex_session::VortexSession;

// Hypothetical:
use vortex_jit::{JitKernels, JitConfig};

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

const NUM_VALUES: usize = 1 << 20;            // 1M rows
const BIT_WIDTHS: &[u8] = &[3, 7, 11, 17];    // const param sweep
const EXCEPTION_DENSITY: &[f64] = &[0.0, 0.01, 0.05];

static SESSION_BASELINE: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty().with::<ArraySession>()
});

static SESSION_JIT: LazyLock<VortexSession> = LazyLock::new(|| {
    VortexSession::empty()
        .with::<ArraySession>()
        .with::<JitKernels>(JitKernels::new(JitConfig::default()))
});

fn build_chain(
    bit_width: u8,
    exception_density: f64,
) -> vortex_array::ArrayRef {
    let mut rng = StdRng::seed_from_u64(0xC0DE_C0DE);
    // 1. Generate f64s with predictable exponent shape (ALP-friendly).
    // 2. ALP encode -> i64 child + exceptions.
    // 3. FoR encode the i64 with a chosen reference.
    // 4. Delta encode.
    // 5. BitPack at width = bit_width with random patches at exception_density.
    // Exact builder calls follow each encoding's
    // `try_from_primitive_array` / `encode` entry points.
    todo!("see encodings/{alp,fastlanes}/src/.../compress.rs for builders")
}

fn with_throughput<'a, 'b>(
    bencher: Bencher<'a, 'b>,
    n_rows: usize,
) -> Bencher<'a, 'b> {
    bencher
        .input_counter(move |_: &_| ItemsCount::new(n_rows))
        .input_counter(move |_: &_| BytesCount::of_many::<f64>(n_rows))
}

#[divan::bench(args = BIT_WIDTHS, consts = [0.0, 0.01, 0.05])]
fn baseline_canonicalize<const DENSITY_BPS: u32>(bencher: Bencher, bit_width: u8) {
    let density = DENSITY_BPS as f64 / 10_000.0;
    let array = build_chain(bit_width, density);
    with_throughput(bencher, NUM_VALUES)
        .with_inputs(|| (array.clone(), SESSION_BASELINE.create_execution_ctx()))
        .bench_refs(|(a, ctx)| {
            a.clone().execute::<Canonical>(ctx).unwrap()
        });
}

#[divan::bench(args = BIT_WIDTHS)]
fn hand_fused(bencher: Bencher, bit_width: u8) {
    let array = build_chain(bit_width, 0.0);
    let buffers = extract_raw_buffers(&array);
    with_throughput(bencher, NUM_VALUES)
        .with_inputs(|| buffers.clone())
        .bench_refs(|b| hand_fused_alp_for_delta_bp(b));
}

#[divan::bench(args = BIT_WIDTHS)]
fn jit_cold(bencher: Bencher, bit_width: u8) {
    // Per-iteration fresh JIT cache → measures *compile + run*.
    with_throughput(bencher, NUM_VALUES)
        .with_inputs(|| {
            let session = VortexSession::empty()
                .with::<ArraySession>()
                .with::<JitKernels>(JitKernels::new(JitConfig::default()));
            (build_chain(bit_width, 0.0), session.create_execution_ctx())
        })
        .bench_refs(|(a, ctx)| a.clone().execute::<Canonical>(ctx).unwrap());
}

#[divan::bench(args = BIT_WIDTHS)]
fn jit_warm(bencher: Bencher, bit_width: u8) {
    // Shared JIT cache across iterations → measures *cached run only*.
    let array = build_chain(bit_width, 0.0);
    // Warm the cache once outside the timed region.
    {
        let mut ctx = SESSION_JIT.create_execution_ctx();
        let _ = array.clone().execute::<Canonical>(&mut ctx).unwrap();
    }
    with_throughput(bencher, NUM_VALUES)
        .with_inputs(|| (array.clone(), SESSION_JIT.create_execution_ctx()))
        .bench_refs(|(a, ctx)| a.clone().execute::<Canonical>(ctx).unwrap());
}

fn main() { divan::main(); }
```

What to look for in the numbers:

- `baseline` vs `hand_fused` tells us the **fusion ceiling** — the theoretical
  win from removing intermediates. If this is <1.5×, JIT is probably not worth
  it for this chain.
- `jit_warm` vs `hand_fused` tells us the **codegen quality** — how close
  Cranelift gets to a human at this. <0.8× means the IR is leaving cycles on
  the table.
- `jit_cold` vs `jit_warm` tells us the **compile cost** — divided by the
  per-call delta, you get the amortization break-even (in rows). That sets the
  threshold for §6 point 4.
- Sweeping `bit_width` exercises const-folding; sweeping exception density
  exercises the patch path.

---

## Appendix A: research on the inter-module ABI

What follows is what the surveys found about how today's encodings talk to each
other. None of this is a JIT-specific contract — it's the *current* ABI that
any JIT would have to bridge to.

**Encoding identity & dispatch.** Each encoding is a `VTable` (e.g. `ALP`,
`FoR`, `Delta`, `BitPacked`) with a stable `encoding_id()`. The scheduler in
`vortex-array/src/executor.rs:161-292` walks the tree iteratively: encodings
return `ExecutionStep::ExecuteSlot(child_idx, matcher)` to suspend themselves
while a child decodes. Plugins on `(parent_id, child_id)` pairs intercept this
walk via `ArrayKernels` (`vortex-array/src/optimizer/kernels.rs`).

**Type info.** `DType::Primitive(PType, Nullability)` carries the element type
through metadata (`vortex-array/src/dtype/mod.rs:58-73`). `ArrayView::dtype()`
and `ArrayView::encoding_id()` give us everything codegen needs to pick the
right IR type and resolve the right decompression kernel.

**Per-encoding decompression entry points.**
- ALP: `execute_decompress` in
  `encodings/alp/src/alp/decompress.rs:64`. Scalar `iter_mut().for_each` over
  `decode_single` (`encodings/alp/src/alp/mod.rs:253-266`). Not autovectorized.
- Delta: `delta_decompress` in
  `encodings/fastlanes/src/delta/array/delta_decompress.rs:24`. Calls fastlanes
  `Delta::undelta::<LANES>` then `Transpose::untranspose`. SIMD via fastlanes.
- FoR: `decompress` in
  `encodings/fastlanes/src/for/array/for_decompress.rs:48`. Has a
  BitPacked-child fast path at lines 51-78 that calls `fused_decompress` at
  line 81 — **the existing precedent for our JIT design**.
- BitPacked: `unpack_array` in
  `encodings/fastlanes/src/bitpacking/array/bitpack_decompress.rs:26`, which
  walks 1024-element blocks via `BitUnpackedChunks` and delegates per-chunk
  unpack to `UnpackStrategy::unpack_chunk` in
  `encodings/fastlanes/src/bitpacking/array/unpack_iter.rs:22-29`.

**The `UnpackStrategy` trait is the most important prior art.** It is exactly
the inter-module fusion ABI we are talking about — a small interface that
lets one encoding (FoR) inject work into another's hot loop (BitPacked's
per-chunk unpack). It just isn't generalized: only FoR over BitPacked. A JIT
generalizes it by emitting IR for arbitrary chains instead of writing a new
`Strategy` impl per pair.

**Patches.** ALP exceptions
(`encodings/alp/src/alp/compress.rs:307-329`) and BitPacked patches are stored
as `(indices, values)` pairs alongside the encoded payload and applied as a
post-loop scatter. The hand-fused FoR+BitPacked path applies them after the
loop with a closure that re-adds the reference
(`encodings/fastlanes/src/for/array/for_decompress.rs:122-127`). This is the
pattern the JIT should match.

**What is *not* nested today.** The chain `ALP → FoR → Delta → BitPacked` is
aspirational. Per the encoding survey: ALP's `encoded` is a primitive `i32`/`i64`,
not a compressed array. Delta stores bases as a primitive child rather than as
a recursively-compressed FoR. So the JIT prototype has two prerequisites the
encoding survey turned up:
1. Teach the sampling compressor to actually produce nested chains.
2. Teach the JIT to walk those chains via the `ArrayKernels` plugin
   interception.

Both are tractable; both are out of scope for this sketch. Listing them so
they don't get lost.

**No prior JIT exploration in the workspace.** Confirmed via Cargo search:
no `cranelift`, `inkwell`, `llvm`, `wasmtime`, or `iced-x86` dependencies.
This would be the first JIT in the tree.

---

## Appendix B: what this sketch does *not* answer

- Whether Cranelift produces code competitive with `rustc -C opt-level=3
  -C target-cpu=native` for the fastlanes-style 1024-block unpack loops.
  Cranelift's optimizer is weaker than LLVM's; the unpack loops have already
  been hand-tuned upstream. We may discover the JIT loses to the baseline.
  The bench in §7 is the only honest way to find out.
- Whether LLVM (via `inkwell` or `mlir-sys`) would be a better backend for the
  same framework. LLVM gives stronger codegen at higher compile cost. The
  framework abstractions in §6 should not bake in a backend choice.
- Whether persistent on-disk caching of compiled kernels is worth the
  complexity (cranelift_object + signature versioning) vs accepting cold-start
  cost on every process launch.
- How the JIT interacts with `vortex-datafusion` / `vortex-duckdb` query
  pipelines — those have their own expression compilers that might want to
  reach down into the encoding chain rather than going through
  `execute::<Canonical>`.
