# JIT Compression-Kernel Fusion — Design Sketch

Status: **sketch / RFC**. Not implemented. Lives on the
`claude/jit-compression-kernel-design-g6wGm` branch as a thinking aid.

> **Scope (revised):** there are two halves to a real system — (A) discovering
> a nested encoding tree from data via the sampling compressor, and (B) given
> a typed decompression tree, emit a JIT kernel for it. **(A) is punted** as a
> separate compression-policy problem; this document is about (B). See §8
> for the Stage-IR formulation that follows from that scope cut.

The goal is to evaluate using Cranelift as a runtime code generator that
specializes a *decompression tree* (ALP over FoR over Delta over BitPacked)
into a single fused kernel per chain fingerprint, then to think about what we
would have to build to turn that one prototype into a framework. The first
half is the sketch; the second half is the framework discussion; §8 covers
the Stage IR; the appendix is the prior art we have inside the repo today.

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

## 8. Stage IR — the refinement once tree discovery is punted

Once we accept a typed `DecodeTree` as input, the per-encoding API simplifies
dramatically. Encodings stop emitting Cranelift IR. They emit a **Stage list**,
which is lowered to IR by one central `vortex-jit` pass.

```rust
pub enum DecodeTree {
    Leaf       { buf: BufId, ptype: PType },
    BitPacked  { width: u8, child: Box<DecodeTree>, patches: Option<PatchRef> },
    For        { reference: i64, child: Box<DecodeTree> },
    Delta      { lanes: u8, bases: BufId, child: Box<DecodeTree> },
    Alp        { e: u8, f: u8, child: Box<DecodeTree>, patches: Option<PatchRef> },
}

pub enum Stage {
    UnpackBits   { width: u8, in_t: PType, out_t: PType },
    AddConst     { value: i64, t: PType },
    PrefixSum    { lanes: u8, bases: BufId, t: PType },
    Untranspose  { lanes: u8, t: PType },
    IntToFloat   { in_t: PType, out_t: PType, scale: f64 },
    ApplyPatches { idx: BufId, val: BufId, n: BufId, t: PType }, // post-loop
    StoreOut     { t: PType },
}

pub trait EncodingCodegen {
    fn lower(&self, child: Vec<Stage>, ptype: PType) -> Vec<Stage>;
    fn const_params(&self) -> SmallVec<[u64; 4]>;
}
```

**Lane-form vs buffer-form** is the only cross-stage protocol:

- *Lane-form* — `[Value; 1024]` SSA values. Cheap for `iadd_imm`, `fmul`, any
  elementwise op.
- *Buffer-form* — `*mut T` to a 1024-slot stack scratch. Required when a stage
  needs random access (untranspose) or cross-element flow (prefix sum across
  lane groups).

The framework inserts implicit *materialize* / *load* transitions when
adjacent stages disagree on shape. Cranelift folds the round-trip away when no
permutation sits between them; it can't fold around the untranspose, and
that's exactly where materialization belongs anyway.

**Lowering example** — `Alp(For(Delta(BitPacked)))` for `f64`:

```
UnpackBits{11, u8, i64}    →  AddConst{ref, i64}
→ PrefixSum{16, bases, i64} →  Untranspose{16, i64}
→ IntToFloat{i64, f64, F10[f]*IF10[e]}
→ StoreOut{f64}
[post-loop] ApplyPatches{alp_idx, alp_val, n, f64}
```

**Three kernel ops, one Stage list.** `decompress` / `filter` / `take` share
all stages; only the driver loop and the terminal stage change:

| Op | Driver | Terminal |
|----|--------|----------|
| Decompress | `for block in 0..n` | `StoreOut` linear |
| Filter | `for block in 0..n` | `MaskedCompact` writes lane to `out[ptr]`, bump ptr |
| Take | iterate touched blocks only | `StoreSelective` per index |

`Take` inherits the threshold heuristic from
`encodings/fastlanes/src/bitpacking/compute/take.rs:43-45` as a *lowering*
decision (skip whole blocks with no taken index) rather than a runtime branch.

**Fingerprint becomes trivial.** Hash of the canonical Stage list +
`out_ptype` + `op`. Two trees that lower to the same stages share a kernel.
`For(reference=0, BitPacked)` deduplicates against plain `BitPacked` because
lowering elides `AddConst{0, _}`.

**What changes in the §6 punch list under this refinement:**

| # | Change |
|---|--------|
| 1 | Replaced by `EncodingCodegen` + central Stage→IR lowering. ~20 lines per encoding. |
| 2 | Hash of canonical Stage list. |
| 4 | `stage_list.len() + n_blocks` is the cost-model input. |
| 5 | `ApplyPatches` is just another Stage; lowering inserts it. |
| 7 | Per-stage unit tests; chain tests on top. |
| 9 | Each Stage declares its supported `(in_t, out_t)` matrix. |
| 11 | Filter/Take share the Stage list; only driver + terminal differ. |

Items 3, 6, 8, 10, 12 unchanged.

---

## 9. Extensibility and composability

A framework where adding an encoding requires modifying `vortex-jit` is not a
framework. This section flips §8's Stage *enum* into an open *trait* surface so
new encodings ship in their own crates and compose with everything else.

### Three extension points

```rust
// vortex-jit core — knows nothing about ALP/FoR/etc.

pub trait JitStage: Send + Sync {
    fn tag(&self) -> StageTag;
    fn fingerprint(&self) -> &[u8];
    fn input(&self) -> SmallVec<[Form; 1]>;   // multi-input for tree merges
    fn output(&self) -> Form;
    fn declare(&self, sig: &mut SigBuilder);  // request runtime args / extern syms
    fn emit(&self, cx: &mut EmitCtx<'_>);     // emit IR for one block
}

pub trait EncodingCodegen: Send + Sync {
    fn lower(
        &self,
        children: Vec<Pipeline>,
        ptype: PType,
        params: &[u8],
    ) -> VortexResult<Pipeline>;
}

pub trait JitDriver: Send + Sync {
    fn outer(&self, cx: &mut EmitCtx, body: &mut dyn FnMut(&mut EmitCtx));
    fn terminal(&self, ptype: PType) -> Box<dyn JitStage>;
}
```

`vortex-jit` ships these traits plus `Form`, `Pipeline`, `EmitCtx`, the cache,
and the safety wrapper. It does **not** ship any concrete stages.

### Crate layout

```
vortex-jit/             trait surface, Pipeline, EmitCtx, cache, compiler
vortex-jit-stages/      built-in UnpackBits, AddConst, PrefixSum, ...
encodings/alp/          impl EncodingCodegen for ALP
encodings/fastlanes/    impl EncodingCodegen for {FoR, Delta, BitPacked}
encodings/<your-new>/   impl EncodingCodegen + any new JitStage impls
```

Built-ins are a library, not a privileged set. A third-party crate that needs
`ZigzagDecode` either reuses one in `vortex-jit-stages` or ships its own — the
framework can't tell the difference.

### `EmitCtx` is the contract

```rust
impl<'a> EmitCtx<'a> {
    pub fn fb(&mut self) -> &mut FunctionBuilder<'a>;
    pub fn block_idx(&self) -> Value;
    pub fn lanes_in(&self) -> &Lanes;
    pub fn lanes_out(&mut self, lanes: Lanes);
    pub fn scratch(&mut self, ptype: PType) -> *mut T;
    pub fn runtime_arg(&mut self, key: ArgKey) -> Value;
    pub fn extern_call(&mut self, sym: ExternId, args: &[Value]) -> Value;
}
```

`declare()` runs once at compile time. The framework collects every stage's
requested runtime args and extern symbols, deduplicates, and lays out the
final `unsafe extern "C"` signature. Stages never touch the signature
directly, so they can be authored independently and composed without
coordination.

### Composability — what the framework enforces

`Pipeline::push` validates form compatibility and auto-inserts the cheap
transitions:

```rust
match (prev_output, next_input) {
    // exact match
    (a, b) if a.compatible(b) => /* ok */,
    // lane <-> buffer mismatch -> framework inserts materialize/load
    (Lane(t, l),   Buffer(t, l)) => insert(materialize(t, l)),
    (Buffer(t, l), Lane(t, l))   => insert(load(t, l)),
    // layout mismatch (transposed -> linear) -> require explicit Untranspose
    (Lane(_, FastLanesT(_)), Lane(_, Linear)) => bail!("explicit Untranspose required"),
    // type mismatch -> hard error
    _ => bail!("incompatible forms"),
}
```

The framework owns the *protocol* between stages (form/type/layout matching,
implicit transitions). Stages own the *content* (their IR template). New
stages don't have to know about other stages' shapes; they just declare their
own input and output forms honestly.

### Multi-child encodings — tree, not chain

```rust
pub struct DecodeNode {
    pub encoding: Arc<dyn EncodingCodegen>,
    pub children: Vec<DecodeNode>,
    pub buffers: Vec<BufId>,
    pub ptype: PType,
    pub const_params: Box<[u8]>,
}

pub fn lower(node: &DecodeNode) -> VortexResult<Pipeline> {
    let child_pipelines = node.children.iter()
        .map(lower)
        .collect::<VortexResult<Vec<_>>>()?;
    node.encoding.lower(child_pipelines, node.ptype, &node.const_params)
}
```

Each encoding's `lower` decides how to interleave child pipelines. Dict uses
one child as a values *buffer* and the other as the lane stream feeding a
`DictLookup` stage. DateTimeParts produces a `RecomposeTimestamp` stage with
three lane-form inputs and one output. Multi-input is handled by `JitStage`
returning a `SmallVec<[Form; 1]>` from `input()`.

### Driver extensibility

Same shape. Built-in `DecompressDriver` / `FilterDriver` / `TakeDriver` live
beside the built-in stages. Third parties add `SumDriver`, `CountDriver`,
`HashBuildDriver` without touching encoding crates. The same lowered
`Pipeline` runs under any compatible driver.

### Registration

Same pattern as `ArrayKernels`. Each crate exposes a `register` fn that
installs its codegen/stage/driver into a `JitRegistry` session var. No
`inventory` or `linkme` — deterministic, controllable ordering.

```rust
pub fn register(jit: &mut JitRegistry) {
    jit.register_codegen::<YourEncodingVTable>(YourEncodingCodegen);
    jit.register_stage::<YourCustomStage>();   // optional
    jit.register_driver::<YourCustomDriver>(); // optional
}
```

### Worked example — Zigzag in a third-party crate, zero changes to `vortex-jit`

```rust
// encodings/zigzag/src/jit.rs
pub struct ZigzagDecodeStage { t: PType }

impl JitStage for ZigzagDecodeStage {
    fn tag(&self) -> StageTag { StageTag::new("zigzag::decode", self.t) }
    fn fingerprint(&self) -> &[u8] { &[self.t as u8] }
    fn input(&self) -> SmallVec<[Form; 1]> {
        smallvec![Form::Lane(self.t, Layout::Either)]
    }
    fn output(&self) -> Form { Form::Lane(self.t.signed(), Layout::Same) }
    fn declare(&self, _: &mut SigBuilder) {}
    fn emit(&self, cx: &mut EmitCtx) {
        let lanes = cx.lanes_in().clone();
        let out = lanes.iter().map(|&x| {
            let shifted = cx.fb().ins().sshr_imm(x, 1);
            let lsb     = cx.fb().ins().band_imm(x, 1);
            let neg     = cx.fb().ins().ineg(lsb);
            cx.fb().ins().bxor(shifted, neg)
        }).collect();
        cx.lanes_out(Lanes::Lane(out));
    }
}

pub struct ZigzagCodegen;

impl EncodingCodegen for ZigzagCodegen {
    fn lower(&self, children: Vec<Pipeline>, ptype: PType, _: &[u8])
        -> VortexResult<Pipeline>
    {
        let mut p = children.into_iter().next().unwrap();
        p.push(Box::new(ZigzagDecodeStage { t: ptype }))?;
        Ok(p)
    }
}

pub fn register(jit: &mut JitRegistry) {
    jit.register_codegen::<Zigzag>(ZigzagCodegen);
}
```

That's the whole extension. Zigzag now composes with every other registered
encoding in either direction (over Delta, under Dict, inside DateTimeParts)
without `vortex-jit` knowing anything about it.

### Final framework boundaries

| Owns | Lives in |
|------|----------|
| `JitStage`, `EncodingCodegen`, `JitDriver` traits | `vortex-jit` |
| `Form`, `Layout`, `Pipeline`, `EmitCtx`, `SigBuilder` | `vortex-jit` |
| Cache, fingerprint hashing, safety wrapper, fallback | `vortex-jit` |
| Outer-loop generation, materialize/load transitions | `vortex-jit` |
| Built-in stages (UnpackBits, AddConst, ...) | `vortex-jit-stages` |
| Built-in drivers (Decompress, Filter, Take) | `vortex-jit-stages` |
| Per-encoding `EncodingCodegen` impls | each `encodings/*` crate |
| Per-encoding custom `JitStage` impls (if any) | each `encodings/*` crate |
| Per-encoding registration | each `encodings/*` crate |

The catalog of stages/drivers/codegens is unbounded. The protocol between
them is fixed. That's the line that makes this a framework.

---

## 10. Typing inside `emit` — what other projects do

The Zigzag example in §9 leaves `emit` deliberately under-specified. This
section pins down the typing/error/SIMD/null model and contrasts it with the
prior art.

### Problems with the §9 sketch

1. `lanes_in().clone()` pretends 1024 SSA values are owned data.
2. `Value` is untyped at the Rust level — `sshr_imm` is integer-only, but
   nothing stops a stage from handing it an `f64`.
3. "1024 scalar SSA values per stage" is not how SIMD works on real hardware.
4. No `VortexResult` — a half-built function on a partial-codegen error is
   junk that the framework can't recover from cleanly.

### Three layers, three error windows

| Layer | Job | Enforces |
|-------|-----|----------|
| Composition (`Pipeline::push`) | "Does this stage fit after the previous one?" | `Form` matching — public, declared by stage |
| Emit body (`LaneSlice<T>`) | "Am I calling ops valid for this type?" | Typed newtypes around `Value` |
| Escape hatch (`cx.fb()`) | "Anything Cranelift can express" | CLIF verifier post-construction |

Composition errors fire before any IR is emitted. Type errors fire inside the
stage. Malformed-IR errors fire when the framework verifies the assembled
function. Three error windows means each kind of mistake gets caught at the
earliest layer that can see it.

### The improved Zigzag

```rust
fn emit(&self, cx: &mut EmitCtx<'_>) -> VortexResult<()> {
    let xs: LaneSlice<Int> = cx.take_input().into_int(self.t)?;

    let out = xs.map_chunks(cx, |b, x| {
        let shifted = b.sshr_imm(x, 1);
        let lsb     = b.band_imm(x, 1);
        let neg     = b.ineg(lsb);
        b.bxor(shifted, neg)
    });

    cx.put_output(out.into_lanes());
    Ok(())
}
```

What changed:

- `LaneSlice<Int>` — typed wrapper carrying `PType` + `Vec<Value>` of physical
  SIMD chunks (not 1024 scalars).
- `map_chunks(cx, |b, x|)` iterates physical chunks (i64x4 on AVX2, i64x8 on
  AVX512, scalar on portable build). Framework picks the chunk width.
- `b: IntBuilder` exposes only ops valid for integer SIMD; `b.fdiv(...)`
  doesn't compile.
- `take_input` consumes; `put_output` deposits. Linear ownership.
- `VortexResult<()>` — errors propagate; framework drops the in-progress
  function on `Err`.

### `Lanes` shape

```rust
pub enum Lanes {
    Int(LaneSlice<Int>),
    Float(LaneSlice<Float>),
    Bool(LaneSlice<Bool>),
}
impl Lanes {
    pub fn into_int(self, expected: PType) -> VortexResult<LaneSlice<Int>>;
    pub fn into_float(self, expected: PType) -> VortexResult<LaneSlice<Float>>;
    pub fn into_bool(self) -> VortexResult<LaneSlice<Bool>>;
}
```

Three Rust types instead of one untyped `Vec<Value>`. Generic over `T` only
where it pays — Int/Float/Bool buckets, not per-width wrappers — so `dyn`
trait objects stay ergonomic.

### `EmitCtx` services beyond `fb()`

| Service | Purpose |
|---------|---------|
| `cx.const_int(t, v)`, `cx.const_float(t, v)` | Typed constants; stage doesn't pick `iconst` vs `fconst` |
| `cx.if_then_else(cond, then, else)` | Block plumbing wrapped; no block IDs / sealing for stages |
| `cx.lane_loop(\|cx, i\| ...)` | Runtime loop over lanes; framework picks unroll vs loop from `prefers_unroll()` |
| `cx.scratch(t)` | 1024-slot stack buffer for materialize stages |
| `cx.runtime_arg(key)` | Resolved at signature-layout time from `declare()` |
| `cx.extern_call(sym, args)` | Pre-declared Rust callback (patches, special ops) |
| `cx.set_loc("zigzag::decode::i64")` | Source-loc for `perf-jitdump` |

### Validity / nulls as a side-channel

A `ValidityLane` flows alongside `Lanes`. Compose-only stages (Zigzag,
AddConst) propagate it unchanged by default. Stages that consume or produce
nulls declare it via `consumes_validity() / produces_validity()` on the trait,
parallel to how `declare()` declares runtime args.

### What other projects do

**Cranelift's wasm translator (closest analog).** Untyped `Value`; operand
stack carries WASM types from bytecode; verifier catches mismatches
post-construction. Fast to build, hostile to authors.

**Inkwell (LLVM Rust bindings).** Typed wrappers — `IntValue<'ctx>`,
`FloatValue<'ctx>`. Type errors are Rust compile errors. Maximum safety, but
`'ctx` lifetimes make `dyn Trait` painful and break our extension model.

**MLIR + ODS.** Ops are declarative; verifier auto-generated from declared
input/output types. Heaviest infrastructure, most principled — third-party
dialects compose because the verifier enforces a stable contract.

**Halide `Expr`.** Wrapped IR node with `Type`. Operators check eagerly,
implicit coercion table handles mismatches. Ergonomic but coercion rules are
their own problem.

**XLA / StableHLO.** Typed shapes + element types; every op call returns
`StatusOr<XlaOp>`. Explicit error propagation everywhere; noisy but
predictable.

**Numba.** No IR-layer typing — runs type-inference first, then lowers typed
AST to LLVM. Nicer authoring (write Python), but third parties can't easily
extend the inferencer.

### Where vortex-jit lands

**Closest to MLIR's spirit, Cranelift's mechanics.**

- *Composition layer* (Form-check at `Pipeline::push`) does what MLIR's
  verifier does — third-party extensibility hinges on a stable, declarative
  shape contract.
- *Emit layer* (typed `LaneSlice<T>` newtypes over raw `Value`) takes
  Cranelift's untyped speed and adds inkwell-style guard rails, but only at
  the boundaries that matter (Int vs Float vs Bool). The few-bucket newtype
  set keeps `dyn` trait objects ergonomic.
- *Escape hatch* (`cx.fb()`) is always there — Cranelift's verifier is the
  backstop.

Formal at composition (Form), informal-but-typed at emit (LaneSlice),
informal at IR (raw Cranelift + verifier). That's the trio MLIR enforces
formally end-to-end and Cranelift enforces informally end-to-end; vortex-jit
sits between them.

---

## 11. What JIT implementations optimize for — and where vortex-jit lands

"JIT" labels several very different engineering bets. This section maps the
landscape and pins down which one vortex-jit is making.

### Five regimes

| Regime | Examples | Budget | Goal | How |
|--------|----------|--------|------|-----|
| Peak throughput | LLVM, HotSpot C2, TurboFan | Unbounded | Best assembly | 50+ passes; LICM, vectorize, unroll, alias |
| Compile speed + safety | Cranelift, Singlepass, V8 Liftoff | ~ms / fn | No miscompiles, predictable code | Single-pass, modest peephole, basic regalloc |
| Query compilation | HyPer, Umbra, ClickHouse JIT, Postgres JIT | Per-query | *What counts as one function* | Operator fusion, push-style codegen |
| Schedule search | Halide, TVM, Triton, XLA | Offline | Best compute→hardware mapping | Decouple algorithm/schedule, autotune |
| Type specialization | V8 TurboFan, LuaJIT, PyPy | Amortized | Speculative monomorphization | Profile-guided, type feedback, deopt |

Most JITs aren't optimizing the same thing. Cranelift-in-wasmtime versus
LLVM-in-Photon are nearly opposite engineering bets.

### One-line bets per system

- **LLVM** — best code, eventually. Hundreds of passes. Beat for steady-state codegen.
- **Cranelift** — fast, secure, ok code. Single-pass RA2, fewer passes, verifier as contract. 3–10× faster compile than LLVM, 1.3–2× slower code.
- **HyPer/Umbra** — collapse N operators into one function via push-style codegen. Wins come from IR-construction strategy, not LLVM's passes.
- **ClickHouse JIT** — fuse aggregation, comparison, filter into one LLVM module per query. Wins are dispatch elimination.
- **Postgres JIT** — tuple deforming (const-fold column offsets) + expr eval. Break-even ≈ 1M rows.
- **Halide** — let the expert hand-write the schedule; algorithm is separate.
- **TVM/XLA/Triton** — autotune over schedule space.
- **V8/LuaJIT** — speculate on runtime types, deopt on miss. Amortized over long sessions.

### Where vortex-jit lives

**Mostly query compilation (Regime 3), partly compile-speed + safety
(Regime 2).** Big win is HyPer-style fusion across encoding boundaries:
three or four block loops collapse into one. Cranelift over LLVM is a
deliberate budget choice — we want JIT useful at 1–10ms compile budgets, not
100ms+.

Explicitly out of scope:

- Speculation — encoding metadata gives static types.
- Tracing — chains are statically known from the encoding tree.
- Profile-guided — data shape is in metadata.
- Schedule search — only two decisions (full-unroll-vs-loop,
  materialize-here-vs-defer); heuristics suffice.
- Tiering — could revisit (Cranelift baseline → LLVM optimizing for hot
  chains); skip for v1.

### Concrete optimization targets, in priority order

1. **Fusion across stages — one outer loop, not N.** The HyPer win.
   Everything else is zero without this.
2. **Constant baking from encoding metadata.** `bit_width=11` →
   unrolled band/ushr; `reference=42` → `iadd_imm 42`; ALP `(e, f)` → one
   `f64const`. Wins live in lowering, not the backend.
3. **Materialize/load elision at lane↔buffer transitions.** Framework
   elides round-trips when no permutation sits between; Cranelift's SROA
   cleans the rest.
4. **Vector-width selection.** Framework picks chunk width per target
   (i64x4/x8/scalar) so stages write one body and get SIMD on every ISA.
   Only place we need Cranelift's codegen to match fastlanes' hand-tuned
   intrinsics.
5. **Block-skip in Take.** Lowering inserts a per-block "any taken
   indices?" check before unpack. Reuses
   `bitpacking/compute/take.rs:43-45` threshold as a lowering decision.
6. **Patch inlining vs scatter.** Above some density, per-block merge;
   below, post-loop scatter. Both kernels coexist in cache under
   different fingerprints.
7. **Compile-cost amortization gate.** Cost model in §6 item 4: don't
   JIT chains that can't recoup compile cost. Cranelift's ms-range
   compile makes the break-even much lower than LLVM would.
8. **Cache by canonical stage list.** Equivalent chains share kernels.
   `For(reference=0, BitPacked)` deduplicates against plain `BitPacked`
   because lowering elides AddConst{0}.

### What we don't optimize (and don't need to)

- Function-call inlining in the hot loop — no calls except post-loop
  patch helpers; stages are inline by construction.
- Alias analysis — Vortex buffer ownership guarantees disjointness;
  `MemFlags::trusted()` and move on.
- Bounds-check elision — wrapper validates lengths once before the
  call; in-loop accesses are unchecked.
- Per-call type checks — composition validation runs at `Pipeline::push`,
  not per invocation.
- LICM — encoding params are already hoisted as IR constants by
  construction.

The set of optimizations we care about is small and concrete — six or
seven items — versus LLVM's hundred. That's because the work has been
done upstream by framework structure: fusion via lowering, constants via
metadata, disjointness via ownership. The JIT just has to emit the
obvious code well.

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
