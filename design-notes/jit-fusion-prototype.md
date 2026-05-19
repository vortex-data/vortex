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

const NUM_VALUES: usize = 65_536;             // 64 fastlanes blocks; 512 KiB f64 -> L2-resident
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

## 12. Vector fusion at the IR level — and why this is HyPer's pattern

The framework's stage protocol from §10 is the IR-level mechanization of
fastlanes' Rust-level closure-parameter trick, which is itself a
Rust-source mechanization of HyPer-style push codegen. This section pins
that down because it's the single most important concept in the design.

### HyPer's push model

The pull / Volcano model executes `next()` per tuple, with virtual calls
between operators and data round-tripping through memory at every
boundary. Neumann's push model inverts it: operators have a compile-time
`produce()` / `consume()` that emit code, not runtime iterators.

```text
SELECT SUM(x) FROM t WHERE x > 5

scan.produce()    emits:  for tuple in t { filter.consume(tuple) }
filter.consume(t) emits:  if t.x > 5 { sum.consume(t) }
sum.consume(t)    emits:  acc += t.x

  → stitched: for t in t { if t.x > 5 { acc += t.x } }
```

One loop, zero virtual calls. Inter-operator data flow is in SSA values,
not memory. `consume()` is a *continuation*: the child knows what to do
with the value it produced because the parent provided the consumer code
as a callback during codegen.

### Fastlanes already does this in Rust, at the macro level

`fastlanes-0.5.0/src/macros.rs:100-174` — the `unpack!` macro takes a
closure `|idx, elem|` and calls it per row of the bit-unpack. The closure
body is the consumer continuation.

FFoR's fused unpack (`fastlanes-0.5.0/src/ffor.rs:65-69`) plugs a fused op
into exactly that slot:

```rust
for lane in 0..Self::LANES {
    unpack!($T, W, input, lane, |$idx, $elem| {
        output[$idx] = $elem.wrapping_add(reference)   // <-- the fusion
    });
}
```

vs plain unpack (`fastlanes-0.5.0/src/bitpacking.rs:110-114`):

```rust
for lane in 0..Self::LANES {
    unpack!($T, W, input, lane, |$idx, $elem| {
        output[$idx] = $elem                              // <-- no fusion
    });
}
```

Same machinery, different consumer body. Push-style fusion, mechanized at
Rust source.

One level higher,
`encodings/fastlanes/src/for/array/for_decompress.rs:33-46` makes the
strategy runtime-pluggable for one specific pair:

```rust
impl<T: ...> UnpackStrategy<T> for FoRStrategy<T> {
    #[inline(always)]
    unsafe fn unpack_chunk(&self, bit_width: usize, chunk: &[T], dst: &mut [T]) {
        unsafe { FoR::unchecked_unfor_pack(bit_width, chunk, self.reference, dst); }
    }
}
```

That's HyPer's `consume()` callback at the chunk level, hand-wired for
FoR over BitPacked. The JIT generalizes it to all pairs.

### What LLVM emits from `unfor_pack` for u32, W=11

After `#[inline(always)]` propagates the closure body into the `unpack!`
macro expansion, and the outer `for lane in 0..32` auto-vectorizes:

```text
  vpbroadcastd ymm15, dword [reference]          ; broadcast ref once

.outer:                                           ; per 8-lane SIMD chunk
  vmovdqu      ymm0, [packed + 0]                 ; load packed word
  vpsrld       ymm1, ymm0, 0                      ; row 0: shift
  vpand        ymm1, ymm1, ymm_mask_11            ; row 0: mask
  vpaddd       ymm1, ymm1, ymm15                  ; <-- FoR fusion (1 vpaddd)
  vmovdqu      [out + 0], ymm1                    ; row 0: store

  vpsrld       ymm1, ymm0, 11
  vpand        ymm1, ymm1, ymm_mask_11
  vpaddd       ymm1, ymm1, ymm15                  ; <-- FoR fusion
  vmovdqu      [out + 32], ymm1

  vpsrld       ymm1, ymm0, 22                     ; row 2: straddles 2 words
  vpand        ymm1, ymm1, ymm_mask_10
  vmovdqu      ymm2, [packed + 32]
  vpslld       ymm3, ymm2, 10
  vpand        ymm3, ymm3, ymm_mask_1_hi
  vpor         ymm1, ymm1, ymm3
  vpaddd       ymm1, ymm1, ymm15                  ; <-- FoR fusion
  vmovdqu      [out + 64], ymm1
  ; ... 29 more rows ...
```

The fusion: one `vpaddd` between unpack's final `vpor`/`vpand` and the
store. Zero memory round-trip. ~32 vector adds per block — single-digit
ns net cost.

### What the Cranelift equivalent looks like — explicit SIMD, no auto-vectorization

Cranelift's auto-vectorizer is much weaker than LLVM's. The "outer loop
over scalar lanes auto-vectorizes" trick doesn't apply. We emit SIMD
types directly:

```text
function %fused_unpack_for_u32_w11(packed: i64, out: i64, n_blocks: i32, ref: i32) {
block0(v_packed, v_out, v_n, v_ref):
    v_ref_vec = splat.i32x8 v_ref                     ; broadcast once
    v_mask11  = vconst.i32x8 0x7FF; 0x7FF; ...
    jump block1(iconst.i32 0)

block1(v_i):
    v_cond = icmp ult v_i, v_n
    brif v_cond, block2, exit

block2:
    ; row 0
    v_pw0 = load.i32x8 v_pack_b+0
    v_r0  = ushr_imm v_pw0, 0
    v_r0  = band v_r0, v_mask11
    v_r0  = iadd v_r0, v_ref_vec                       ; <<< fusion point
    store.i32x8 v_r0, v_out_b+0

    ; row 1
    v_r1 = ushr_imm v_pw0, 11
    v_r1 = band v_r1, v_mask11
    v_r1 = iadd v_r1, v_ref_vec                        ; <<< fusion
    store.i32x8 v_r1, v_out_b+32

    ; row 2 (straddles)
    v_lo = ushr_imm v_pw0, 22
    v_lo = band v_lo, v_mask10
    v_pw1 = load.i32x8 v_pack_b+32
    v_hi = ishl_imm v_pw1, 10
    v_hi = band v_hi, v_mask1
    v_r2 = bor v_lo, v_hi
    v_r2 = iadd v_r2, v_ref_vec                        ; <<< fusion
    store.i32x8 v_r2, v_out_b+64

    ; ... 29 more rows ...

    jump block1(iadd_imm v_i, 1)
}
```

Same instruction sequence as LLVM, emitted explicitly. Cranelift's
codegen for `i32x8 iadd` is the same `vpaddd` LLVM emits. **The codegen
quality gap closes when we hand-write the vector IR** — the gap exists
because LLVM does inference; we sidestep by being explicit.

### How the framework mechanizes vector fusion

The stage protocol from §10 *is* the IR-level form of fastlanes' closure
trick. Chunks flow between stages as SSA Values, exactly the way `$elem`
flows through the closure body.

```rust
impl JitStage for BitPackedStage {
    fn output(&self) -> Form {
        Form::Lane(self.t, Layout::FastLanesTransposed(self.lanes))
    }
    fn emit(&self, cx: &mut EmitCtx) -> VortexResult<()> {
        let mut chunks = Vec::with_capacity(32);
        for row in 0..32 {
            chunks.push(self.emit_one_row(cx, row)?);   // i32x8 Value, no store
        }
        cx.put_output(Lanes::Int(LaneSlice::from_chunks(self.t, chunks)));
        Ok(())
    }
}

impl JitStage for ForStage {
    fn input(&self) -> SmallVec<[Form; 1]> {
        smallvec![Form::Lane(self.t, Layout::Either)]
    }
    fn output(&self) -> Form { Form::Lane(self.t, Layout::Same) }
    fn emit(&self, cx: &mut EmitCtx) -> VortexResult<()> {
        let xs = cx.take_input().into_int(self.t)?;
        let ref_vec = cx.const_broadcast_int(self.t, self.reference);
        let out = xs.map_chunks(cx, |b, x| b.iadd(x, ref_vec));   // one vpaddd per chunk
        cx.put_output(out.into_lanes());
        Ok(())
    }
}

impl JitStage for StoreOutStage {
    fn input(&self) -> SmallVec<[Form; 1]> {
        smallvec![Form::Lane(self.t, Layout::Linear)]
    }
    fn emit(&self, cx: &mut EmitCtx) -> VortexResult<()> {
        let xs = cx.take_input().into_int(self.t)?;
        xs.store_to(cx, cx.runtime_arg(ArgKey::Out)?);   // one vmovdqu per chunk
        Ok(())
    }
}
```

When the framework chains these three stages into one Cranelift function,
BitPacked's output `Vec<Value>` is consumed by FoR's `take_input` directly
— **no IR emitted in between, no memory touched, no temp buffer**. The
inner IR is identical to the hand-fused version above:

```text
v_r0 = ushr_imm v_pw0, 0
v_r0 = band     v_r0, v_mask11
v_r0 = iadd     v_r0, v_ref_vec      ; ← BitPacked produced v_r0; FoR consumed it
                                     ;   and produced this iadd; nothing in between.
store.i32x8 v_r0, v_out_b+0          ; ← StoreOut consumed FoR's output.
```

HyPer's produce/consume continuation, written in Cranelift IR builders
instead of C++ string templates. SSA Values are exactly the right
currency for both compiler-construction continuations and SIMD-vector
register-resident fusion — the concepts collapse to the same mechanism.

### Adding a new stage that vector-fuses

A Zigzag stage that wants to fuse over the FoR output:

```rust
impl JitStage for ZigzagStage {
    fn input(&self)  -> SmallVec<[Form; 1]> {
        smallvec![Form::Lane(self.t, Layout::Either)]
    }
    fn output(&self) -> Form { Form::Lane(self.t.signed(), Layout::Same) }
    fn emit(&self, cx: &mut EmitCtx) -> VortexResult<()> {
        let xs = cx.take_input().into_int(self.t)?;
        let out = xs.map_chunks(cx, |b, x| {
            let shifted = b.sshr_imm(x, 1);
            let lsb     = b.band_imm(x, 1);
            let neg     = b.ineg(lsb);
            b.bxor(shifted, neg)            // 4 extra vector ops per chunk
        });
        cx.put_output(out.into_lanes());
        Ok(())
    }
}
```

Chained as `Zigzag(For(BitPacked))`, the assembled inner IR per chunk:

```text
v_unpacked  = ... ushr / band / bor ...           ; from BitPacked
v_for       = iadd      v_unpacked, v_ref_vec     ; from FoR
v_shifted   = sshr_imm  v_for, 1                  ; from Zigzag
v_lsb       = band_imm  v_for, 1
v_neg       = ineg      v_lsb
v_zigzag    = bxor      v_shifted, v_neg
store.i32x8 v_zigzag, ...                         ; from StoreOut
```

Same code a human would write if hand-fusing Zigzag+FoR+BitPacked,
generated mechanically because each stage's `take_input`/`put_output` is
the IR-level form of HyPer's produce/consume.

---

## 13. Chunks vs produce/consume, and runtime estimates at 65k

### Two authoring APIs, identical IR

| | Approach A: chunks | Approach B: produce/consume |
|--|-------------------|-----------------------------|
| Stage sees | `LaneSlice<T>` = Vec of SSA Values for one block | One chunk at a time, via continuation callback |
| Granularity | 1024-lane block (fat) | 8/16-lane SIMD chunk (thin) |
| Multi-input | Trivial — take multiple Vecs | "Pipeline breaker" — must materialize one |
| `dyn Trait` | Works directly | Continuation types fight trait objects |
| Unit test | Hand-build stub Vec, call emit, inspect output | Need a mock producer to fire callbacks |
| IR after Cranelift inlining | Identical | Identical |

**Recommendation: Approach A**, with `cx.lane_loop(\|cx, i\| ...)` as the
escape hatch when a stage genuinely wants a runtime loop.

The HyPer "no memory between stages" property is preserved either way: a
`Vec<Value>` held during codegen is a list of SSA Value references in
*compiler* memory, never in the emitted program's memory. The only loads
and stores in the emitted code are at the leaves (BitPacked reading
packed bytes) and the terminal (`StoreOut` writing canonical output).
Everything in between is register-resident SSA.

Approach A wins for Vortex specifically because:

1. Multi-input encodings (Dict, DateTimeParts, Sparse) break B — HyPer
   itself materializes at multi-input boundaries.
2. B's `for<'a> FnMut(&mut EmitCtx<'a>, Value)` continuations don't
   compose with `dyn JitStage`; §9 extensibility collapses.
3. A unit-tests trivially with a hand-built `Vec<Value>` stub; B needs
   a mock producer.
4. A's 1024-lane granularity matches fastlanes' existing
   `UnpackStrategy::unpack_chunk` ABI — it's what's already there, lifted
   to IR.

### Runtime estimates at 65k

All numbers below are first-principles estimates for a modern x86 core
(AVX-512, DDR5), 65,536-element f64 column, chain
`ALP(e=10,f=2) → FoR(ref=42) → Delta → BitPacked(W=11)`.
**Not measured. The §7 bench is the only honest way to confirm.** Public
fastlanes throughput numbers used as anchors.

**Sizing.** Output 512 KiB → L2-resident, not L1. Compressed payload at
W=11 ≈ 88 KiB → fits in L1. This is an L2-write-bandwidth workload, not
a DRAM-bandwidth one.

| Config | Per-call | Notes |
|--------|----------|-------|
| Current Vortex, full chain | 150–300 μs | 4 passes through L2; ALP scalar loop alone is ~175 μs |
| Current Vortex, FoR+BP only (existing fusion) | 30–50 μs | One fused pass — the partial fusion that exists today |
| AOT hand-tuned (ceiling) | 30–50 μs | All 4 layers fused, intrinsics, no temps; Delta prefix-sum bottleneck |
| **JIT warm** (cache hit) | 35–60 μs | 80–90% of AOT; gap is Cranelift's weaker peephole |
| **JIT cold** (first call) | 1–3 ms | Cranelift compile dominates; per-call cost = warm |

**Where the wins come from.** Two effects stack:

- **ALP vectorization (~5–7×).** Current scalar
  `iter_mut().for_each(decode_single)` doesn't autovectorize because the
  function-call shape blocks LLVM. JIT emits `vcvtdq2pd` + `vmulpd`
  directly. This is the *biggest single win*, and it's not really about
  fusion — it's about the JIT not having to fight LLVM's vectorizer.
- **Fusion (~3× across 3 eliminated passes).** Each intermediate buffer
  pass costs ~30 μs at L2 write bandwidth. Current chain has 3 such
  passes; JIT/AOT have 0.

**Bottleneck shifts by config:**

- Current Vortex full chain → ALP scalar loop
  (`encodings/alp/src/alp/mod.rs:253-261`).
- AOT / JIT → Delta's 16-way per-lane prefix-sum carry chain.
- Memory passes → L2 write bandwidth.

### Amortization at this size

Per-call savings (JIT warm vs current full chain): **~150 μs**.
Cranelift compile cost: **~2 ms** for a 4-stage chain.

**Break-even: ~13 calls of the same chain.**

Single 65k array used once → JIT loses. Same chain used across a parquet
column-set (e.g., 200 columns at the same encoding shape) → JIT wins
trivially. Cost-model gate (§6 item 4) needs both `chain_length >= 2`
*and* either `n_blocks × stage_count > threshold` or
`prior_uses_of_chain >= 1` (the cache-hit case).

### Why 65k as the bench size

- 64 fastlanes blocks — large enough to exercise the block loop but
  small enough that one block isn't the entire run.
- 512 KiB f64 output — L2-resident, isolates compute cost from DRAM
  bandwidth.
- Compile-amortization is *borderline* here, which makes the cold/warm
  spread visible in the bench. 1M would hide compile cost; 1k would
  hide steady-state cost.
- Matches a realistic per-column working set for analytic queries.

---

## 14. v0 implementation results

The `vortex-jit` crate is a working v0 of the framework. ~600 lines of Rust,
real Cranelift codegen, three composed pipelines tested against reference
Rust implementations, divan bench at 65k.

### What's implemented

- `JitStage` trait, `Pipeline` with form-compatibility validation,
  `EmitCtx` with `take_input`/`put_output`/`map_chunks` (Approach A).
- `Compiler` driver: declare-pass collects runtime args + externs;
  build-pass emits one Cranelift function with the outer block loop and
  post-loop tail.
- Built-in stages: `LoadIn`, `ForAdd`, `DeltaPrefixSum`, `StoreOut`,
  `ApplyPatchesPostLoop` (post-loop extern call into a Rust helper).
- Three end-to-end tests pass against reference implementations:
  - `LoadIn → ForAdd → StoreOut`
  - `LoadIn → DeltaPrefixSum → ForAdd → StoreOut`
  - `LoadIn → ForAdd → StoreOut + [PostLoop] ApplyPatches`

### What the IR shows

For `LoadIn → ForAdd(7) → StoreOut` at 4-lane block size, the emitted
Cranelift IR is exactly the fusion pattern from §12:

```text
block2:
    v11 = load.i32 v10        ; LoadIn lane 0
    v12 = load.i32 v10+4      ; LoadIn lane 1
    v13 = load.i32 v10+8
    v14 = load.i32 v10+12
    v15 = iconst.i32 7        ; ForAdd reference (baked as constant)
    v16 = iadd v11, v15       ; <-- fusion: load result flows straight into iadd
    v17 = iadd v12, v15
    v18 = iadd v13, v15
    v19 = iadd v14, v15
    store v16, v24            ; <-- store consumes iadd result, no intermediate
    store v17, v24+4
    store v18, v24+8
    store v19, v24+12
```

Zero intermediate buffer. The reference is baked as `iconst.i32 7`. SSA
Values flow load → iadd → store directly. This is HyPer push-style in
Cranelift IR.

### Measured numbers at 65k (1024-lane blocks, 64 blocks)

Pipeline: `LoadIn → DeltaPrefixSum → ForAdd(42) → StoreOut` on `i32`.

| Config | Mean | Throughput |
|--------|------|------------|
| `staged_rust` (autovec'd Rust reference) | 52 μs | 5.1 GB/s |
| `jit_warm` (cached Cranelift, **scalar** IR) | 106 μs | 2.5 GB/s |
| `jit_cold` (full compile + run) | 124 ms | 0.002 GB/s |

**The scalar JIT loses 2× to autovec'd Rust.** This is exactly §11's
prediction landing precisely:
- LLVM autovec emits `vpaddd ymm` for the `tmp[i].wrapping_add(42)` loop
  in `staged_rust`. Cranelift's scalar IR emits one `iadd` per lane.
- The §11 framing was clear: Cranelift's auto-vectorizer is weak, so v0
  must emit explicit SIMD types (`i32x8`, `i32x4`) to be competitive. v0
  doesn't yet — the `LaneSlice` carries one SSA Value per scalar lane.

**Compile cost is 40-50× higher than the §13 estimate** (124 ms vs 1-3
ms). For 4 stages with 1024-lane unrolled block bodies, the IR
construction is heavy — ~16k IR insts per block × 64 blocks ≈ 1M IR
insts to lay out and register-allocate. The §13 estimate assumed
~200 IR insts; reality is two orders of magnitude bigger.

### What this measurement actually validates

- **The framework's composition mechanics work.** Three different chains
  built from the same five stages, all pass differential tests.
- **The fusion pattern is real in the emitted IR.** No intermediate
  buffers between stages; constants baked in; HyPer's "no memory
  between operators" achieved.
- **Patches as a post-loop extern call works end-to-end.** The Cranelift
  module resolves the registered Rust symbol; the call IR is emitted
  cleanly in `block3`; the Rust helper executes correctly.

### What this measurement does NOT yet validate

- **Whether the JIT can beat autovec'd Rust.** Not at scalar IR — the
  measurement confirms it doesn't. Next phase: emit `i32x8`/`i32x4`
  vector IR in `LaneSlice`. The trait surface, `map_chunks`, and
  `Form::Lane(_, Layout::Linear)` already accommodate this; the change
  is local to `emit.rs` and the stage implementations.
- **Whether the compile budget is realistic.** At 124 ms for 4 stages
  × 1024 unroll, JIT is only viable for very-hot chains used across
  many columns. Two mitigations:
  1. Emit a runtime loop instead of full unroll for large block sizes
     (the `cx.lane_loop` hook in §10) — reduces IR size by ~Nx where
     N is the block size.
  2. Use vector chunks (~32-64 per block at i32x8) rather than scalar
     lanes (1024 per block).

### Path to making the JIT win

In priority order:

1. **Emit `i32x8`/`i32x4` vector chunks instead of scalar lanes.** Same
   trait surface; `LaneSlice::chunks` holds vector-typed Values instead
   of scalar-typed. ~80% of the gap closes here.
2. **Loop the per-block body for large blocks.** Cuts compile cost by
   ~100× without affecting steady-state perf.
3. **Constant-fold the bit-width unpack pattern in BitPacked** (when we
   add it). Per §11, this is where the constant-baking JIT wins shine.

None of these change the framework's public surface. The §9/10/12
design holds; v0 just leaves the SIMD codegen for the next pass.

---

## 15. v1: SIMD lift, and a structural Cranelift limit

v1 adds the `i32x4` / `i64x2` / `f32x4` / `f64x2` SIMD path that §14
listed as the priority-one optimization. The trait surface is unchanged.
The only changes are:

- `PType::simd_type()` / `simd_lanes()` return the 128-bit vector type
  for the primitive.
- `LaneSlice` gains a `lanes_per_chunk: u32` field. `1` for scalar mode,
  `simd_lanes()` for SIMD mode.
- `EmitCtx` gains `load_chunk` / `store_chunk` / `splat` methods.
- `LoadIn`, `ForAdd`, `StoreOut` emit SIMD ops by default.
- `DeltaPrefixSum` keeps a serial carry chain. When it sees a SIMD input
  it extracts lanes, runs the serial prefix sum, re-inserts. Loses the
  SIMD benefit locally; correctness preserved.

### IR proof — SIMD fusion in one block

`LoadIn → ForAdd(7) → StoreOut` at BLOCK=4 emits:

```text
block2:
    v10 = iadd.i64 v0, v9                      ; in_ptr + block_offset
    v11 = load.i32x4 notrap aligned v10        ; one vector load (was 4 scalars)
    v12 = iconst.i32 7
    v13 = splat.i32x4 v12                      ; broadcast reference
    v14 = iadd v11, v13                        ; one vector iadd (was 4 scalars)
    v19 = iadd.i64 v1, v18                     ; out_ptr + block_offset
    store notrap aligned v14, v19              ; one vector store (was 4 scalars)
    jump block1(v21)
```

Four scalar load-iadd-store triples collapsed into one each. The
reference is baked as `iconst.i32 7` and broadcast once via `splat`.
That's exactly the fastlanes/HyPer fusion pattern from §12, now in
real Cranelift IR.

### Measured at 65k

Two pipelines, three configs each:

| Pipeline | Config | Mean | Throughput |
|----------|--------|------|------------|
| `LoadIn → ForAdd → StoreOut` | `staged_rust` (LLVM, AVX-512) | 7.8 μs | **33.6 GB/s** |
| `LoadIn → ForAdd → StoreOut` | `jit_warm` (Cranelift, SSE2) | 16.1 μs | 16.3 GB/s |
| `LoadIn → ForAdd → StoreOut` | `jit_cold` (compile + run) | 5.0 ms | (compile-bound) |
| `LoadIn → Delta → ForAdd → StoreOut` | `staged_rust` | 43.3 μs | 6.1 GB/s |
| `LoadIn → Delta → ForAdd → StoreOut` | `jit_warm` | 66.5 μs | 3.9 GB/s |
| `LoadIn → Delta → ForAdd → StoreOut` | `jit_cold` | 16.2 ms | |

### Wins vs v0

- `jit_warm` for `LoadIn → ForAdd → StoreOut` went from "not measured
  (would have been ~50+ μs scalar)" to **16 μs**. Roughly 3-4× speedup
  from the SIMD lift alone.
- `jit_warm` for `Delta → ForAdd` went from **106 μs → 67 μs** (~1.6×).
- `jit_cold` went from **124 ms → 16 ms** (8×) — fewer per-block IR
  ops, less work for Cranelift's regalloc.

### What we don't yet beat: LLVM autovec

LLVM autovec at AVX-512 stays 2× ahead on both pipelines. The exact gap
matches the SIMD-width ratio: 512-bit (16 lanes) vs 128-bit (4 lanes) =
4× nominal, but memory bandwidth bounds both, so the *effective* ratio
is ~2×.

I verified Cranelift's 128-bit cap directly by trying `I32X8`:

```
compile: Other error: define_function: Compilation error:
Unsupported feature: Unexpected SSA-value type: i32x8
```

Cranelift's x86_64 backend supports 128-bit SIMD universally but does
not yet support 256-bit (AVX2) or 512-bit (AVX-512) codegen, even on
hosts that have those features. This is a structural limit of the
backend, not the framework. The path to closing the gap is one of:

1. **Wait for Cranelift wider-SIMD support.** Tracked upstream; would
   automatically widen `simd_type()` once available with no framework
   changes.
2. **Switch backend to LLVM (via inkwell or mlir-sys).** Multi-month
   project; would close the gap to ~0 but adds a 50-100× heavier
   compile-time dependency.
3. **Where it matters most, emit 2x or 4x parallel 128-bit ops per
   logical chunk** and rely on Cranelift's register allocator to keep
   them in flight. Manual unrolling at the IR level. Speculative; would
   need measurement.

### What this doesn't mean

It doesn't mean the framework loses. It means *this particular
bench, on a host with AVX-512, against LLVM autovec* loses by 2×.
Specifically:

- Where LLVM can't autovec (multi-stage chains with unpredictable
  control flow, mixed types, function-call shapes — exactly ALP's
  `iter_mut().for_each(decode_single)` from
  `encodings/alp/src/alp/mod.rs:253-261`), the JIT is free of that
  block and wins.
- Where compile cost amortizes (column-set scans where one kernel
  serves many columns), the cold-path penalty disappears.
- The fusion across encoding boundaries is real and present in the IR;
  the 128-bit limit just means we don't double the win.

### What this validates

The framework works end-to-end:
- Pipeline composition with form validation
- Cranelift SIMD codegen
- Patch-via-extern post-loop calls
- Correctness against reference Rust on every pipeline tested

The path to "match LLVM" runs through Cranelift backend improvements or
a backend swap. The path to "beat Vortex's current scalar ALP" remains
clear — emit SIMD IR for the ALP decompress (int→float convert +
multiply) and you immediately win 5-7× because the current Rust code
can't autovec through the function-call shape. That's the next
implementation step that the v1 framework directly enables.

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
