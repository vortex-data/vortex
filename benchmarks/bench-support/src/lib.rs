// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Support for benchmarks measured once per CPU feature set.
//!
//! Two attributes. [`cpu_features`] marks a benchmark as measured on every walltime leg, and
//! [`main`] goes on the `fn main` of a binary carrying such benchmarks. Which feature sets
//! exist, what each is built with, and where it runs are all in `.github/workflows/codspeed.yml`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::ItemFn;
use syn::ReturnType;
use syn::parse_macro_input;

/// Stack size of the thread [`main`] runs the benchmarks on, matching the 8 MiB Linux gives the
/// main thread so nothing that fit before overflows now.
const BENCH_THREAD_STACK_SIZE: usize = 8 << 20;

/// Measure this benchmark on every walltime CPU-feature leg instead of in simulation.
///
/// Takes no argument: the benchmark runs on all of them. Write it *above* `#[divan::bench]`,
/// whose arguments it fills in — the name is qualified with the leg that produced it, both as
/// a `::` component the CI filter selects on and as a suffix on the leaf name CodSpeed
/// reports under, so the legs report one series each rather than fighting over a shared name.
/// `ignore` takes the benchmark out of the sharded simulation job. A plain `cargo bench` runs
/// it as before, under its bare name.
///
/// ```ignore
/// #[vortex_bench_support::cpu_features]
/// #[divan::bench(args = INPUT_SIZE, sample_size = 256)]
/// fn words_gather_dispatch(bencher: Bencher, len: usize) { /* ... */ }
/// ```
///
/// `sample_size` is required: it is how many times one sample runs the benchmark, and the
/// number reported is the sample's time divided by it. The runner brackets every sample with
/// its own hooks and timestamps, and what that costs the sample that follows is a fixed
/// amount that differs from one host to the next by about a microsecond. Left to itself,
/// divan sizes samples by its first, cold run and settles on one iteration for anything over
/// a few microseconds, so on a bad host every case in a binary read the same ~1.2 µs slower
/// regardless of its length. Pick the size so that a sample lasts at least ~100 µs on the
/// smallest argument: 64 for a 2 µs case, 16 for a 10 µs one, 4 for 50 µs. When the arguments
/// span a wide range, add `sample_count` below the default of 1,000 so the largest one does
/// not take seconds.
///
/// Spell it out in full rather than importing it: benchmark files are read a function at a
/// time, and the path says where the behaviour comes from. The binary's `fn main` carries
/// [`main`], which the walltime legs need for a repeatable layout.
///
/// This is for code that is written once and *compiled* differently per feature set — a
/// shipped entry point that selects its kernel through `cfg(target_feature)`, or a scalar
/// loop whose auto-vectorization depends on the build. A hand-written kernel for one
/// instruction set extension is a different thing: it cannot run on the other legs, so it
/// does not belong here. Keep those on `#[cfg(not(codspeed))]` for local A/B runs.
#[proc_macro_attribute]
pub fn cpu_features(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        let attr = TokenStream2::from(attr);
        return syn::Error::new_spanned(
            attr,
            "`#[cpu_features]` takes no argument; a tagged benchmark runs on every feature-set leg",
        )
        .to_compile_error()
        .into();
    }

    let mut function = parse_macro_input!(item as ItemFn);

    let Some(bench) = function.attrs.iter_mut().find(|attr| {
        attr.path()
            .segments
            .last()
            .is_some_and(|s| s.ident == "bench")
    }) else {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "`#[cpu_features]` must be written directly above the `#[divan::bench]` it applies to",
        )
        .to_compile_error()
        .into();
    };

    let existing: TokenStream2 = match &bench.meta {
        syn::Meta::Path(_) => TokenStream2::new(),
        syn::Meta::List(list) => list.tokens.clone(),
        syn::Meta::NameValue(value) => {
            return syn::Error::new_spanned(
                value,
                "expected `#[divan::bench]` or `#[divan::bench(..)]`",
            )
            .to_compile_error()
            .into();
        }
    };

    let has_option = |option: &str| {
        existing
            .clone()
            .into_iter()
            .any(|token| matches!(&token, proc_macro2::TokenTree::Ident(i) if i == option))
    };

    for reserved in ["name", "ignore"] {
        if has_option(reserved) {
            return syn::Error::new_spanned(
                &existing,
                format!(
                    "`{reserved}` is set by `#[cpu_features]`; remove it from `#[divan::bench]`"
                ),
            )
            .to_compile_error()
            .into();
        }
    }

    if !has_option("sample_size") {
        return syn::Error::new_spanned(
            bench,
            "`#[cpu_features]` needs `sample_size` in `#[divan::bench(..)]`: enough iterations \
             per sample for one sample to last ~100 µs or more, so the runner's per-sample \
             overhead, which varies by host, is amortised (see the attribute's docs)",
        )
        .to_compile_error()
        .into();
    }

    // The default used to be `DIVAN_SAMPLE_COUNT` in the workflow, but a value set there beats
    // one set on the benchmark, and the cases that need thousands of iterations per sample
    // need fewer samples in exchange. divan's own default of 100 samples left the same commit
    // measured twice up to 2.2x apart on the 1,024 element cases; 1,000 holds them to ~6-10%.
    let sample_count = if has_option("sample_count") {
        quote!()
    } else {
        quote!(sample_count = 1000,)
    };

    // Skipping simulation, rather than naming the legs to run on, is what keeps the leg list
    // in the workflow alone. `env!` rather than reading the environment during expansion:
    // rustc records it as a dependency of the crate, so changing legs rebuilds the benchmarks.
    //
    // The leg lands in the name twice, and each copy earns its place. The prefix is a `::`
    // component, which is what the workflow's filter selects on. The suffix is part of the
    // leaf name, which is the only part CodSpeed keeps: its walltime reports identify a
    // benchmark by leaf name and arguments, so without the suffix every leg would report
    // into one `words_gather_dispatch[65536]` series instead of one series each.
    let name = function.sig.ident.to_string();
    let separator = if existing.is_empty() {
        quote!()
    } else {
        quote!(,)
    };
    let bench_path = bench.path().clone();
    bench.meta = syn::parse_quote! {
        #bench_path(
            #existing #separator
            #sample_count
            name = concat!(
                env!("VORTEX_BENCH_PREFIX"),
                #name,
                env!("VORTEX_BENCH_SUFFIX"),
            ),
            ignore = env!("VORTEX_BENCH_VARIANT") == "simulation",
        )
    };

    quote!(#function).into()
}

/// Run the benchmarks on a thread of their own, so where their stack and heap land is set by
/// the binary rather than by the process.
///
/// Write it on `fn main`, which must take no arguments and return `()`:
///
/// ```ignore
/// #[vortex_bench_support::main]
/// fn main() {
///     divan::main();
/// }
/// ```
///
/// Every binary with a [`cpu_features`] benchmark needs it; the workflow refuses to measure
/// one without it. The body runs unchanged on a spawned thread with a fixed-size stack, and a
/// panic there is re-raised on the main thread so the exit status is what it was.
///
/// Some µs-scale cases have two stable timings up to 1.5× apart, and which one a run lands in
/// depends on the layout of the process rather than on the code: `arrow_checked_add_u32` in
/// `vortex-compute` read 13 µs or 20 µs on the neon leg from one run to the next. On the main
/// thread, with the address space randomised, it landed in either about half the time; on a
/// spawned thread it held one timing through every one of those runs, and through a sweep of
/// the environment block's size (which, contrary to the obvious guess, did not move it on
/// either thread). The spawned thread's stack is a fresh mapping and the allocators in use
/// serve it from mappings of their own, so its layout is set by the binary alone.
#[proc_macro_attribute]
pub fn main(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        let attr = TokenStream2::from(attr);
        return syn::Error::new_spanned(attr, "`#[vortex_bench_support::main]` takes no argument")
            .to_compile_error()
            .into();
    }

    let function = parse_macro_input!(item as ItemFn);
    let signature = &function.sig;

    if signature.ident != "main" {
        return syn::Error::new_spanned(
            &signature.ident,
            "`#[vortex_bench_support::main]` goes on `fn main`",
        )
        .to_compile_error()
        .into();
    }
    if !signature.inputs.is_empty()
        || !matches!(signature.output, ReturnType::Default)
        || signature.asyncness.is_some()
    {
        return syn::Error::new_spanned(
            signature,
            "`#[vortex_bench_support::main]` expects `fn main()` with no arguments and no return type",
        )
        .to_compile_error()
        .into();
    }

    let attrs = &function.attrs;
    let vis = &function.vis;
    let body = &function.block;

    // The body becomes the closure, so a `return` in it leaves the benchmarks as before. An
    // explicit stack size rather than the default keeps `RUST_MIN_STACK` from putting the
    // environment back into the layout.
    quote! {
        #(#attrs)*
        #vis fn main() -> ::std::io::Result<()> {
            let benchmarks = ::std::thread::Builder::new()
                .name("vortex-bench".to_owned())
                .stack_size(#BENCH_THREAD_STACK_SIZE)
                .spawn(|| #body)?;
            if let Err(panic) = benchmarks.join() {
                ::std::panic::resume_unwind(panic);
            }
            Ok(())
        }
    }
    .into()
}
