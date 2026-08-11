// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Instruction-set gating for Vortex benchmarks.
//!
//! CodSpeed's sharded job measures simulated instruction counts on x86-64. That says little
//! about a hand-written SIMD kernel and nothing at all about Arm. [`isa`] marks a benchmark
//! as belonging to one instruction set, which routes it to the CI leg that builds with that
//! instruction set enabled and measures walltime on silicon that has it.
//!
//! One leg per instruction set, so what a leg measures is what its name says: an AVX-512
//! benchmark is compiled `+avx512f,+avx512bw` and run on an AVX-512 machine, not compiled
//! for AVX2 and steered into an AVX-512 kernel at runtime.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Ident;
use syn::ItemFn;
use syn::parse_macro_input;

/// An instruction set, and so a CI leg: one build, one walltime run.
struct Isa {
    /// Leg tag, matched against `VORTEX_BENCH_VARIANT` and used to qualify benchmark names.
    /// Every tag here needs a matching entry in `.github/workflows/codspeed.yml`.
    tag: &'static str,
    /// The `target_arch` this instruction set exists on.
    target_arch: &'static str,
}

const ISAS: &[Isa] = &[
    Isa {
        tag: "sse2",
        target_arch: "x86_64",
    },
    Isa {
        tag: "avx2",
        target_arch: "x86_64",
    },
    Isa {
        tag: "avx512",
        target_arch: "x86_64",
    },
    Isa {
        tag: "neon",
        target_arch: "aarch64",
    },
    Isa {
        tag: "sve",
        target_arch: "aarch64",
    },
];

/// Resolve the argument of `#[isa(..)]` to the legs that measure the benchmark.
///
/// `any` is the arch-neutral case — a scalar baseline, or a shipped entry point — measured on
/// every leg, so each build has something of its own to compare its kernels against.
fn legs_for(isa: &str) -> Option<Vec<&'static Isa>> {
    if isa == "any" {
        return Some(ISAS.iter().collect());
    }
    ISAS.iter().find(|leg| leg.tag == isa).map(|leg| vec![leg])
}

/// Measure this benchmark on the CI leg that builds for the given instruction set.
///
/// Must sit *above* `#[divan::bench]`, whose arguments it fills in: the benchmark's name is
/// qualified with the leg that produced it, and `ignore` is set so the benchmark is skipped
/// on every other leg. A plain `cargo bench` runs it regardless.
///
/// ```ignore
/// #[isa(avx2)]
/// #[divan::bench(args = INPUT_SIZE)]
/// fn words_gather_avx2(bencher: Bencher, len: usize) { /* ... */ }
/// ```
///
/// Accepts `sse2`, `avx2`, `avx512`, `neon`, `sve`, and `any` for benchmarks that are not
/// tied to an instruction set. The `#[cfg(target_arch = ...)]` is implied — an `avx2`
/// benchmark cannot compile for Arm, so writing that out would only create a chance for it
/// to disagree with the tag.
///
/// An untagged benchmark is untouched: it keeps running on the simulation shards under its
/// own name, which is the right home for anything that is not architecture-specific.
#[proc_macro_attribute]
pub fn isa(attr: TokenStream, item: TokenStream) -> TokenStream {
    let isa = parse_macro_input!(attr as Ident);
    let mut function = parse_macro_input!(item as ItemFn);

    let Some(legs) = legs_for(&isa.to_string()) else {
        let known = ISAS
            .iter()
            .map(|leg| leg.tag)
            .collect::<Vec<_>>()
            .join(", ");
        return syn::Error::new(
            isa.span(),
            format!("unknown instruction set `{isa}`; expected one of {known}, any"),
        )
        .to_compile_error()
        .into();
    };

    let Some(bench) = function.attrs.iter_mut().find(|attr| {
        attr.path()
            .segments
            .last()
            .is_some_and(|s| s.ident == "bench")
    }) else {
        return syn::Error::new(
            isa.span(),
            "`#[isa]` must be written directly above the `#[divan::bench]` attribute it applies to",
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

    for reserved in ["name", "ignore"] {
        if existing
            .clone()
            .into_iter()
            .any(|token| matches!(&token, proc_macro2::TokenTree::Ident(i) if i == reserved))
        {
            return syn::Error::new_spanned(
                &existing,
                format!("`{reserved}` is set by `#[isa]`; remove it from `#[divan::bench]`"),
            )
            .to_compile_error()
            .into();
        }
    }

    // `env!` rather than reading the environment during expansion: rustc records it as a
    // dependency of the crate, so changing legs rebuilds the benchmarks.
    let name = function.sig.ident.to_string();
    let tags = legs.iter().map(|leg| leg.tag);

    let mut arches: Vec<&str> = legs.iter().map(|leg| leg.target_arch).collect();
    arches.dedup();

    let separator = if existing.is_empty() {
        quote!()
    } else {
        quote!(,)
    };
    let bench_path = bench.path().clone();
    bench.meta = syn::parse_quote! {
        #bench_path(
            #existing #separator
            name = concat!(env!("VORTEX_BENCH_PREFIX"), #name),
            ignore = {
                let leg = env!("VORTEX_BENCH_VARIANT");
                !(leg == "local" #(|| leg == #tags)*)
            },
        )
    };

    quote! {
        #[cfg(any(#(target_arch = #arches),*))]
        #function
    }
    .into()
}
