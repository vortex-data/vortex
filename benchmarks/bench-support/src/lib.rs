// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Instruction-set gating for Vortex benchmarks.
//!
//! CodSpeed's sharded job measures simulated instruction counts on x86-64. That says little
//! about a hand-written SIMD kernel and nothing at all about Arm. [`isa`] marks a benchmark
//! as belonging to one instruction set, which routes it to a walltime CI leg running on that
//! architecture's silicon.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Ident;
use syn::ItemFn;
use syn::parse_macro_input;

/// The architecture leg an instruction set belongs to.
struct Leg {
    /// Variant tag, matched against `VORTEX_BENCH_VARIANT`.
    variant: &'static str,
    /// `target_arch` the instruction set exists on.
    target_arch: &'static str,
}

const X86: Leg = Leg {
    variant: "x86",
    target_arch: "x86_64",
};
const ARM: Leg = Leg {
    variant: "arm",
    target_arch: "aarch64",
};

/// Resolve an instruction set to the legs that measure it.
///
/// `any` is the arch-neutral case: a scalar baseline or a shipped entry point, measured on
/// every leg so each architecture has something to compare its kernels against. Everything
/// else names one instruction set and so implies exactly one architecture.
fn legs_for(isa: &str) -> Option<Vec<Leg>> {
    match isa {
        "sse2" | "avx2" | "avx512" => Some(vec![X86]),
        "neon" | "sve" => Some(vec![ARM]),
        "any" => Some(vec![X86, ARM]),
        _ => None,
    }
}

/// Measure this benchmark on the CI leg for the given instruction set.
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
/// Accepts `sse2`, `avx2`, `avx512`, `neon`, `sve`, and `any` for benchmarks that are not tied
/// to an instruction set. The `#[cfg(target_arch = ...)]` is implied — an `avx2` benchmark cannot
/// compile for Arm, so writing that out would only create a chance for it to disagree with
/// the tag.
///
/// An untagged benchmark is untouched: it keeps running on the simulation shards under its
/// own name, which is the right home for anything that is not architecture-specific.
#[proc_macro_attribute]
pub fn isa(attr: TokenStream, item: TokenStream) -> TokenStream {
    let isa = parse_macro_input!(attr as Ident);
    let mut function = parse_macro_input!(item as ItemFn);

    let Some(legs) = legs_for(&isa.to_string()) else {
        return syn::Error::new(
            isa.span(),
            format!(
                "unknown instruction set `{isa}`; expected one of sse2, avx2, avx512, neon, sve, any"
            ),
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
    // dependency of the crate, so changing the variant rebuilds the benchmark.
    let name = function.sig.ident.to_string();
    let variants = legs.iter().map(|leg| leg.variant);
    let arches = legs.iter().map(|leg| leg.target_arch);

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
                let variant = env!("VORTEX_BENCH_VARIANT");
                !(variant == "local" #(|| variant == #variants)*)
            },
        )
    };

    quote! {
        #[cfg(any(#(target_arch = #arches),*))]
        #function
    }
    .into()
}
