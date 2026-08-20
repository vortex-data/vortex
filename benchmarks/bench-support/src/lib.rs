// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Support for benchmarks measured once per CPU feature set.
//!
//! One attribute, [`cpu_features`]. Which feature sets exist, what each is built with, and
//! where it runs are all in `.github/workflows/codspeed.yml`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::ItemFn;
use syn::parse_macro_input;

/// Measure this benchmark on every walltime CPU-feature leg instead of in simulation.
///
/// Takes no argument: the benchmark runs on all of them. Write it *above* `#[divan::bench]`,
/// whose arguments it fills in — the name is qualified with the leg that produced it, so the
/// legs report one series each rather than fighting over a shared name, and `ignore` takes
/// the benchmark out of the sharded simulation job. A plain `cargo bench` runs it as before,
/// under its bare name.
///
/// ```ignore
/// #[vortex_bench_support::cpu_features]
/// #[divan::bench(args = INPUT_SIZE)]
/// fn words_gather_dispatch(bencher: Bencher, len: usize) { /* ... */ }
/// ```
///
/// Spell it out in full rather than importing it: benchmark files are read a function at a
/// time, and the path says where the behaviour comes from.
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

    for reserved in ["name", "ignore"] {
        if existing
            .clone()
            .into_iter()
            .any(|token| matches!(&token, proc_macro2::TokenTree::Ident(i) if i == reserved))
        {
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

    // Skipping simulation, rather than naming the legs to run on, is what keeps the leg list
    // in the workflow alone. `env!` rather than reading the environment during expansion:
    // rustc records it as a dependency of the crate, so changing legs rebuilds the benchmarks.
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
            name = concat!(env!("VORTEX_BENCH_PREFIX"), #name),
            ignore = env!("VORTEX_BENCH_VARIANT") == "simulation",
        )
    };

    quote!(#function).into()
}
