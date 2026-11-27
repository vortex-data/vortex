// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared procedural macros for Vortex.

use proc_macro::TokenStream;

use quote::quote;

/// Compiles a function with the Rust target features of x86-64-v3.
/// This does not perform runtime detection; callers must establish v3 support.
#[proc_macro_attribute]
pub fn target_cpu_v3(args: TokenStream, item: TokenStream) -> TokenStream {
    if !args.is_empty() {
        return quote!(compile_error!("target_cpu_v3 takes no arguments");).into();
    }

    // Match the features guaranteed by fearless_simd's Avx2 token.
    let mut output = TokenStream::from(quote! {
        #[target_feature(enable = "avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,fxsr,lzcnt,movbe,popcnt,xsave")]
    });
    output.extend(item);
    output
}
