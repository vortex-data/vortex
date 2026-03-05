// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Proc macros for CUDA conditional compilation.
//!
//! These macros simplify gating code based on CUDA availability.
//!
//! # Usage
//!
//! ```ignore
//! use vortex_cuda_macros::{cuda_available, cuda_not_available, cuda_test};
//!
//! // Only compiled when CUDA is available
//! #[cuda_available]
//! fn cuda_only_function() { /* ... */ }
//!
//! // Only compiled when CUDA is NOT available
//! #[cuda_not_available]
//! fn fallback_function() { /* ... */ }
//!
//! // Ignore tests when CUDA is not available
//! #[vortex_cuda_macros::test]
//! async fn my_test() {
//! ...
//! }
//! ```

use std::process::Command;
use std::sync::LazyLock;

use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

/// Cached result of nvcc availability check.
static NVCC_AVAILABLE: LazyLock<bool> = LazyLock::new(|| {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
});

/// Conditionally compiles the annotated item only when CUDA is available.
#[proc_macro_attribute]
pub fn cuda_available(_attr: TokenStream, item: TokenStream) -> TokenStream {
    if *NVCC_AVAILABLE {
        item
    } else {
        TokenStream::new()
    }
}

/// Conditionally compiles the annotated item only when CUDA is not available.
#[proc_macro_attribute]
pub fn cuda_not_available(_attr: TokenStream, item: TokenStream) -> TokenStream {
    if *NVCC_AVAILABLE {
        TokenStream::new()
    } else {
        item
    }
}

/// Test attribute to ignore tests if CUDA isn't available. Supports both sync and async tests (using tokio).
///
/// Must be named `test` to work with frameworks like `rstest`.
#[proc_macro_attribute]
pub fn test(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item = parse_macro_input!(item as syn::ItemFn);
    if *NVCC_AVAILABLE {
        if item.sig.asyncness.is_some() {
            quote! {
                #[tokio::test]
                #item
            }
        } else {
            quote! {
                #[test]
                #item
            }
        }
        .into()
    } else {
        if item.sig.asyncness.is_some() {
            quote! {
                #[tokio::test]
                #[ignore]
                #item
            }
        } else {
            quote! {
                #[test]
                #[ignore]
                #item
            }
        }
        .into()
    }
}
