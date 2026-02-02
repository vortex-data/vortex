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
//! // Only compiled in test builds when CUDA is available
//! #[cuda_test]
//! mod tests {
//!     // ...
//! }
//! ```

use std::process::Command;
use std::sync::LazyLock;

use proc_macro::TokenStream;
use quote::quote;
use syn::Item;
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

/// Conditionally compiles the annotated item only in test builds when CUDA is available.
#[proc_macro_attribute]
pub fn cuda_tests(_attr: TokenStream, item: TokenStream) -> TokenStream {
    if *NVCC_AVAILABLE {
        let item = parse_macro_input!(item as Item);
        quote! {
            #[cfg(test)]
            #item
        }
        .into()
    } else {
        TokenStream::new()
    }
}
