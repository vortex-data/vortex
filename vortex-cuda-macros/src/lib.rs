// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Proc macros for vortex-cuda.

use proc_macro::TokenStream;
use quote::quote;
use syn::ItemFn;
use syn::meta::ParseNestedMeta;
use syn::parse::Result;
use syn::parse_macro_input;

struct TestArgs {
    crate_path: syn::Path,
}

impl Default for TestArgs {
    fn default() -> Self {
        Self {
            crate_path: syn::parse_quote!(::vortex_cuda),
        }
    }
}

impl TestArgs {
    fn parse(&mut self, meta: ParseNestedMeta) -> Result<()> {
        if meta.path.is_ident("crate") {
            self.crate_path = meta.value()?.parse()?;
            Ok(())
        } else {
            Err(meta.error("unsupported attribute"))
        }
    }
}

/// A test attribute that automatically skips if nvcc is not available.
///
/// This attribute wraps `#[tokio::test]` and adds a check at the beginning of the test
/// to return early if the CUDA compiler is not installed.
///
/// # Example
///
/// ```ignore
/// #[vortex_cuda::test]
/// async fn test_my_cuda_kernel() {
///     // test body - only runs if nvcc is available
/// }
/// ```
///
/// When used inside the `vortex-cuda` crate itself, use:
///
/// ```ignore
/// #[vortex_cuda::test(crate = crate)]
/// async fn test_my_cuda_kernel() {
///     // test body
/// }
/// ```
#[proc_macro_attribute]
pub fn test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut args = TestArgs::default();
    let args_parser = syn::meta::parser(|meta| args.parse(meta));
    parse_macro_input!(attr with args_parser);

    let input = parse_macro_input!(item as ItemFn);
    let name = &input.sig.ident;
    let block = &input.block;
    let attrs = &input.attrs;
    let vis = &input.vis;
    let crate_path = &args.crate_path;

    let expanded = quote! {
        #(#attrs)*
        #[tokio::test]
        #vis async fn #name() {
            if !#crate_path::has_nvcc() {
                return;
            }
            #block
        }
    };

    TokenStream::from(expanded)
}
