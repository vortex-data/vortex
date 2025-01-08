#![allow(clippy::unwrap_used)]

mod fromavro;
mod schema;
mod toavro;

use proc_macro::TokenStream;

#[proc_macro_derive(FromAvro)]
pub fn derive_from_avro(input: TokenStream) -> TokenStream {
    fromavro::derive_from_avro(input)
}

#[proc_macro_derive(ToAvro)]
pub fn derive_to_avro(input: TokenStream) -> TokenStream {
    toavro::derive_to_avro(input)
}
