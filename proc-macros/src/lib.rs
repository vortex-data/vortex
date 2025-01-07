#![allow(unused)]

use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, Fields, FieldsNamed};

#[proc_macro_derive(FromAvro)]
pub fn derive_macro_avro(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let input_span = input.span();
    let name = input.ident;

    let impls = match input.data {
        Data::Struct(s) => match s.fields {
            Fields::Named(fields) => generate_try_from_avrovalue(&name, &fields),
            _ => {
                return quote_spanned! {
                    input_span =>
                    compile_error!("FromAvro can only be derived for named structs")
                }
                .into()
            }
        },
        // AvroSchema is an Enum type.
        Data::Enum(_) => {
            todo!("should impl From<Tuple> for each of the enum variants")
        }

        // unions are unsupported.
        Data::Union(_) => {
            unimplemented!()
        }
    };

    TokenStream::from(impls)
}

fn generate_try_from_avrovalue(
    typename: &syn::Ident,
    fields: &FieldsNamed,
) -> proc_macro2::TokenStream {
    // We get back from AvroValue::Record a Vec<(String, AvroValue)>.
    // Before hand, at compile time we generate code to extract the fields using the given name, and then
    // attempting to cast them each to the correct type.

    let from_avros = fields.named.iter().enumerate().map(|(idx, f)| {
        let name = f.ident.clone().unwrap();
        let typ = f.ty.clone();
        let idx = syn::Index::from(idx);
        let extracted_name = format_ident!("extracted_{}", name);
        quote! {
            let (name, avro_value) = fields.next().expect(format!("record field {}", stringify!(#name)).as_str());
            assert_eq!(name, stringify!(#name), "field name mismatch: expected {} but got {}", stringify!(#name), name);

            // Assign to the given name field.
            let #extracted_name: #typ = <#typ as TryFrom<proc_macro_traits::AvroValue>>::try_from(avro_value)?;
        }
    }).collect::<Vec<_>>();

    let assignments = fields
        .named
        .iter()
        .map(|f| {
            let name = f.ident.clone().unwrap();
            let extracted_name = format_ident!("extracted_{}", name);
            quote! {
                #name: #extracted_name,
            }
        })
        .collect::<Vec<_>>();

    quote! {
        impl TryFrom<proc_macro_traits::AvroValue> for #typename {
            type Error = vortex_error::VortexError;

            fn try_from(value: proc_macro_traits::AvroValue) -> Result<Self, Self::Error> {
                let proc_macro_traits::AvroValue::Record(fields) = value else {
                    vortex_error::vortex_bail!("expected a record");
                };

                let mut fields = fields.into_iter();

                // Extract all of the fields from the fields Vec.
                #(#from_avros)*;

                Ok(Self { #(#assignments)* })
            }
        }
    }
}
