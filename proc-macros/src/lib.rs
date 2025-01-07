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

    let read_schema = generate_schema_struct(&typename, &fields);

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

        impl proc_macro_traits::FromAvro for #typename {
            fn read_schema() -> apache_avro::Schema {
                #read_schema
            }
        }
    }
}

/// Generate an Apache Avro schema for a struct type.
fn generate_schema_struct(typename: &syn::Ident, fields: &FieldsNamed) -> proc_macro2::TokenStream {
    // Generate RecordField for each of the struct fields.
    let fields = fields
        .named
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let name = f.ident.clone().unwrap();
            let typ = f.ty.clone();
            quote! {
                apache_avro::schema::RecordField {
                    name: stringify!(#name).to_string(),
                    doc: None,
                    schema: <#typ as proc_macro_traits::FromAvro>::read_schema(),
                    aliases: core::default::Default::default(),
                    default: core::default::Default::default(),
                    // TODO(aduffy): I have no idea what this is.
                    order: apache_avro::schema::RecordFieldOrder::Ignore,
                    position: #idx,
                    custom_attributes: core::default::Default::default(),
                }
            }
        })
        .collect::<Vec<_>>();

    // Generate the RecordSchema
    quote! {
        apache_avro::Schema::Record(apache_avro::schema::RecordSchema {
            name: apache_avro::schema::Name {
                name: stringify!(#typename).to_string(),
                namespace: None,
            },
            fields: vec![#(#fields,)*],
            aliases: core::default::Default::default(),
            doc: core::default::Default::default(),
            lookup: core::default::Default::default(),
            attributes: core::default::Default::default(),
        })
    }
}
