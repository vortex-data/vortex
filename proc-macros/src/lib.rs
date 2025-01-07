#![allow(clippy::unwrap_used)]

use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, Fields, FieldsNamed};

#[proc_macro_derive(FromAvro)]
pub fn derive_from_avro(input: TokenStream) -> TokenStream {
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
    let from_avros = fields.named.iter().map(|f| {
        let name = f.ident.clone().unwrap();
        let typ = f.ty.clone();

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

    let read_schema = generate_schema_struct(typename, fields);

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

#[proc_macro_derive(ToAvro)]
pub fn derive_to_avro(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let input_span = input.span();
    let name = input.ident;

    let to_avro_impl = match input.data {
        Data::Struct(s) => match s.fields {
            Fields::Named(fields) => generate_to_avro_value(&name, &fields),
            _ => {
                return quote_spanned! {
                    input_span =>
                    compile_error!("ToAvro can only be derived for named structs")
                }
                .into()
            }
        },
        Data::Enum(_) => {
            todo!("should impl ToAvro for each of the enum variants")
        }
        Data::Union(_) => {
            unimplemented!()
        }
    };

    // panic!("throw: {}", to_avro_impl.to_string());
    to_avro_impl.into()
}

fn generate_to_avro_value(typename: &syn::Ident, fields: &FieldsNamed) -> proc_macro2::TokenStream {
    // Generate the From<$typename> for AvroValue.

    let record_fields = fields.named.iter().map(|f| {
        let name = f.ident.clone().unwrap();
        let typ = f.ty.clone();
        quote! {
            (stringify!(#name).to_string(), <#typ as Into<proc_macro_traits::AvroValue>>::into(value.#name))
        }
    }).collect::<Vec<_>>();

    let schema = generate_schema_struct(typename, fields);

    let impls = quote! {
        impl From<#typename> for proc_macro_traits::AvroValue {
            fn from(value: #typename) -> Self {
                let fields = vec![#(#record_fields,)*];
                Self::Record(fields)
            }
        }

        impl proc_macro_traits::ToAvro for #typename {
            fn write_schema() -> apache_avro::Schema {
                #schema
            }
        }
    };

    impls
}
