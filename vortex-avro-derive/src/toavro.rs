use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, Fields, FieldsNamed};

pub(crate) fn derive_to_avro(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let input_span = input.span();
    let name = input.ident;

    let to_avro_impl = match input.data {
        Data::Struct(s) => match s.fields {
            Fields::Named(fields) => derive_toavro_struct(&name, &fields),
            _ => {
                return quote_spanned! {
                    input_span =>
                    compile_error!("ToAvro can only be derived for named structs")
                }
                .into()
            }
        },
        Data::Enum(e) => {
            let is_unit = e.variants.iter().all(|v| v.fields.is_empty());
            if is_unit {
                derive_to_avro_enum_unit(&name, &e)
            } else {
                derive_to_avro_enum_tagged(&name, &e)
            }
        }
        Data::Union(_) => {
            unimplemented!()
        }
    };

    to_avro_impl.into()
}

fn derive_toavro_struct(typename: &syn::Ident, fields: &FieldsNamed) -> proc_macro2::TokenStream {
    let record_fields = fields.named.iter().map(|f| {
        let name = f.ident.clone().unwrap();
        let typ = f.ty.clone();
        quote! {
            (stringify!(#name).to_string(), <#typ as Into<vortex_avro::AvroValue>>::into(value.#name))
        }
    }).collect::<Vec<_>>();

    let schema = crate::schema::generate_schema_struct(typename, fields);

    let impls = quote! {
        impl From<#typename> for vortex_avro::AvroValue {
            fn from(value: #typename) -> Self {
                let fields = vec![#(#record_fields,)*];
                Self::Record(fields)
            }
        }

        impl vortex_avro::ToAvro for #typename {
            fn write_schema() -> apache_avro::Schema {
                #schema
            }
        }
    };

    impls
}

fn derive_to_avro_enum_unit(typename: &syn::Ident, e: &syn::DataEnum) -> proc_macro2::TokenStream {
    // impl From<#name> for AvroValue.
    let match_clauses = e
        .variants
        .iter()
        .enumerate()
        .map(|(idx, v)| {
            let name = v.ident.clone();
            quote! {
                <#typename>::#name => (#idx as u32, stringify!(#name).to_string()),
            }
        })
        .collect::<Vec<_>>();

    let variants = e
        .variants
        .iter()
        .map(|v| {
            let name = v.ident.clone();
            quote! {
                stringify!(#name).to_string()
            }
        })
        .collect::<Vec<_>>();

    let enum_schema = quote! {
        apache_avro::Schema::Enum(apache_avro::schema::EnumSchema {
            name: apache_avro::schema::Name {
                name: stringify!(#typename).to_string(),
                namespace: None,
            },
            aliases: None,
            doc: None,
            symbols: vec![#(#variants,)*],
            default: None,
            attributes: core::default::Default::default(),
        })
    };

    quote! {
        impl From<#typename> for vortex_avro::AvroValue {
            fn from(value: #typename) -> Self {
                let (idx, name) = match value {
                    #(#match_clauses)*
                };

                Self::Enum(idx, name)
            }
        }

        impl vortex_avro::ToAvro for #typename {
            fn write_schema() -> apache_avro::Schema {
                #enum_schema
            }
        }
    }
}

#[allow(clippy::panic)]
fn derive_to_avro_enum_tagged(
    _typename: &syn::Ident,
    _e: &syn::DataEnum,
) -> proc_macro2::TokenStream {
    panic!("derive_to_avro_enum_tagged not implemented");
}
