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
            Fields::Unit => derive_toavro_struct_unit(&name),
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
            fn write_schema(prefix: impl AsRef<str>) -> vortex_avro::avro_private::Schema {
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
        vortex_avro::avro_private::Schema::Enum(vortex_avro::avro_private::schema::EnumSchema {
            name: vortex_avro::avro_private::schema::Name {
                name: stringify!(#typename).to_string(),
                namespace: Some(prefix.as_ref().to_string()),
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
            fn write_schema(prefix: impl AsRef<str>) -> vortex_avro::avro_private::Schema {
                #enum_schema
            }
        }
    }
}

fn derive_toavro_struct_unit(typename: &syn::Ident) -> proc_macro2::TokenStream {
    quote! {
        impl From<#typename> for vortex_avro::AvroValue {
            fn from(value: #typename) -> Self {
                Self::Record(vec![])
            }
        }

        impl vortex_avro::ToAvro for #typename {
            fn write_schema(prefix: impl AsRef<str>) -> vortex_avro::avro_private::Schema {
                vortex_avro::avro_private::Schema::Record(vortex_avro::avro_private::schema::RecordSchema {
                    name: vortex_avro::avro_private::schema::Name {
                        name: stringify!(#typename).to_string(),
                        namespace: Some(prefix.as_ref().to_string()),
                    },
                    aliases: None,
                    doc: None,
                    fields: vec![],
                    lookup: Default::default(),
                    attributes: Default::default(),
                })
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
