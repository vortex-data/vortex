use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, Fields, FieldsNamed};

pub(crate) fn derive_from_avro(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let input_span = input.span();
    let name = input.ident;

    let impls = match input.data {
        Data::Struct(s) => {
            match s.fields {
                Fields::Named(fields) => generate_try_from_avrovalue(&name, &fields),
                Fields::Unit => generate_try_from_avrovalue_unit(&name),
                _ => return quote_spanned! {
                    input_span =>
                    compile_error!("FromAvro can only be derived for named structs or unit structs")
                }
                .into(),
            }
        }
        Data::Enum(e) => {
            let is_unit = e.variants.iter().all(|v| v.fields.is_empty());
            if is_unit {
                derive_from_avro_enum_unit(&name, &e)
            } else {
                derive_from_avro_enum_tagged(&name, &e)
            }
        }
        Data::Union(_) => {
            quote_spanned! {
                input_span =>
                compile_error!("cannot derive FromAvro for Rust unions. Please use an enum to generate an Avro union type");
            }
        }
    };

    TokenStream::from(impls)
}

fn derive_from_avro_enum_unit(name: &syn::Ident, e: &syn::DataEnum) -> proc_macro2::TokenStream {
    // Turn each variant into a stringified enum variant for the avro EnumSchema.
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

    // Generate the EnumSchema.
    let enum_schema = quote! {
        vortex_avro::avro_private::Schema::Enum(vortex_avro::avro_private::schema::EnumSchema {
            name: vortex_avro::avro_private::schema::Name {
                name: stringify!(#name).to_string(),
                namespace: None,
            },
            aliases: None,
            doc: None,
            symbols: vec![#(#variants,)*],
            default: None,
            attributes: core::default::Default::default(),
        })
    };

    // Create the match clauses, one for each variant.
    let match_clauses = e
        .variants
        .iter()
        .map(|v| {
            let name = v.ident.clone();
            quote! {
                stringify!(#name) => Ok(Self::#name),
            }
        })
        .collect::<Vec<_>>();

    quote! {
        impl TryFrom<vortex_avro::AvroValue> for #name {
            type Error = vortex_error::VortexError;

            fn try_from(value: vortex_avro::AvroValue) -> Result<Self, Self::Error> {
                let vortex_avro::AvroValue::Enum(variant_idx, variant_name) = value else {
                    vortex_error::vortex_bail!("expected an enum");
                };

                match variant_name.as_str() {
                    #(#match_clauses)*
                    _ => vortex_error::vortex_bail!("unknown variant: {}", variant_name),
                }
            }
        }

        impl FromAvro for #name {
            fn read_schema() -> vortex_avro::avro_private::Schema {
                #enum_schema
            }
        }
    }
}

#[allow(clippy::panic)]
fn derive_from_avro_enum_tagged(
    _name: &syn::Ident,
    _e: &syn::DataEnum,
) -> proc_macro2::TokenStream {
    panic!("derive_from_avro_enum_tagged not implemented");
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
            let #extracted_name: #typ = <#typ as TryFrom<vortex_avro::AvroValue>>::try_from(avro_value)?;
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

    let read_schema = crate::schema::generate_schema_struct(typename, fields);

    quote! {
        impl TryFrom<vortex_avro::AvroValue> for #typename {
            type Error = vortex_error::VortexError;

            fn try_from(value: vortex_avro::AvroValue) -> Result<Self, Self::Error> {
                let vortex_avro::AvroValue::Record(fields) = value else {
                    vortex_error::vortex_bail!("expected a record");
                };

                let mut fields = fields.into_iter();

                // Extract all of the fields from the fields Vec.
                #(#from_avros)*;

                Ok(Self { #(#assignments)* })
            }
        }

        impl vortex_avro::FromAvro for #typename {
            fn read_schema() -> vortex_avro::avro_private::Schema {
                #read_schema
            }
        }
    }
}

fn generate_try_from_avrovalue_unit(typename: &syn::Ident) -> proc_macro2::TokenStream {
    quote! {
        impl TryFrom<vortex_avro::AvroValue> for #typename {
            type Error = vortex_error::VortexError;

            fn try_from(value: vortex_avro::AvroValue) -> Result<Self, Self::Error> {
                let vortex_avro::AvroValue::Record(fields) = value else {
                    vortex_error::vortex_bail!("expected a record");
                };

                Ok(Self)
            }
        }

        impl vortex_avro::FromAvro for #typename {
            fn read_schema() -> vortex_avro::avro_private::Schema {
                vortex_avro::avro_private::Schema::Record(
                    vortex_avro::avro_private::schema::RecordSchema {
                        name: vortex_avro::avro_private::Name {
                            name: stringify!(#typename).to_string(),
                            namespace: None,
                        },
                        aliases: None,
                        doc: None,
                        fields: vec![],
                        lookup: Default::default(),
                        attributes: Default::default(),
                    }
                )
            }
        }
    }
}
