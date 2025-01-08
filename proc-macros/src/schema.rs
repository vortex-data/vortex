use quote::quote;
use syn::FieldsNamed;

/// Generate an Apache Avro schema for a struct type.
pub(crate) fn generate_schema_struct(
    typename: &syn::Ident,
    fields: &FieldsNamed,
) -> proc_macro2::TokenStream {
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
