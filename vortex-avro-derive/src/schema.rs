use quote::quote;
use syn::FieldsNamed;

/// Generate an Apache Avro schema for a struct type.
pub(crate) fn generate_schema_struct(
    typename: &syn::Ident,
    fields: &FieldsNamed,
) -> proc_macro2::TokenStream {
    let lookup = fields
        .named
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let name = f.ident.clone().unwrap();
            let idx = syn::Index::from(idx);
            quote! {
                (stringify!(#name).to_string(), #idx)
            }
        })
        .collect::<Vec<_>>();

    let fields = fields
        .named
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let name = f.ident.clone().unwrap();
            let typ = f.ty.clone();
            quote! {
                vortex_avro::avro_private::schema::RecordField {
                    name: stringify!(#name).to_string(),
                    doc: None,
                    schema: <#typ as vortex_avro::ToAvro>::write_schema(
                        format!("{}.{}", stringify!(#typename), stringify!(#name))),
                    aliases: core::default::Default::default(),
                    default: core::default::Default::default(),
                    order: vortex_avro::avro_private::schema::RecordFieldOrder::Ignore,
                    position: #idx,
                    custom_attributes: core::default::Default::default(),
                }
            }
        })
        .collect::<Vec<_>>();

    // Generate the RecordSchema
    quote! {
        vortex_avro::avro_private::Schema::Record(vortex_avro::avro_private::schema::RecordSchema {
            name: vortex_avro::avro_private::schema::Name {
                name: stringify!(#typename).to_string(),
                namespace: None,
            },
            fields: vec![#(#fields,)*],
            aliases: core::default::Default::default(),
            doc: core::default::Default::default(),
            lookup: std::collections::BTreeMap::from_iter([#(#lookup),*]),
            attributes: core::default::Default::default(),
        })
    }
}
