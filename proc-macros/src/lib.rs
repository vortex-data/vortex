use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, Fields, FieldsNamed, FieldsUnnamed};

#[proc_macro_derive(FromTuple)]
pub fn derive_macro_avro(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    let input_span = input.span();
    let name = input.ident;

    let input_var = syn::Ident::new("input", name.span());

    let (assignment_expr, tuple_types) = match input.data {
        Data::Struct(s) => match s.fields {
            Fields::Named(fields) => (
                generate_from_named(&name, &input_var, &fields),
                fields
                    .named
                    .iter()
                    .map(|f| f.ty.clone())
                    .collect::<Vec<_>>(),
            ),
            Fields::Unnamed(fields) => (
                generate_from_unnamed(&name, &input_var, &fields),
                fields
                    .unnamed
                    .iter()
                    .map(|f| f.ty.clone())
                    .collect::<Vec<_>>(),
            ),
            Fields::Unit => {
                return quote_spanned! {
                    input_span =>
                    compile_error!("FromTuple can only be derived for field or tuple structs")
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

    let output = quote! {
        impl From<(#( #tuple_types, )*)> for #name {
            #[allow(unused)]
            fn from(#input_var: ( #( #tuple_types, )* )) -> Self {
                #assignment_expr
            }
        }
    };

    TokenStream::from(output)
}

/// Generate the constructor expression for a named struct.
fn generate_from_named(
    typename: &syn::Ident,
    input: &syn::Ident,
    fields: &FieldsNamed,
) -> proc_macro2::TokenStream {
    // Get the assignment internals here.
    let assignments = fields
        .named
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            let name = &f.ident;
            let idx = syn::Index::from(idx);

            quote! {
                #name: #input.#idx,
            }
        })
        .collect::<Vec<_>>();

    quote! {
        #typename { #(#assignments)* }
    }
}

fn generate_from_unnamed(
    typename: &syn::Ident,
    input_ident: &syn::Ident,
    fields: &FieldsUnnamed,
) -> proc_macro2::TokenStream {
    let assignments = fields
        .unnamed
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            quote! {
                #input_ident.#idx,
            }
        })
        .collect::<Vec<_>>();

    quote! {
        #typename(#(#assignments)*)
    }
}
