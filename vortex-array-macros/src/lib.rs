// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Proc macros for `vortex-array`.

use proc_macro::TokenStream;
use quote::format_ident;
use quote::quote;
use syn::Field;
use syn::Fields;
use syn::Ident;
use syn::ItemStruct;
use syn::Path;
use syn::Type;
use syn::Visibility;
use syn::parse_macro_input;
use syn::spanned::Spanned;

/// Generate slot index constants, a borrowed view struct, and a typed ext trait
/// from a slot struct definition.
///
/// Fields must be `ArrayRef` (required slot) or `Option<ArrayRef>` (optional slot).
/// Field declaration order determines slot indices.
///
/// # Example
///
/// ```ignore
/// #[array_slots(Patched)]
/// pub struct PatchedSlots {
///     pub inner: ArrayRef,
///     pub lane_offsets: ArrayRef,
///     pub patch_indices: ArrayRef,
///     pub patch_values: ArrayRef,
/// }
/// ```
///
/// # Generated output
///
/// Given the above, the macro generates:
///
/// ```ignore
/// // --- The original struct is preserved as-is ---
/// pub struct PatchedSlots { ... }
///
/// // --- Slot index constants and conversion methods on the struct ---
/// impl PatchedSlots {
///     pub const INNER: usize = 0;
///     pub const LANE_OFFSETS: usize = 1;
///     pub const PATCH_INDICES: usize = 2;
///     pub const PATCH_VALUES: usize = 3;
///     pub const COUNT: usize = 4;
///     pub const NAMES: [&'static str; 4] = ["inner", "lane_offsets", "patch_indices", "patch_values"];
///
///     /// Take ownership of slots from an `ArraySlots`.
///     pub fn from_slots(slots: ArraySlots) -> Self { ... }
///
///     /// Convert back into storage order.
///     pub fn into_slots(self) -> ArraySlots { ... }
/// }
///
/// // --- Borrowed view with &ArrayRef / Option<&ArrayRef> fields ---
/// pub struct PatchedSlotsView<'a> {
///     pub inner: &'a ArrayRef,
///     pub lane_offsets: &'a ArrayRef,
///     pub patch_indices: &'a ArrayRef,
///     pub patch_values: &'a ArrayRef,
/// }
///
/// impl<'a> PatchedSlotsView<'a> {
///     pub fn from_slots(slots: &'a [Option<ArrayRef>]) -> Self { ... }
///     pub fn to_owned(&self) -> PatchedSlots { ... }
/// }
///
/// // --- Ext trait with per-field accessors + slots_view() ---
/// pub trait PatchedArraySlotsExt: TypedArrayRef<Patched> {
///     fn inner(&self) -> &ArrayRef { ... }         // indexes slots directly
///     fn lane_offsets(&self) -> &ArrayRef { ... }
///     fn patch_indices(&self) -> &ArrayRef { ... }
///     fn patch_values(&self) -> &ArrayRef { ... }
///     fn slots_view(&self) -> PatchedSlotsView<'_> { ... }
/// }
///
/// impl<T: TypedArrayRef<Patched>> PatchedArraySlotsExt for T {}
/// ```
///
/// # Required vs optional slots
///
/// - `ArrayRef` — the slot must be present. `from_slots()` panics if `None`.
///   The ext trait accessor returns `&ArrayRef`. The view field is `&'a ArrayRef`.
///
/// - `Option<ArrayRef>` — the slot may be absent. `from_slots()` preserves `None`.
///   The ext trait accessor returns `Option<&ArrayRef>`. The view field is
///   `Option<&'a ArrayRef>`.
///
/// The underlying storage is always `ArraySlots` — the field type only
/// controls whether the macro inserts a `.vortex_expect()` unwrap or not.
#[proc_macro_attribute]
pub fn array_slots(attr: TokenStream, item: TokenStream) -> TokenStream {
    let encoding = parse_macro_input!(attr as Path);
    let item_struct = parse_macro_input!(item as ItemStruct);

    match expand_array_slots(encoding, item_struct) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_array_slots(
    encoding: Path,
    item_struct: ItemStruct,
) -> syn::Result<proc_macro2::TokenStream> {
    if !item_struct.generics.params.is_empty() || item_struct.generics.where_clause.is_some() {
        return Err(syn::Error::new(
            item_struct.generics.span(),
            "#[array_slots] does not support generic slot structs",
        ));
    }

    let fields = match &item_struct.fields {
        Fields::Named(fields) => &fields.named,
        _ => {
            return Err(syn::Error::new(
                item_struct.span(),
                "#[array_slots] requires a struct with named fields",
            ));
        }
    };

    let encoding_ident = encoding
        .segments
        .last()
        .map(|segment| &segment.ident)
        .ok_or_else(|| syn::Error::new(encoding.span(), "missing encoding type"))?;

    let struct_ident = &item_struct.ident;
    let struct_vis = &item_struct.vis;
    let view_ident = format_ident!("{}View", ident_name(struct_ident));
    let ext_ident = format_ident!("{}ArraySlotsExt", ident_name(encoding_ident));

    let field_specs = fields
        .iter()
        .enumerate()
        .map(|(index, field)| SlotField::new(field, index, struct_ident))
        .collect::<syn::Result<Vec<_>>>()?;

    let idx_consts = field_specs.iter().map(SlotField::idx_const);
    let view_fields = field_specs.iter().map(SlotField::view_field);
    let view_from_slots = field_specs.iter().map(SlotField::view_from_slots);
    let view_to_owned = field_specs.iter().map(SlotField::view_to_owned);
    let owned_from_slots = field_specs.iter().map(SlotField::owned_from_slots);
    let into_slots = field_specs.iter().map(SlotField::storage_slot);
    let ext_methods = field_specs.iter().map(SlotField::ext_method);
    let slot_names = field_specs.iter().map(|field| field.slot_name.as_str());
    let slot_count = field_specs.len();

    Ok(quote! {
        #item_struct

        impl #struct_ident {
            #(#idx_consts)*

            #[doc = "Total number of slots."]
            pub const COUNT: usize = #slot_count;

            #[doc = "Slot names in storage order."]
            pub const NAMES: [&'static str; #slot_count] = [#(#slot_names),*];

            #[doc = "Convert owned slot storage into an owned slot struct."]
            pub fn from_slots(mut slots: ::vortex_array::ArraySlots) -> Self {
                Self {
                    #(#owned_from_slots,)*
                }
            }

            #[doc = "Convert this slot struct into storage order."]
            pub fn into_slots(self) -> ::vortex_array::ArraySlots {
                ::vortex_array::smallvec::smallvec![#(#into_slots),*]
            }
        }

        #[derive(Clone, Copy, Debug)]
        #[doc = concat!("Borrowed view of `", stringify!(#struct_ident), "`.")]
        #struct_vis struct #view_ident<'a> {
            #(#view_fields,)*
        }

        impl<'a> #view_ident<'a> {
            #[doc = "Borrow a slot slice as a typed view."]
            pub fn from_slots(slots: &'a [Option<::vortex_array::ArrayRef>]) -> Self {
                Self {
                    #(#view_from_slots,)*
                }
            }

            #[doc = "Clone all referenced slots into an owned slot struct."]
            pub fn to_owned(&self) -> #struct_ident {
                #struct_ident {
                    #(#view_to_owned,)*
                }
            }
        }

        #[doc = concat!("Typed array accessors for `", stringify!(#encoding_ident), "`.")]
        #struct_vis trait #ext_ident: ::vortex_array::TypedArrayRef<#encoding> {
            #(#ext_methods)*

            #[doc = "Returns a borrowed view of all slots."]
            fn slots_view(&self) -> #view_ident<'_> {
                #view_ident::from_slots(self.as_ref().slots())
            }
        }

        impl<T: ::vortex_array::TypedArrayRef<#encoding>> #ext_ident for T {}
    })
}

struct SlotField {
    field_ident: Ident,
    field_vis: Visibility,
    const_ident: Ident,
    slot_name: String,
    slot_type: SlotFieldType,
    index: usize,
    expect_message: syn::LitStr,
    struct_ident: Ident,
}

impl SlotField {
    fn new(field: &Field, index: usize, struct_ident: &Ident) -> syn::Result<Self> {
        let field_ident = field
            .ident
            .clone()
            .ok_or_else(|| syn::Error::new(field.span(), "slot fields must be named"))?;
        let field_name = ident_name(&field_ident);
        let const_ident = format_ident!("{}", to_screaming_snake_case(&field_name));
        let slot_type = SlotFieldType::from_syn_type(&field.ty)?;
        let expect_message = syn::LitStr::new(
            &format!("{} {} slot", ident_name(struct_ident), field_name),
            field.span(),
        );

        Ok(Self {
            field_ident,
            field_vis: field.vis.clone(),
            const_ident,
            slot_name: field_name,
            slot_type,
            index,
            expect_message,
            struct_ident: struct_ident.clone(),
        })
    }

    fn idx_const(&self) -> proc_macro2::TokenStream {
        let const_ident = &self.const_ident;
        let index = self.index;
        let slot_name = &self.slot_name;

        quote! {
            #[doc = concat!("Slot index for `", #slot_name, "`.")]
            pub const #const_ident: usize = #index;
        }
    }

    fn view_field(&self) -> proc_macro2::TokenStream {
        let field_ident = &self.field_ident;
        let field_vis = &self.field_vis;
        let ty = self.slot_type.view_field_ty();

        quote! {
            #field_vis #field_ident: #ty
        }
    }

    fn view_from_slots(&self) -> proc_macro2::TokenStream {
        let field_ident = &self.field_ident;
        let struct_ident = &self.struct_ident;
        let const_ident = &self.const_ident;
        let expect_message = &self.expect_message;

        match self.slot_type {
            SlotFieldType::Required => quote! {
                #field_ident: ::vortex_error::VortexExpect::vortex_expect(
                    slots[#struct_ident::#const_ident].as_ref(),
                    #expect_message,
                )
            },
            SlotFieldType::Optional => quote! {
                #field_ident: slots[#struct_ident::#const_ident].as_ref()
            },
        }
    }

    fn view_to_owned(&self) -> proc_macro2::TokenStream {
        let field_ident = &self.field_ident;

        match self.slot_type {
            SlotFieldType::Required => quote! {
                #field_ident: ::std::clone::Clone::clone(self.#field_ident)
            },
            SlotFieldType::Optional => quote! {
                #field_ident: self.#field_ident.cloned()
            },
        }
    }

    fn owned_from_slots(&self) -> proc_macro2::TokenStream {
        let field_ident = &self.field_ident;
        let struct_ident = &self.struct_ident;
        let const_ident = &self.const_ident;
        let expect_message = &self.expect_message;

        match self.slot_type {
            SlotFieldType::Required => quote! {
                #field_ident: ::vortex_error::VortexExpect::vortex_expect(
                    slots[#struct_ident::#const_ident].take(),
                    #expect_message,
                )
            },
            SlotFieldType::Optional => quote! {
                #field_ident: slots[#struct_ident::#const_ident].take()
            },
        }
    }

    fn storage_slot(&self) -> proc_macro2::TokenStream {
        let field_ident = &self.field_ident;

        match self.slot_type {
            SlotFieldType::Required => quote! {
                Some(self.#field_ident)
            },
            SlotFieldType::Optional => quote! {
                self.#field_ident
            },
        }
    }

    fn ext_method(&self) -> proc_macro2::TokenStream {
        let field_ident = &self.field_ident;
        let struct_ident = &self.struct_ident;
        let const_ident = &self.const_ident;
        let expect_message = &self.expect_message;

        match self.slot_type {
            SlotFieldType::Required => quote! {
                #[inline]
                fn #field_ident(&self) -> &::vortex_array::ArrayRef {
                    ::vortex_error::VortexExpect::vortex_expect(
                        self.as_ref().slots()[#struct_ident::#const_ident].as_ref(),
                        #expect_message,
                    )
                }
            },
            SlotFieldType::Optional => quote! {
                #[inline]
                fn #field_ident(&self) -> Option<&::vortex_array::ArrayRef> {
                    self.as_ref().slots()[#struct_ident::#const_ident].as_ref()
                }
            },
        }
    }
}

#[derive(Clone, Copy)]
enum SlotFieldType {
    Required,
    Optional,
}

impl SlotFieldType {
    fn from_syn_type(ty: &Type) -> syn::Result<Self> {
        if is_array_ref_type(ty) {
            return Ok(Self::Required);
        }

        if let Some(inner_ty) = option_inner_type(ty)
            && is_array_ref_type(inner_ty)
        {
            return Ok(Self::Optional);
        }

        Err(syn::Error::new(
            ty.span(),
            "#[array_slots] fields must be ArrayRef or Option<ArrayRef>",
        ))
    }

    fn view_field_ty(self) -> proc_macro2::TokenStream {
        match self {
            Self::Required => quote! { &'a ::vortex_array::ArrayRef },
            Self::Optional => quote! { Option<&'a ::vortex_array::ArrayRef> },
        }
    }
}

fn is_array_ref_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Path(type_path)
            if type_path.qself.is_none()
                && type_path
                    .path
                    .segments
                    .last()
                    .is_some_and(|segment| segment.ident == "ArrayRef")
    )
}

fn option_inner_type(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }

    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };

    match args.args.first()? {
        syn::GenericArgument::Type(inner_ty) => Some(inner_ty),
        _ => None,
    }
}

fn ident_name(ident: &Ident) -> String {
    ident.to_string().trim_start_matches("r#").to_owned()
}

fn to_screaming_snake_case(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    let mut prev_is_lower_or_digit = false;

    for ch in name.chars() {
        if ch.is_ascii_uppercase() && prev_is_lower_or_digit {
            result.push('_');
        }
        result.push(ch.to_ascii_uppercase());
        prev_is_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
    }

    result
}
