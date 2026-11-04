// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expansion of `#[plan_slots]`, the plan counterpart of `#[array_slots]`.

use proc_macro2::TokenStream;
use quote::format_ident;
use quote::quote;
use syn::Fields;
use syn::ItemStruct;
use syn::Path;
use syn::spanned::Spanned;

use crate::SlotField;
use crate::SlotFieldType;
use crate::SlotKind;
use crate::ident_name;
use crate::partition_by_slot_index;
use crate::strip_slot_attrs;

pub(crate) fn expand_plan_slots(vtable: Path, item_struct: ItemStruct) -> syn::Result<TokenStream> {
    if !item_struct.generics.params.is_empty() || item_struct.generics.where_clause.is_some() {
        return Err(syn::Error::new(
            item_struct.generics.span(),
            "#[plan_slots] does not support generic slot structs",
        ));
    }

    let fields = match &item_struct.fields {
        Fields::Named(fields) => &fields.named,
        _ => {
            return Err(syn::Error::new(
                item_struct.span(),
                "#[plan_slots] requires a struct with named fields",
            ));
        }
    };

    let vtable_ident = vtable
        .segments
        .last()
        .map(|segment| &segment.ident)
        .ok_or_else(|| syn::Error::new(vtable.span(), "missing plan vtable type"))?;
    let vtable_name = ident_name(vtable_ident);

    let struct_ident = item_struct.ident.clone();
    let struct_vis = item_struct.vis.clone();
    let view_ident = format_ident!("{}View", ident_name(&struct_ident));
    let ext_ident = format_ident!("{}PlanSlotsExt", vtable_name);

    let field_specs = fields
        .iter()
        .map(|field| SlotField::new(field, &struct_ident, SlotKind::Plan))
        .collect::<syn::Result<Vec<_>>>()?;
    let (fixed_specs, tail_spec) = partition_by_slot_index(&field_specs, SlotKind::Plan)?;

    let codegen = PlanGen {
        vtable_name: &vtable_name,
        struct_ident: &struct_ident,
    };

    let idx_consts = fixed_specs.iter().copied().map(SlotField::idx_const);
    let counts = codegen.counts(&fixed_specs, tail_spec);
    let slot_name = codegen.slot_name(tail_spec);
    let into_slots = codegen.gen_into_slots(&fixed_specs, tail_spec);
    let check_len = codegen.check_len(tail_spec);

    let view_fields = field_specs.iter().map(|spec| codegen.view_field(spec));
    let view_from_slots = field_specs.iter().map(|spec| codegen.view_from_slots(spec));
    let view_to_owned = field_specs.iter().map(|spec| codegen.view_to_owned(spec));
    let ext_decls = field_specs.iter().map(|spec| codegen.ext_decl(spec));
    let ext_impls = field_specs.iter().map(|spec| codegen.ext_impl(spec));

    // A view needs its lifetime even when there is no slot to borrow.
    let (marker_field, marker_init) = if field_specs.is_empty() {
        (
            quote! { _lifetime: ::std::marker::PhantomData<&'a ()>, },
            quote! { _lifetime: ::std::marker::PhantomData, },
        )
    } else {
        (quote! {}, quote! {})
    };

    let item_struct = strip_slot_attrs(item_struct);

    Ok(quote! {
        #item_struct

        impl #struct_ident {
            #(#idx_consts)*

            #counts

            #[doc = "Returns the display name of the slot at `idx`."]
            #slot_name

            #[doc = "Converts this slot struct into built plan slots in storage order."]
            #into_slots
        }

        #[derive(Clone, Copy, Debug)]
        #[doc = concat!("Borrowed view of `", stringify!(#struct_ident), "`.")]
        #struct_vis struct #view_ident<'a> {
            #(#view_fields,)*
            #marker_field
        }

        impl<'a> #view_ident<'a> {
            #[doc = "Borrows plan slots as a typed view."]
            #[doc = ""]
            #[doc = "Errors when the number of slots is wrong or a required slot is absent. Every"]
            #[doc = "fixed slot is built; variadic tail slots are built on access."]
            pub fn try_from_slots(
                slots: &'a ::vortex_layout::plan::physical::PlanSlots,
            ) -> ::vortex_error::VortexResult<Self> {
                #check_len
                Ok(Self {
                    #(#view_from_slots,)*
                    #marker_init
                })
            }

            #[doc = "Clones all referenced plans into an owned slot struct, building any unbuilt"]
            #[doc = "variadic tail slot."]
            pub fn to_owned(&self) -> ::vortex_error::VortexResult<#struct_ident> {
                Ok(#struct_ident {
                    #(#view_to_owned,)*
                })
            }
        }

        #[doc = concat!("Typed slot accessors for `", stringify!(#vtable_ident), "` plans.")]
        #struct_vis trait #ext_ident {
            #(#ext_decls)*

            #[doc = "Returns a borrowed view of all slots, building every fixed slot."]
            fn slots_view(&self) -> ::vortex_error::VortexResult<#view_ident<'_>>;
        }

        impl #ext_ident for ::vortex_layout::plan::physical::Plan<#vtable> {
            #(#ext_impls)*

            fn slots_view(&self) -> ::vortex_error::VortexResult<#view_ident<'_>> {
                #view_ident::try_from_slots(self.slots())
            }
        }
    })
}

/// Shared context for generating one `#[plan_slots]` expansion.
struct PlanGen<'a> {
    vtable_name: &'a str,
    struct_ident: &'a syn::Ident,
}

impl PlanGen<'_> {
    fn absent_message(&self, spec: &SlotField) -> String {
        format!("{} {} slot is absent", self.vtable_name, spec.slot_name)
    }

    fn counts(&self, fixed_specs: &[&SlotField], tail_spec: Option<&SlotField>) -> TokenStream {
        let names = fixed_specs.iter().map(|field| field.slot_name.as_str());
        let fixed_count = fixed_specs.len();

        match tail_spec {
            None => quote! {
                #[doc = "Total number of slots."]
                pub const COUNT: usize = #fixed_count;

                #[doc = "Slot display names in storage order."]
                pub const NAMES: [&'static str; #fixed_count] = [#(#names),*];
            },
            Some(tail) => {
                let offset_const = &tail.const_ident;
                let tail_name = &tail.slot_name;
                quote! {
                    #[doc = concat!("Offset at which the `", #tail_name, "` slots begin.")]
                    pub const #offset_const: usize = #fixed_count;

                    #[doc = "Number of fixed (non-variadic) slots."]
                    pub const FIXED_COUNT: usize = #fixed_count;

                    #[doc = "Display names of the fixed slots in storage order."]
                    pub const FIXED_NAMES: [&'static str; #fixed_count] = [#(#names),*];
                }
            }
        }
    }

    fn slot_name(&self, tail_spec: Option<&SlotField>) -> TokenStream {
        match tail_spec {
            None => quote! {
                pub fn slot_name(idx: usize) -> ::std::borrow::Cow<'static, str> {
                    match Self::NAMES.get(idx) {
                        Some(name) => ::std::borrow::Cow::Borrowed(*name),
                        None => ::std::borrow::Cow::Owned(format!("slot[{idx}]")),
                    }
                }
            },
            Some(tail) => {
                let offset_const = &tail.const_ident;
                let tail_name = &tail.slot_name;
                quote! {
                    pub fn slot_name(idx: usize) -> ::std::borrow::Cow<'static, str> {
                        if idx < Self::FIXED_COUNT {
                            ::std::borrow::Cow::Borrowed(Self::FIXED_NAMES[idx])
                        } else {
                            ::std::borrow::Cow::Owned(format!(
                                concat!(#tail_name, "[{}]"),
                                idx - Self::#offset_const
                            ))
                        }
                    }
                }
            }
        }
    }

    fn gen_into_slots(
        &self,
        fixed_specs: &[&SlotField],
        tail_spec: Option<&SlotField>,
    ) -> TokenStream {
        let fixed_count = fixed_specs.len();
        let fixed = fixed_specs.iter().map(|spec| {
            let field_ident = &spec.field_ident;
            match spec.slot_type {
                SlotFieldType::Required => quote! {
                    Some(::vortex_layout::plan::physical::LazyPlanRef::ready(self.#field_ident))
                },
                SlotFieldType::Optional => quote! {
                    self.#field_ident.map(::vortex_layout::plan::physical::LazyPlanRef::ready)
                },
                SlotFieldType::VariadicTail => {
                    unreachable!("variadic tail is appended after fixed fields")
                }
            }
        });
        let tail = tail_spec.map(|tail| {
            let tail_ident = &tail.field_ident;
            quote! {
                .chain(self.#tail_ident.into_iter().map(|plan| {
                    Some(::vortex_layout::plan::physical::LazyPlanRef::ready(plan))
                }))
            }
        });

        quote! {
            pub fn into_slots(self) -> ::vortex_layout::plan::physical::PlanSlots {
                let fixed: [Option<::vortex_layout::plan::physical::LazyPlanRef>; #fixed_count] =
                    [#(#fixed),*];
                fixed.into_iter()#tail.collect()
            }
        }
    }

    fn check_len(&self, tail_spec: Option<&SlotField>) -> TokenStream {
        let struct_ident = self.struct_ident;
        let vtable_name = self.vtable_name;
        match tail_spec {
            None => quote! {
                if slots.len() != #struct_ident::COUNT {
                    ::vortex_error::vortex_bail!(
                        "{} expects {} slots but got {}",
                        #vtable_name,
                        #struct_ident::COUNT,
                        slots.len()
                    );
                }
            },
            Some(_) => quote! {
                if slots.len() < #struct_ident::FIXED_COUNT {
                    ::vortex_error::vortex_bail!(
                        "{} expects at least {} slots but got {}",
                        #vtable_name,
                        #struct_ident::FIXED_COUNT,
                        slots.len()
                    );
                }
            },
        }
    }

    fn view_field(&self, spec: &SlotField) -> TokenStream {
        let field_ident = &spec.field_ident;
        let field_vis = &spec.field_vis;
        let ty = match spec.slot_type {
            SlotFieldType::Required => quote! { &'a ::vortex_layout::plan::physical::PlanRef },
            SlotFieldType::Optional => {
                quote! { Option<&'a ::vortex_layout::plan::physical::PlanRef> }
            }
            SlotFieldType::VariadicTail => {
                quote! { ::vortex_layout::plan::physical::PlanSlotSlice<'a> }
            }
        };
        quote! { #field_vis #field_ident: #ty }
    }

    fn view_from_slots(&self, spec: &SlotField) -> TokenStream {
        let field_ident = &spec.field_ident;
        let struct_ident = self.struct_ident;
        let const_ident = &spec.const_ident;
        let absent = self.absent_message(spec);
        let slot_name = &spec.slot_name;

        match spec.slot_type {
            SlotFieldType::Required => quote! {
                #field_ident: slots
                    .get(#struct_ident::#const_ident)?
                    .ok_or_else(|| ::vortex_error::vortex_err!(#absent))?
            },
            SlotFieldType::Optional => quote! {
                #field_ident: slots.get(#struct_ident::#const_ident)?
            },
            SlotFieldType::VariadicTail => quote! {
                #field_ident: ::vortex_layout::plan::physical::PlanSlotSlice::try_new(
                    &slots[#struct_ident::#const_ident..],
                    #slot_name,
                )?
            },
        }
    }

    fn view_to_owned(&self, spec: &SlotField) -> TokenStream {
        let field_ident = &spec.field_ident;
        match spec.slot_type {
            SlotFieldType::Required => quote! {
                #field_ident: ::std::clone::Clone::clone(self.#field_ident)
            },
            SlotFieldType::Optional => quote! {
                #field_ident: self.#field_ident.cloned()
            },
            SlotFieldType::VariadicTail => quote! {
                #field_ident: self.#field_ident.to_vec()?
            },
        }
    }

    fn ext_decl(&self, spec: &SlotField) -> TokenStream {
        let field_ident = &spec.field_ident;
        let slot_name = &spec.slot_name;
        match spec.slot_type {
            SlotFieldType::Required => quote! {
                #[doc = concat!("Returns the plan in the `", #slot_name, "` slot, building it on")]
                #[doc = "first access."]
                fn #field_ident(
                    &self,
                ) -> ::vortex_error::VortexResult<&::vortex_layout::plan::physical::PlanRef>;
            },
            SlotFieldType::Optional => quote! {
                #[doc = concat!("Returns the plan in the `", #slot_name, "` slot, building it on")]
                #[doc = "first access, or `None` when the slot is absent."]
                fn #field_ident(
                    &self,
                ) -> ::vortex_error::VortexResult<
                    Option<&::vortex_layout::plan::physical::PlanRef>,
                >;
            },
            SlotFieldType::VariadicTail => quote! {
                #[doc = concat!("Returns the `", #slot_name, "` slots, each built on access.")]
                fn #field_ident(&self) -> ::vortex_layout::plan::physical::PlanSlotSlice<'_>;
            },
        }
    }

    fn ext_impl(&self, spec: &SlotField) -> TokenStream {
        let field_ident = &spec.field_ident;
        let struct_ident = self.struct_ident;
        let const_ident = &spec.const_ident;
        let absent = self.absent_message(spec);
        let slot_name = &spec.slot_name;

        match spec.slot_type {
            SlotFieldType::Required => quote! {
                #[inline]
                fn #field_ident(
                    &self,
                ) -> ::vortex_error::VortexResult<&::vortex_layout::plan::physical::PlanRef> {
                    self.slots()
                        .get(#struct_ident::#const_ident)?
                        .ok_or_else(|| ::vortex_error::vortex_err!(#absent))
                }
            },
            SlotFieldType::Optional => quote! {
                #[inline]
                fn #field_ident(
                    &self,
                ) -> ::vortex_error::VortexResult<
                    Option<&::vortex_layout::plan::physical::PlanRef>,
                > {
                    self.slots().get(#struct_ident::#const_ident)
                }
            },
            SlotFieldType::VariadicTail => quote! {
                #[inline]
                fn #field_ident(&self) -> ::vortex_layout::plan::physical::PlanSlotSlice<'_> {
                    ::vortex_layout::plan::physical::PlanSlotSlice::new(
                        self.slots()[..].get(#struct_ident::#const_ident..).unwrap_or(&[]),
                        #slot_name,
                    )
                }
            },
        }
    }
}
