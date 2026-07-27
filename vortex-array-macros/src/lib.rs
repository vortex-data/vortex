// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Proc macros for `vortex-array`.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::format_ident;
use quote::quote;
use syn::Field;
use syn::Fields;
use syn::Ident;
use syn::ItemStruct;
use syn::LitInt;
use syn::Path;
use syn::Token;
use syn::Type;
use syn::Visibility;
use syn::parse::Parse;
use syn::parse::ParseStream;
use syn::parse_macro_input;
use syn::spanned::Spanned;

/// Name of the per-field attribute that pins a field to a slot index.
const SLOT_ATTR: &str = "slot";

/// Generate slot index constants, a borrowed view struct, and a typed ext trait
/// from a slot struct definition.
///
/// Fields must be `ArrayRef` (required slot), `Option<ArrayRef>` (optional slot), or
/// `Vec<ArrayRef>` (variadic tail of required slots).
///
/// Every field must carry a `#[slot(..)]` attribute naming the exact slot index it maps
/// to. The attribute — not the declaration order — defines the storage layout, so fields
/// may be reordered, grouped, or documented in any order without changing the slot
/// indices an array is built from or read back with.
///
/// # Example
///
/// ```ignore
/// #[array_slots(Patched)]
/// pub struct PatchedSlots {
///     #[slot(0)]
///     pub inner: ArrayRef,
///     #[slot(1)]
///     pub lane_offsets: ArrayRef,
///     #[slot(2)]
///     pub patch_indices: ArrayRef,
///     #[slot(3)]
///     pub patch_values: ArrayRef,
/// }
/// ```
///
/// # Generated output
///
/// Given the above, the macro generates:
///
/// ```ignore
/// // --- The original struct, minus the consumed `#[slot(..)]` attributes ---
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
/// # Slot index annotations
///
/// - Fixed fields use `#[slot(N)]`, where `N` is the exact index of the slot.
/// - A variadic tail uses `#[slot(N..)]`, where `N` is the index its first slot occupies.
///
/// The annotations are validated at compile time: the fixed indices must cover
/// `0..FIXED_COUNT` exactly — no duplicates and no gaps — and a variadic tail must start
/// immediately after the last fixed slot. A field without a `#[slot(..)]` attribute is a
/// compile error, so the layout can never silently fall back to declaration order.
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
///
/// # Variadic tail slots
///
/// One field may be `Vec<ArrayRef>`, declaring that every slot from its index onward
/// belongs to a homogeneous, variable-length run of required slots. This supports
/// encodings like `Chunked` (`[chunk_offsets, chunks...]`), `Struct`
/// (`[validity?, fields...]`), and `Union` (`[type_ids, children...]`).
///
/// ```ignore
/// #[array_slots(Chunked)]
/// pub struct ChunkedSlots {
///     #[slot(0)]
///     pub chunk_offsets: ArrayRef,
///     #[slot(1..)]
///     pub chunks: Vec<ArrayRef>,
/// }
/// ```
///
/// For a struct with a variadic tail, the macro generates a different set of
/// constants — slot count is no longer a compile-time constant:
///
/// ```ignore
/// impl ChunkedSlots {
///     pub const CHUNK_OFFSETS: usize = 0;
///     /// Offset at which the `chunks` slots begin.
///     pub const CHUNKS_OFFSET: usize = 1;
///     /// Number of fixed (non-variadic) slots.
///     pub const FIXED_COUNT: usize = 1;
///     /// Names of the fixed slots in storage order.
///     pub const FIXED_NAMES: [&'static str; 1] = ["chunk_offsets"];
///
///     /// Name of the slot at `idx`, e.g. "chunk_offsets" or "chunks[3]".
///     pub fn slot_name(idx: usize) -> String { ... }
///
///     pub fn from_slots(slots: ArraySlots) -> Self { ... }
///     pub fn into_slots(self) -> ArraySlots { ... }
/// }
/// ```
///
/// The view field and ext trait accessor for the tail are a `vortex_array::SlotSlice`,
/// a borrowed run of required slots supporting `len()`, `get()`, `iter()`, and
/// indexing:
///
/// ```ignore
/// pub struct ChunkedSlotsView<'a> {
///     pub chunk_offsets: &'a ArrayRef,
///     pub chunks: SlotSlice<'a>,
/// }
///
/// pub trait ChunkedArraySlotsExt: TypedArrayRef<Chunked> {
///     fn chunk_offsets(&self) -> &ArrayRef { ... }
///     fn chunks(&self) -> SlotSlice<'_> { ... }
///     fn slots_view(&self) -> ChunkedSlotsView<'_> { ... }
/// }
/// ```
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

    let struct_ident = item_struct.ident.clone();
    let struct_vis = item_struct.vis.clone();
    let view_ident = format_ident!("{}View", ident_name(&struct_ident));
    let ext_ident = format_ident!("{}ArraySlotsExt", ident_name(encoding_ident));

    let field_specs = fields
        .iter()
        .map(|field| SlotField::new(field, &struct_ident))
        .collect::<syn::Result<Vec<_>>>()?;

    // The `#[slot(..)]` annotations, not the declaration order, define storage order.
    let (fixed_specs, tail_spec) = partition_by_slot_index(&field_specs)?;

    let idx_consts = fixed_specs.iter().copied().map(SlotField::idx_const);
    let view_fields = field_specs.iter().map(SlotField::view_field);
    let view_from_slots = field_specs.iter().map(SlotField::view_from_slots);
    let view_to_owned = field_specs.iter().map(SlotField::view_to_owned);
    let ext_methods = field_specs.iter().map(SlotField::ext_method);

    let counts = gen_counts(&fixed_specs, tail_spec);
    let from_slots = gen_from_slots(&fixed_specs, tail_spec);
    let into_slots = gen_into_slots(&fixed_specs, tail_spec);

    // `#[slot(..)]` is inert helper syntax consumed here; strip it from the emitted struct.
    let item_struct = strip_slot_attrs(item_struct);

    Ok(quote! {
        #item_struct

        impl #struct_ident {
            #(#idx_consts)*

            #counts

            #[doc = "Convert owned slot storage into an owned slot struct."]
            #from_slots

            #[doc = "Convert this slot struct into storage order."]
            #into_slots
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

/// Split the fields into the fixed slots (sorted by slot index) and the optional variadic
/// tail, validating that the annotated indices describe a complete, gap-free layout.
fn partition_by_slot_index(
    field_specs: &[SlotField],
) -> syn::Result<(Vec<&SlotField>, Option<&SlotField>)> {
    let mut fixed_specs = Vec::with_capacity(field_specs.len());
    let mut tail_spec: Option<&SlotField> = None;

    for spec in field_specs {
        if matches!(spec.slot_type, SlotFieldType::VariadicTail) {
            if let Some(previous) = tail_spec {
                return Err(syn::Error::new(
                    spec.index_span,
                    format!(
                        "#[array_slots] allows at most one variadic tail, but `{}` is already \
                         declared as one",
                        previous.slot_name
                    ),
                ));
            }
            tail_spec = Some(spec);
        } else {
            fixed_specs.push(spec);
        }
    }

    fixed_specs.sort_by_key(|spec| spec.index);

    for (expected, spec) in fixed_specs.iter().enumerate() {
        if spec.index == expected {
            continue;
        }
        // `fixed_specs` is sorted, so a smaller index than expected means the previous
        // field claimed the same slot, and a larger one means nothing claimed `expected`.
        return Err(if spec.index < expected {
            syn::Error::new(
                spec.index_span,
                format!(
                    "#[array_slots] slot index {} is claimed by both `{}` and `{}`",
                    spec.index,
                    fixed_specs[expected - 1].slot_name,
                    spec.slot_name
                ),
            )
        } else {
            syn::Error::new(
                spec.index_span,
                format!(
                    "#[array_slots] no field claims slot index {expected}; fixed slot indices \
                     must cover 0..{} without gaps",
                    fixed_specs.len()
                ),
            )
        });
    }

    if let Some(tail) = tail_spec
        && tail.index != fixed_specs.len()
    {
        return Err(syn::Error::new(
            tail.index_span,
            format!(
                "#[array_slots] variadic tail `{}` must start at slot index {}, immediately after \
                 the {} fixed slot(s), but is annotated `#[slot({}..)]`",
                tail.slot_name,
                fixed_specs.len(),
                fixed_specs.len(),
                tail.index
            ),
        ));
    }

    Ok((fixed_specs, tail_spec))
}

/// Remove the inert `#[slot(..)]` helper attributes so the struct can be re-emitted.
fn strip_slot_attrs(mut item_struct: ItemStruct) -> ItemStruct {
    if let Fields::Named(fields) = &mut item_struct.fields {
        for field in &mut fields.named {
            field.attrs.retain(|attr| !attr.path().is_ident(SLOT_ATTR));
        }
    }
    item_struct
}

fn gen_counts(
    fixed_specs: &[&SlotField],
    tail_spec: Option<&SlotField>,
) -> proc_macro2::TokenStream {
    let names = fixed_specs.iter().map(|field| field.slot_name.as_str());
    let fixed_count = fixed_specs.len();

    match tail_spec {
        None => quote! {
            #[doc = "Total number of slots."]
            pub const COUNT: usize = #fixed_count;

            #[doc = "Slot names in storage order."]
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

                #[doc = "Names of the fixed slots in storage order."]
                pub const FIXED_NAMES: [&'static str; #fixed_count] = [#(#names),*];

                #[doc = "Name of the slot at the given index."]
                pub fn slot_name(idx: usize) -> String {
                    if idx < Self::FIXED_COUNT {
                        Self::FIXED_NAMES[idx].to_string()
                    } else {
                        format!(concat!(#tail_name, "[{}]"), idx - Self::#offset_const)
                    }
                }
            }
        }
    }
}

fn gen_from_slots(
    fixed_specs: &[&SlotField],
    tail_spec: Option<&SlotField>,
) -> proc_macro2::TokenStream {
    let owned_from_slots = fixed_specs.iter().copied().map(SlotField::owned_from_slots);

    match tail_spec {
        None => quote! {
            pub fn from_slots(mut slots: ::vortex_array::ArraySlots) -> Self {
                Self {
                    #(#owned_from_slots,)*
                }
            }
        },
        Some(tail) => {
            let tail_ident = &tail.field_ident;
            let offset_const = &tail.const_ident;
            let expect_message = &tail.expect_message;
            quote! {
                pub fn from_slots(mut slots: ::vortex_array::ArraySlots) -> Self {
                    let __variadic_tail: ::std::vec::Vec<::vortex_array::ArrayRef> = slots
                        .drain(Self::#offset_const..)
                        .map(|slot| ::vortex_error::VortexExpect::vortex_expect(
                            slot,
                            #expect_message,
                        ))
                        .collect();
                    Self {
                        #(#owned_from_slots,)*
                        #tail_ident: __variadic_tail,
                    }
                }
            }
        }
    }
}

fn gen_into_slots(
    fixed_specs: &[&SlotField],
    tail_spec: Option<&SlotField>,
) -> proc_macro2::TokenStream {
    let fixed_into_slots = fixed_specs.iter().copied().map(SlotField::storage_slot);

    match tail_spec {
        None => quote! {
            pub fn into_slots(self) -> ::vortex_array::ArraySlots {
                ::vortex_array::smallvec::smallvec![#(#fixed_into_slots),*]
            }
        },
        Some(tail) => {
            let tail_ident = &tail.field_ident;
            quote! {
                pub fn into_slots(self) -> ::vortex_array::ArraySlots {
                    let mut slots: ::vortex_array::ArraySlots =
                        ::vortex_array::smallvec::smallvec![#(#fixed_into_slots),*];
                    slots.extend(self.#tail_ident.into_iter().map(Some));
                    slots
                }
            }
        }
    }
}

struct SlotField {
    field_ident: Ident,
    field_vis: Visibility,
    const_ident: Ident,
    slot_name: String,
    slot_type: SlotFieldType,
    index: usize,
    index_span: Span,
    expect_message: syn::LitStr,
    struct_ident: Ident,
}

impl SlotField {
    fn new(field: &Field, struct_ident: &Ident) -> syn::Result<Self> {
        let field_ident = field
            .ident
            .clone()
            .ok_or_else(|| syn::Error::new(field.span(), "slot fields must be named"))?;
        let field_name = ident_name(&field_ident);
        let slot_type = SlotFieldType::from_syn_type(&field.ty)?;
        let annotation = SlotIndexAttr::from_field(field, &field_name)?;

        match (slot_type, annotation.variadic) {
            (SlotFieldType::VariadicTail, false) => {
                return Err(syn::Error::new(
                    annotation.span,
                    format!(
                        "`{field_name}` is a variadic `Vec<ArrayRef>` tail, so it must be \
                         annotated `#[slot({}..)]`",
                        annotation.index
                    ),
                ));
            }
            (SlotFieldType::Required | SlotFieldType::Optional, true) => {
                return Err(syn::Error::new(
                    annotation.span,
                    format!(
                        "`#[slot(N..)]` declares a variadic `Vec<ArrayRef>` tail; `{field_name}` \
                         occupies a single slot, so annotate it `#[slot({})]`",
                        annotation.index
                    ),
                ));
            }
            _ => {}
        }

        let const_ident = match slot_type {
            SlotFieldType::VariadicTail => {
                format_ident!("{}_OFFSET", to_screaming_snake_case(&field_name))
            }
            _ => format_ident!("{}", to_screaming_snake_case(&field_name)),
        };
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
            index: annotation.index,
            index_span: annotation.span,
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
            SlotFieldType::VariadicTail => quote! {
                #field_ident: ::vortex_array::SlotSlice::new(
                    &slots[#struct_ident::#const_ident..],
                    #expect_message,
                )
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
            SlotFieldType::VariadicTail => quote! {
                #field_ident: self.#field_ident.to_vec()
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
            SlotFieldType::VariadicTail => {
                unreachable!("variadic tail is drained before fixed fields")
            }
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
            SlotFieldType::VariadicTail => {
                unreachable!("variadic tail is appended after fixed fields")
            }
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
            SlotFieldType::VariadicTail => quote! {
                #[inline]
                fn #field_ident(&self) -> ::vortex_array::SlotSlice<'_> {
                    ::vortex_array::SlotSlice::new(
                        &self.as_ref().slots()[#struct_ident::#const_ident..],
                        #expect_message,
                    )
                }
            },
        }
    }
}

/// A parsed `#[slot(N)]` or `#[slot(N..)]` field annotation.
struct SlotIndexAttr {
    index: usize,
    /// Whether the trailing `..` was present, marking a variadic tail.
    variadic: bool,
    span: Span,
}

impl SlotIndexAttr {
    fn from_field(field: &Field, field_name: &str) -> syn::Result<Self> {
        let mut annotation = None;
        for attr in &field.attrs {
            if !attr.path().is_ident(SLOT_ATTR) {
                continue;
            }
            if annotation.is_some() {
                return Err(syn::Error::new(
                    attr.span(),
                    format!("`{field_name}` has more than one `#[slot(..)]` attribute"),
                ));
            }
            annotation = Some(attr.parse_args::<Self>()?);
        }

        annotation.ok_or_else(|| {
            syn::Error::new(
                field.span(),
                format!(
                    "`{field_name}` is missing a `#[slot(N)]` attribute; every field of an \
                     `#[array_slots]` struct must pin itself to a slot index so that reordering \
                     field declarations cannot change the slot layout"
                ),
            )
        })
    }
}

impl Parse for SlotIndexAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let literal: LitInt = input.parse()?;
        let index = literal.base10_parse::<usize>()?;
        let variadic = input.peek(Token![..]);
        if variadic {
            input.parse::<Token![..]>()?;
        }
        if !input.is_empty() {
            return Err(input.error("expected `#[slot(N)]` or `#[slot(N..)]`"));
        }

        Ok(Self {
            index,
            variadic,
            span: literal.span(),
        })
    }
}

#[derive(Clone, Copy)]
enum SlotFieldType {
    Required,
    Optional,
    VariadicTail,
}

impl SlotFieldType {
    fn from_syn_type(ty: &Type) -> syn::Result<Self> {
        if is_array_ref_type(ty) {
            return Ok(Self::Required);
        }

        if let Some(inner_ty) = wrapper_inner_type(ty, "Option")
            && is_array_ref_type(inner_ty)
        {
            return Ok(Self::Optional);
        }

        if let Some(inner_ty) = wrapper_inner_type(ty, "Vec")
            && is_array_ref_type(inner_ty)
        {
            return Ok(Self::VariadicTail);
        }

        Err(syn::Error::new(
            ty.span(),
            "#[array_slots] fields must be ArrayRef, Option<ArrayRef>, or Vec<ArrayRef>",
        ))
    }

    fn view_field_ty(self) -> proc_macro2::TokenStream {
        match self {
            Self::Required => quote! { &'a ::vortex_array::ArrayRef },
            Self::Optional => quote! { Option<&'a ::vortex_array::ArrayRef> },
            Self::VariadicTail => quote! { ::vortex_array::SlotSlice<'a> },
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

fn wrapper_inner_type<'a>(ty: &'a Type, wrapper: &str) -> Option<&'a Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != wrapper {
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
