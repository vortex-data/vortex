// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use proc_macro2::TokenStream;
use quote::quote;
use syn::Expr;
use syn::ExprLit;
use syn::Fields;
use syn::GenericArgument;
use syn::ImplItem;
use syn::Item;
use syn::ItemImpl;
use syn::ItemStruct;
use syn::ItemTrait;
use syn::Lit;
use syn::Path;
use syn::PathArguments;
use syn::TraitItem;
use syn::Type;
use syn::Visibility;

use super::RESERVED_FIELD_NAMES;
use super::expand_array_slots;
use super::to_screaming_snake_case;

fn expand(encoding: &str, item: TokenStream) -> syn::Result<syn::File> {
    let encoding: Path = syn::parse_str(encoding)?;
    let item_struct: ItemStruct = syn::parse2(item)?;
    let tokens = expand_array_slots(encoding, item_struct)?;
    syn::parse2(tokens)
}

fn expand_err(encoding: &str, item: TokenStream) -> String {
    match expand(encoding, item) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("expected macro expansion error"),
    }
}

fn find_struct<'a>(file: &'a syn::File, name: &str) -> &'a ItemStruct {
    file.items
        .iter()
        .find_map(|item| match item {
            Item::Struct(s) if s.ident == name => Some(s),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing struct `{name}`"))
}

fn find_inherent_impl<'a>(file: &'a syn::File, ty_name: &str) -> &'a ItemImpl {
    file.items
        .iter()
        .find_map(|item| match item {
            Item::Impl(imp) if imp.trait_.is_none() => match imp.self_ty.as_ref() {
                Type::Path(tp)
                    if tp
                        .path
                        .segments
                        .last()
                        .is_some_and(|seg| seg.ident == ty_name) =>
                {
                    Some(imp)
                }
                _ => None,
            },
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing inherent impl for `{ty_name}`"))
}

fn find_trait<'a>(file: &'a syn::File, name: &str) -> &'a ItemTrait {
    file.items
        .iter()
        .find_map(|item| match item {
            Item::Trait(tr) if tr.ident == name => Some(tr),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing trait `{name}`"))
}

fn const_usize(imp: &ItemImpl, name: &str) -> Option<usize> {
    imp.items.iter().find_map(|item| match item {
        ImplItem::Const(c) if c.ident == name => match &c.expr {
            Expr::Lit(ExprLit {
                lit: Lit::Int(int), ..
            }) => int.base10_parse().ok(),
            _ => None,
        },
        _ => None,
    })
}

fn const_str_array(imp: &ItemImpl, name: &str) -> Option<Vec<String>> {
    imp.items.iter().find_map(|item| match item {
        ImplItem::Const(c) if c.ident == name => match &c.expr {
            Expr::Array(arr) => arr
                .elems
                .iter()
                .map(|elem| match elem {
                    Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    }) => Some(s.value()),
                    _ => None,
                })
                .collect(),
            _ => None,
        },
        _ => None,
    })
}

fn has_method(imp: &ItemImpl, name: &str) -> bool {
    imp.items
        .iter()
        .any(|item| matches!(item, ImplItem::Fn(f) if f.sig.ident == name))
}

fn last_path_ident(ty: &Type) -> Option<&syn::Ident> {
    match ty {
        Type::Path(tp) => tp.path.segments.last().map(|seg| &seg.ident),
        _ => None,
    }
}

fn is_ref_array_ref(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Reference(r) if last_path_ident(&r.elem).is_some_and(|i| i == "ArrayRef")
    )
}

fn is_option_ref_array_ref(ty: &Type) -> bool {
    let Type::Path(tp) = ty else { return false };
    let Some(seg) = tp.path.segments.last() else {
        return false;
    };
    if seg.ident != "Option" {
        return false;
    }
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return false;
    };
    matches!(
        args.args.first(),
        Some(GenericArgument::Type(inner)) if is_ref_array_ref(inner)
    )
}

#[test]
fn fixed_slots_reordered_strips_slot_attrs_and_orders_storage() {
    let file = expand(
        "crate::encodings::Patched",
        quote! {
            pub struct PatchedSlots {
                #[slot(2)]
                pub(crate) patch_values: Option<vortex_array::ArrayRef>,
                /// Kept doc attribute.
                #[slot(0)]
                pub inner: ArrayRef,
                #[slot(1)]
                pub lane_offsets: std::option::Option<ArrayRef>,
            }
        },
    )
    .unwrap();

    // Original struct is re-emitted without `#[slot(..)]`, preserving other attributes.
    let emitted = find_struct(&file, "PatchedSlots");
    let Fields::Named(fields) = &emitted.fields else {
        panic!("expected named fields");
    };
    assert_eq!(fields.named.len(), 3);
    for field in &fields.named {
        assert!(
            field.attrs.iter().all(|attr| !attr.path().is_ident("slot")),
            "#[slot] attribute should be stripped"
        );
    }
    let inner_field = fields
        .named
        .iter()
        .find(|f| f.ident.as_ref().is_some_and(|i| i == "inner"))
        .unwrap();
    assert!(
        inner_field
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("doc"))
    );

    // Storage constants follow #[slot(..)] indices, not declaration order.
    let slots_impl = find_inherent_impl(&file, "PatchedSlots");
    assert_eq!(const_usize(slots_impl, "INNER"), Some(0));
    assert_eq!(const_usize(slots_impl, "LANE_OFFSETS"), Some(1));
    assert_eq!(const_usize(slots_impl, "PATCH_VALUES"), Some(2));
    assert_eq!(const_usize(slots_impl, "COUNT"), Some(3));
    assert_eq!(
        const_str_array(slots_impl, "NAMES"),
        Some(vec![
            "inner".to_owned(),
            "lane_offsets".to_owned(),
            "patch_values".to_owned(),
        ])
    );
    assert!(has_method(slots_impl, "from_slots"));
    assert!(has_method(slots_impl, "into_slots"));

    // View struct preserves visibilities and maps required vs optional types.
    let view = find_struct(&file, "PatchedSlotsView");
    let Fields::Named(view_fields) = &view.fields else {
        panic!("expected named view fields");
    };
    let view_by_name: Vec<(&str, bool, bool)> = view_fields
        .named
        .iter()
        .map(|f| {
            (
                match f.ident.as_ref().unwrap().to_string().as_str() {
                    "patch_values" => "patch_values",
                    "inner" => "inner",
                    "lane_offsets" => "lane_offsets",
                    _ => "other",
                },
                matches!(f.vis, Visibility::Public(_)),
                is_ref_array_ref(&f.ty),
            )
        })
        .collect();
    assert_eq!(
        view_by_name,
        vec![
            ("patch_values", false, false),
            ("inner", true, true),
            ("lane_offsets", true, false),
        ]
    );
    assert!(is_option_ref_array_ref(&view_fields.named[0].ty));
    assert!(is_option_ref_array_ref(&view_fields.named[2].ty));

    let view_impl = find_inherent_impl(&file, "PatchedSlotsView");
    assert!(has_method(view_impl, "from_slots"));
    assert!(has_method(view_impl, "to_owned"));

    // Ext trait uses last path segment for its name and preserves method order + blanket impl.
    let ext = find_trait(&file, "PatchedArraySlotsExt");
    let ext_methods: Vec<String> = ext
        .items
        .iter()
        .filter_map(|item| match item {
            TraitItem::Fn(f) => Some(f.sig.ident.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(
        ext_methods,
        vec!["patch_values", "inner", "lane_offsets", "slots_view"]
    );
    assert!(file.items.iter().any(|item| matches!(
        item,
        Item::Impl(imp)
            if imp.trait_.as_ref().is_some_and(|(path, _)| {
                path.segments
                    .last()
                    .is_some_and(|seg| seg.ident == "PatchedArraySlotsExt")
            })
    )));
}

#[test]
fn variadic_tail_emits_offset_fixed_consts_slot_name_and_slot_slice() {
    let file = expand(
        "Chunked",
        quote! {
            pub struct ChunkedSlots {
                #[slot(1..)]
                pub chunks: std::vec::Vec<vortex_array::ArrayRef>,
                #[slot(0)]
                pub chunk_offsets: ArrayRef,
            }
        },
    )
    .unwrap();

    let slots_impl = find_inherent_impl(&file, "ChunkedSlots");
    assert_eq!(const_usize(slots_impl, "CHUNK_OFFSETS"), Some(0));
    assert_eq!(const_usize(slots_impl, "CHUNKS_OFFSET"), Some(1));
    assert_eq!(const_usize(slots_impl, "FIXED_COUNT"), Some(1));
    assert_eq!(
        const_str_array(slots_impl, "FIXED_NAMES"),
        Some(vec!["chunk_offsets".to_owned()])
    );
    assert_eq!(const_usize(slots_impl, "COUNT"), None);
    assert_eq!(const_str_array(slots_impl, "NAMES"), None);
    assert!(has_method(slots_impl, "slot_name"));
    assert!(has_method(slots_impl, "from_slots"));
    assert!(has_method(slots_impl, "into_slots"));

    let view = find_struct(&file, "ChunkedSlotsView");
    let Fields::Named(view_fields) = &view.fields else {
        panic!("expected named view fields");
    };
    let chunks_field = view_fields
        .named
        .iter()
        .find(|f| f.ident.as_ref().is_some_and(|i| i == "chunks"))
        .unwrap();
    assert!(last_path_ident(&chunks_field.ty).is_some_and(|i| i == "SlotSlice"));
}

#[test]
fn variadic_tail_only_and_empty_struct_expand_cleanly() {
    let tail_only = expand(
        "Union",
        quote! {
            pub struct UnionSlots {
                #[slot(0..)]
                pub children: Vec<ArrayRef>,
            }
        },
    )
    .unwrap();
    let tail_impl = find_inherent_impl(&tail_only, "UnionSlots");
    assert_eq!(const_usize(tail_impl, "CHILDREN_OFFSET"), Some(0));
    assert_eq!(const_usize(tail_impl, "FIXED_COUNT"), Some(0));
    assert_eq!(const_str_array(tail_impl, "FIXED_NAMES"), Some(vec![]));

    let empty = expand("Null", quote! { pub struct NullSlots {} }).unwrap();
    let empty_impl = find_inherent_impl(&empty, "NullSlots");
    assert_eq!(const_usize(empty_impl, "COUNT"), Some(0));
    assert_eq!(const_str_array(empty_impl, "NAMES"), Some(vec![]));
}

#[test]
fn raw_identifier_field_strips_prefix_for_const_and_name() {
    let file = expand(
        "Typed",
        quote! {
            pub struct TypedSlots {
                #[slot(0)]
                pub r#type: ArrayRef,
            }
        },
    )
    .unwrap();

    let slots_impl = find_inherent_impl(&file, "TypedSlots");
    assert_eq!(const_usize(slots_impl, "TYPE"), Some(0));
    assert_eq!(
        const_str_array(slots_impl, "NAMES"),
        Some(vec!["type".to_owned()])
    );
}

#[test]
fn screaming_snake_case_handles_snake_camel_and_digits() {
    assert_eq!(to_screaming_snake_case("inner"), "INNER");
    assert_eq!(to_screaming_snake_case("lane_offsets"), "LANE_OFFSETS");
    assert_eq!(to_screaming_snake_case("camelCase"), "CAMEL_CASE");
    assert_eq!(to_screaming_snake_case("v2Data"), "V2_DATA");
    assert_eq!(to_screaming_snake_case("ABC"), "ABC");
}

#[test]
fn rejects_invalid_structs_attrs_and_layouts() {
    let cases = [
        // Generics and non-named structs.
        (
            quote! { pub struct FooSlots<T> { #[slot(0)] pub a: ArrayRef } },
            "does not support generic slot structs",
        ),
        (
            quote! { pub struct FooSlots<'a> { #[slot(0)] pub a: ArrayRef } },
            "does not support generic slot structs",
        ),
        (
            quote! { pub struct FooSlots where u8: Copy { #[slot(0)] pub a: ArrayRef } },
            "does not support generic slot structs",
        ),
        (
            quote! { pub struct FooSlots(ArrayRef); },
            "requires a struct with named fields",
        ),
        (
            quote! { pub struct FooSlots; },
            "requires a struct with named fields",
        ),
        // Invalid #[slot(..)] syntax and presence.
        (
            quote! { pub struct FooSlots { pub child: ArrayRef } },
            "is missing a `#[slot(N)]` attribute",
        ),
        (
            quote! { pub struct FooSlots { #[slot(0)] #[slot(1)] pub child: ArrayRef } },
            "has more than one `#[slot(..)]` attribute",
        ),
        (
            quote! { pub struct FooSlots { #[slot(0..1)] pub child: Vec<ArrayRef> } },
            "expected `#[slot(N)]` or `#[slot(N..)]`",
        ),
        (
            quote! { pub struct FooSlots { #[slot(0, 1)] pub child: ArrayRef } },
            "expected `#[slot(N)]` or `#[slot(N..)]`",
        ),
        (
            quote! { pub struct FooSlots { #[slot(abc)] pub child: ArrayRef } },
            "expected integer literal",
        ),
        // Variadic vs fixed annotation mismatch.
        (
            quote! { pub struct FooSlots { #[slot(0)] pub children: Vec<ArrayRef> } },
            "is a variadic `Vec<ArrayRef>` tail, so it must be annotated `#[slot(0..)]`",
        ),
        (
            quote! { pub struct FooSlots { #[slot(0..)] pub child: ArrayRef } },
            "occupies a single slot, so annotate it `#[slot(0)]`",
        ),
        (
            quote! { pub struct FooSlots { #[slot(0..)] pub child: Option<ArrayRef> } },
            "occupies a single slot, so annotate it `#[slot(0)]`",
        ),
        // Invalid slot index layouts.
        (
            quote! {
                pub struct FooSlots {
                    #[slot(0..)] pub first: Vec<ArrayRef>,
                    #[slot(1..)] pub second: Vec<ArrayRef>,
                }
            },
            "allows at most one variadic tail, but `first` is already declared as one",
        ),
        (
            quote! {
                pub struct FooSlots {
                    #[slot(0)] pub first: ArrayRef,
                    #[slot(0)] pub second: ArrayRef,
                }
            },
            "slot index 0 is claimed by both `first` and `second`",
        ),
        (
            quote! {
                pub struct FooSlots {
                    #[slot(0)] pub first: ArrayRef,
                    #[slot(2)] pub third: ArrayRef,
                }
            },
            "no field claims slot index 1; fixed slot indices must cover 0..2 without gaps",
        ),
        (
            quote! { pub struct FooSlots { #[slot(1)] pub first: ArrayRef } },
            "no field claims slot index 0; fixed slot indices must cover 0..1 without gaps",
        ),
        (
            quote! {
                pub struct FooSlots {
                    #[slot(0)] pub first: ArrayRef,
                    #[slot(2..)] pub tail: Vec<ArrayRef>,
                }
            },
            "variadic tail `tail` must start at slot index 1, immediately after the 1 fixed slot(s), but is annotated `#[slot(2..)]`",
        ),
        (
            quote! {
                pub struct FooSlots {
                    #[slot(0)] pub first: ArrayRef,
                    #[slot(0..)] pub tail: Vec<ArrayRef>,
                }
            },
            "variadic tail `tail` must start at slot index 1",
        ),
    ];

    for (item, expected) in cases {
        let err = expand_err("Foo", item);
        assert!(err.contains(expected), "expected `{expected}`, got `{err}`");
    }
}

#[test]
fn rejects_unsupported_field_types() {
    let invalid_types = [
        quote! { u32 },
        quote! { Option<u32> },
        quote! { Vec<Option<ArrayRef>> },
        quote! { &ArrayRef },
        quote! { Option },
    ];

    for ty in invalid_types {
        let err = expand_err(
            "Foo",
            quote! {
                pub struct FooSlots {
                    #[slot(0)]
                    pub child: #ty,
                }
            },
        );
        assert!(
            err.contains("fields must be ArrayRef, Option<ArrayRef>, or Vec<ArrayRef>"),
            "unexpected error for `{ty}`: {err}"
        );
    }
}

#[test]
fn rejects_all_reserved_field_names_including_raw_idents() {
    for &name in RESERVED_FIELD_NAMES {
        let ident = quote::format_ident!("{name}");
        let err = expand_err(
            "Foo",
            quote! {
                pub struct FooSlots {
                    #[slot(0)]
                    pub #ident: ArrayRef,
                }
            },
        );
        assert!(
            err.contains(&format!("shadows `ArrayView::{name}`")),
            "reserved `{name}` gave: {err}"
        );
    }

    let err = expand_err(
        "Foo",
        quote! {
            pub struct FooSlots {
                #[slot(0)]
                pub r#validity: ArrayRef,
            }
        },
    );
    assert!(err.contains("shadows `ArrayView::validity`"), "{err}");
}
