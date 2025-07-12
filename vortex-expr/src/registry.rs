// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::VTableRegistry;

use crate::{
    BetweenExprEncoding, BinaryExprEncoding, ExprEncodingRef, GetItemExprEncoding,
    LikeExprEncoding, ListContainsExprEncoding, LiteralExprEncoding, MergeExprEncoding,
    NotExprEncoding, PackExprEncoding, RootExprEncoding, SelectExprEncoding,
};

pub type ExprRegistry = VTableRegistry<ExprEncodingRef>;

pub trait ExprRegistryExt {
    /// Creates a default expression registry with built-in Vortex expressions pre-registered.
    fn default() -> Self;
}

impl ExprRegistryExt for ExprRegistry {
    fn default() -> Self {
        let mut this = Self::empty();
        this.register_many([
            ExprEncodingRef::new_ref(BetweenExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(BinaryExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(GetItemExprEncoding.as_ref()),
            // ExprEncodingRef::new_ref(IdentityExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(LikeExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(LiteralExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(ListContainsExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(MergeExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(NotExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(PackExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(SelectExprEncoding.as_ref()),
            ExprEncodingRef::new_ref(RootExprEncoding.as_ref()),
        ]);
        this
    }
}
