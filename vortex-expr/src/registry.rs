use vortex_array::VTableRegistry;

use crate::{
    BetweenExprEncoding, BinaryExprEncoding, ExprEncodingRef, GetItemExprEncoding,
    LikeExprEncoding, ListContainsExprEncoding, LiteralExprEncoding, MergeExprEncoding,
    NotExprEncoding, PackExprEncoding, SelectExprEncoding, VarExprEncoding,
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
            ExprEncodingRef::new_ref(&BetweenExprEncoding),
            ExprEncodingRef::new_ref(&BinaryExprEncoding),
            ExprEncodingRef::new_ref(&GetItemExprEncoding),
            // ExprEncodingRef::new_ref(&IdentityExprEncoding),
            ExprEncodingRef::new_ref(&LikeExprEncoding),
            ExprEncodingRef::new_ref(&LiteralExprEncoding),
            ExprEncodingRef::new_ref(&ListContainsExprEncoding),
            ExprEncodingRef::new_ref(&MergeExprEncoding),
            ExprEncodingRef::new_ref(&NotExprEncoding),
            ExprEncodingRef::new_ref(&PackExprEncoding),
            ExprEncodingRef::new_ref(&SelectExprEncoding),
            ExprEncodingRef::new_ref(&VarExprEncoding),
        ]);
        this
    }
}
