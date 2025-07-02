use vortex_array::VTableRegistry;

use crate::ExprEncodingRef;

pub type ExprRegistry = VTableRegistry<ExprEncodingRef>;

pub trait ExprRegistryExt {
    /// Creates a default expression registry with built-in Vortex expressions pre-registered.
    fn default() -> Self;
}

impl ExprRegistryExt for ExprRegistry {
    fn default() -> Self {
        let mut this = Self::empty();
        this.register_many([
            ExprEncodingRef::new_ref(&BetweenEncoding),
            ExprEncodingRef::new_ref(&Binary),
            ExprEncodingRef::new_ref(&GetItem),
            ExprEncodingRef::new_ref(&Identity),
            ExprEncodingRef::new_ref(&Like),
            ExprEncodingRef::new_ref(&Literal),
            ExprEncodingRef::new_ref(&ListContains),
            ExprEncodingRef::new_ref(&Merge),
            ExprEncodingRef::new_ref(&Not),
            ExprEncodingRef::new_ref(&Pack),
            ExprEncodingRef::new_ref(&Select),
            ExprEncodingRef::new_ref(&Var),
        ]);
        this
    }
}
