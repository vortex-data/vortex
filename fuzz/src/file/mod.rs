use libfuzzer_sys::arbitrary::{Arbitrary, Unstructured};
use vortex_array::ArrayRef;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_expr::ExprRef;
use vortex_expr::arbitrary::{filter_expr, projection_expr};

#[derive(Debug)]
pub struct FuzzFileAction {
    pub array: ArrayRef,
    pub projection: Option<ExprRef>,
    pub filter: Option<ExprRef>,
}

impl<'a> Arbitrary<'a> for FuzzFileAction {
    fn arbitrary(u: &mut Unstructured<'a>) -> libfuzzer_sys::arbitrary::Result<Self> {
        let array = ArbitraryArray::arbitrary(u)?.0;
        let dtype = array.dtype().clone();
        Ok(FuzzFileAction {
            array,
            projection: projection_expr(u, &dtype)?,
            filter: filter_expr(u, &dtype)?,
        })
    }
}
