use vortex_array::{ArrayDType, Canonical, IntoArrayData};
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::{ExprRef, Identity};

/// The definition of a range scan.
#[derive(Debug, Clone)]
pub struct Scan {
    pub projection: ExprRef,
    pub filter: Option<ExprRef>,
}

impl Scan {
    /// Scan all rows with the identity projection.
    pub fn all() -> Self {
        Self {
            projection: Identity::new_expr(),
            filter: None,
        }
    }

    /// Compute the result dtype of the scan given the input dtype.
    pub fn result_dtype(&self, dtype: &DType) -> VortexResult<DType> {
        Ok(self
            .projection
            .evaluate(&Canonical::empty(dtype)?.into_array())?
            .dtype()
            .clone())
    }
}
