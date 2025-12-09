// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::vortex_ensure;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_vector::Datum;
use vortex_vector::Vector;
use vortex_vector::VectorOps;

use crate::arrays::ConstantVTable;
use crate::kernel::BindCtx;
use crate::session::ArraySessionExt;
use crate::Array;
use crate::ArrayRef;

/// Executor for exporting a Vortex [`Vector`] or [`Datum`] from an [`ArrayRef`].
pub trait VectorExecutor {
    /// Execute the array and return the resulting datum after running the optimizer.
    fn execute_datum_optimized(&self, session: &VortexSession) -> VortexResult<Datum>;
    /// Execute the array and return the resulting datum.
    fn execute_datum(&self, session: &VortexSession) -> VortexResult<Datum>;
    /// Execute the array and return the resulting vector after running the optimizer.
    fn execute_vector_optimized(&self, session: &VortexSession) -> VortexResult<Vector>;
    /// Execute the array and return the resulting vector.
    fn execute_vector(&self, session: &VortexSession) -> VortexResult<Vector>;
}

impl VectorExecutor for ArrayRef {
    fn execute_datum_optimized(&self, session: &VortexSession) -> VortexResult<Datum> {
        session
            .arrays()
            .optimizer()
            .optimize_array(self)?
            .execute_datum(session)
    }

    fn execute_datum(&self, session: &VortexSession) -> VortexResult<Datum> {
        log::error!("Executing array: {}", self.display_tree());

        // Attempt to short-circuit constant arrays.
        if let Some(constant) = self.as_opt::<ConstantVTable>() {
            return Ok(Datum::Scalar(constant.scalar().to_vector_scalar()));
        }

        let mut ctx = BindCtx::new(session.clone());

        // NOTE(ngates): in the future we can choose a different mode of execution, or run
        // optimization here, etc.
        let kernel = self.bind_kernel(&mut ctx)?;
        let result = kernel.execute()?;

        vortex_ensure!(
            result.len() == self.len(),
            "Result length mismatch for {}",
            self.encoding_id()
        );

        #[cfg(debug_assertions)]
        {
            vortex_ensure!(
                vortex_vector::vector_matches_dtype(&result, self.dtype()),
                "Executed vector dtype mismatch for {}",
                self.encoding_id(),
            );
        }

        Ok(Datum::Vector(result))
    }

    fn execute_vector_optimized(&self, session: &VortexSession) -> VortexResult<Vector> {
        session
            .arrays()
            .optimizer()
            .optimize_array(self)?
            .execute_vector(session)
    }

    fn execute_vector(&self, session: &VortexSession) -> VortexResult<Vector> {
        let len = self.len();
        Ok(self.execute_datum(session)?.ensure_vector(len))
    }
}
