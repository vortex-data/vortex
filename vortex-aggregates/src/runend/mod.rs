// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate kernels for run-end encoded arrays.

pub(crate) mod is_constant;
pub(crate) mod is_sorted;
pub(crate) mod min_max;
pub(crate) mod sum;

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::RunEnd;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::is_constant::is_constant;

    pub(crate) static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    #[test]
    fn sliced_single_run_is_constant() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new(
            buffer![7_u64, 10].into_array(),
            buffer![2_u64, 3].into_array(),
            &mut ctx,
        )?;

        assert!(is_constant(&array.slice(2..5)?, &mut ctx)?);
        Ok(())
    }
}
