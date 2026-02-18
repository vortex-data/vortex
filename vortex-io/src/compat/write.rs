// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::IoBuf;
use crate::VortexWrite;
use crate::compat::Compat;

/// Compatibility adapter for `VortexWrite` implementations that are based on Tokio.
#[deny(clippy::missing_trait_methods)]
impl<W: VortexWrite> VortexWrite for Compat<W> {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = std::io::Result<B>> {
        Compat::new(self.inner_mut().write_all(buffer))
    }

    fn flush(&mut self) -> impl Future<Output = std::io::Result<()>> {
        Compat::new(self.inner_mut().flush())
    }

    fn shutdown(&mut self) -> impl Future<Output = std::io::Result<()>> {
        Compat::new(self.inner_mut().shutdown())
    }
}
