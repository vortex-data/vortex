use std::future::{ready, Future};
use std::io::{self, Cursor, Write};

use crate::IoBuf;

pub trait VortexWrite {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>>;
    fn flush(&mut self) -> impl Future<Output = io::Result<()>>;
    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>>;
}

impl VortexWrite for Vec<u8> {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        self.extend_from_slice(buffer.as_slice());
        ready(Ok(buffer))
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Ok(()))
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Ok(()))
    }
}

impl<T> VortexWrite for Cursor<T>
where
    Cursor<T>: Write,
{
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        ready(Write::write_all(self, buffer.as_slice()).map(|_| buffer))
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Write::flush(self))
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        ready(Ok(()))
    }
}

impl<W: VortexWrite> VortexWrite for futures::io::Cursor<W> {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        self.set_position(self.position() + buffer.as_slice().len() as u64);
        VortexWrite::write_all(self.get_mut(), buffer)
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        VortexWrite::flush(self.get_mut())
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        VortexWrite::shutdown(self.get_mut())
    }
}

impl<W: VortexWrite> VortexWrite for &mut W {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        (*self).write_all(buffer)
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        (*self).flush()
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        (*self).shutdown()
    }
}
