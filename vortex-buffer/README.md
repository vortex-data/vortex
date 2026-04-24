# Vortex Buffer

For now, a Vortex buffer is implemented as a very thin wrapper around the Tokio bytes crate.
In the future, we may re-implement this ourselves to have more control over alignment
(see https://github.com/tokio-rs/bytes/issues/437)
