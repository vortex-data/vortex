// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TCP transport for streaming MPEG-TS packets to a remote viewer.

use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::TcpStream;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

/// TCP sender that accepts a single client and writes raw MPEG-TS.
///
/// Connect with: `ffplay tcp://host:port`
pub struct TcpSender {
    stream: TcpStream,
}

impl TcpSender {
    /// Creates a TCP listener on the given port and waits for a connection.
    pub async fn listen(port: u16) -> VortexResult<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))
            .await
            .map_err(|e| vortex_err!("TCP bind failed: {e}"))?;

        tracing::info!("Waiting for TCP connection on port {port}...");
        tracing::info!("Connect with: ffplay tcp://localhost:{port}");

        let (stream, addr) = listener
            .accept()
            .await
            .map_err(|e| vortex_err!("TCP accept failed: {e}"))?;

        stream
            .set_nodelay(true)
            .map_err(|e| vortex_err!("TCP set nodelay failed: {e}"))?;

        tracing::info!("Client connected from {addr}");

        Ok(Self { stream })
    }

    /// Sends MPEG-TS packets over TCP.
    pub async fn send(&mut self, data: Vec<u8>) -> VortexResult<()> {
        self.stream
            .write_all(&data)
            .await
            .map_err(|e| vortex_err!("TCP send failed: {e}"))?;
        Ok(())
    }

    /// Closes the TCP connection gracefully.
    pub async fn close(mut self) -> VortexResult<()> {
        self.stream
            .shutdown()
            .await
            .map_err(|e| vortex_err!("TCP close failed: {e}"))?;
        Ok(())
    }
}
