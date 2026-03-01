// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SRT transport for streaming MPEG-TS packets to a remote viewer.

use std::time::Instant;

use bytes::Bytes;
use futures::SinkExt;
use srt_tokio::SrtSocket;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

/// SRT sender in listener mode.
///
/// Waits for a client to connect, then sends MPEG-TS data.
pub struct SrtSender {
    socket: SrtSocket,
}

impl SrtSender {
    /// Creates an SRT listener on the given port and waits for a connection.
    pub async fn listen(port: u16) -> VortexResult<Self> {
        tracing::info!("Waiting for SRT connection on port {port}...");
        tracing::info!("Connect with: vlc srt://<host>:{port}");

        let socket = SrtSocket::builder()
            .listen_on(port)
            .await
            .map_err(|e| vortex_err!("SRT listen failed: {e}"))?;

        tracing::info!("SRT client connected");

        Ok(Self { socket })
    }

    /// Sends MPEG-TS packets over SRT.
    pub async fn send(&mut self, data: Vec<u8>) -> VortexResult<()> {
        self.socket
            .send((Instant::now(), Bytes::from(data)))
            .await
            .map_err(|e| vortex_err!("SRT send failed: {e:?}"))?;
        Ok(())
    }

    /// Closes the SRT connection gracefully.
    pub async fn close(self) -> VortexResult<()> {
        let mut socket = self.socket;
        socket
            .close()
            .await
            .map_err(|e| vortex_err!("SRT close failed: {e:?}"))?;
        Ok(())
    }
}
