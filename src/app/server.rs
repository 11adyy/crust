use std::sync::Arc;

use crate::logger::{LogLevel, log};
use crate::multiplayer::protocol::Packet;
use crate::multiplayer::tcp::TcpServer;

/// Runs a standalone dedicated multiplayer server that accepts TCP connections
/// and relays packets between all connected clients.
///
/// # Lifecycle
///
/// 1. Binds a [`TcpServer`] to `addr`.
/// 2. Enters an infinite accept loop on the calling task.
/// 3. For each accepted connection, spawns a dedicated Tokio task that owns
///    the receive loop for that client.
/// 4. The server runs until the process is killed; there is currently no
///    graceful shutdown signal.
///
/// # Packet handling
///
/// Every packet received from a client is mutated so its `player_id` field
/// reflects the server-assigned connection ID rather than whatever the client
/// sent.  This prevents clients from spoofing another player's identity.
///
/// | Packet variant   | Server action                                                    |
/// |------------------|------------------------------------------------------------------|
/// | `Connect`        | Overwrites `player_id`; sends a `ConnectAck` back to the sender.|
/// | `Position`       | Overwrites `player_id`; broadcast to all other clients.         |
/// | `Rotation`       | Overwrites `player_id`; broadcast to all other clients.         |
/// | `Chat`           | Overwrites `player_id`; broadcast to all other clients.         |
/// | `Disconnect`     | Overwrites `player_id`; broadcast to all other clients.         |
/// | All other types  | Broadcast as-is (no mutation).                                  |
///
/// On a receive error the client is considered disconnected: a synthetic
/// `Disconnect` packet is broadcast to all remaining peers and the client is
/// removed from the server's connection table.
///
/// # Parameters
/// - `addr` – The `host:port` string to listen on (e.g. `"0.0.0.0:25565"`).
///
/// # Errors
/// Logs to `stderr` and returns early if the server cannot bind to `addr`.
/// Per-client receive/send errors are logged but do not terminate the server.
pub async fn run_dedicated_server(addr: &str) {
    match TcpServer::bind(addr).await {
        Ok(server_inst) => {
            
            
            let server = Arc::new(server_inst);
            log(
                LogLevel::Info,
                &format!("Server successfully bound to {}", addr),
            );
            log(LogLevel::Info, "Waiting for connections...");
            
            
            let _ = std::io::Write::flush(&mut std::io::stdout());

            let server_seed: u32 = rand::random();
            log(LogLevel::Info, &format!("Server world seed: {}", server_seed));

            
            
            
            loop {
                match server.accept().await {
                    Ok((id, conn)) => {
                        log(
                            LogLevel::Info,
                            &format!(
                                "Accepted connection from {} with assigned ID {}",
                                conn.addr(),
                                id
                            ),
                        );
                        
                        
                        let server_clone = server.clone();

                        
                        tokio::spawn(async move {
                            loop {
                                match conn.recv().await {
                                    Ok(mut packet) => {
                                        
                                        
                                        
                                        
                                        
                                        
                                        
                                        match packet {
                                            Packet::Connect {
                                                ref mut player_id, ..
                                            } => {
                                                *player_id = id;
                                                
                                                
                                                
                                                
                                                let ack = Packet::ConnectAck {
                                                    success: true,
                                                    player_id: id,
                                                    seed: server_seed,
                                                };
                                                let _ = conn.send(&ack).await;
                                            }
                                            Packet::Position {
                                                ref mut player_id, ..
                                            } => {
                                                *player_id = id;
                                            }
                                            Packet::Rotation {
                                                ref mut player_id, ..
                                            } => {
                                                *player_id = id;
                                            }
                                            Packet::Chat {
                                                ref mut player_id, ..
                                            } => {
                                                *player_id = id;
                                            }
                                            Packet::Disconnect {
                                                ref mut player_id, ..
                                            } => {
                                                *player_id = id;
                                            }
                                            
                                            
                                            
                                            _ => {}
                                        }

                                        
                                        
                                        
                                        
                                        
                                        let _ = server_clone.broadcast_except(&packet, id).await;
                                    }

                                    
                                    
                                    
                                    Err(_) => {
                                        log(
                                            LogLevel::Info,
                                            &format!(
                                                "Connection error with client {}; treating as disconnect",
                                                id
                                            ),
                                        );
                                        
                                        
                                        
                                        
                                        let disconnect_packet =
                                            Packet::Disconnect { player_id: id };
                                        let _ = server_clone
                                            .broadcast_except(&disconnect_packet, id)
                                            .await;

                                        
                                        
                                        
                                        server_clone.remove_client(id).await;

                                        
                                        
                                        break;
                                    }
                                }
                            }
                        });
                    }

                    Err(e) => {
                        
                        
                        log(LogLevel::Error, &format!("Accept error: {}", e));
                    }
                }
            }
        }

        Err(e) => {
            log(
                LogLevel::Error,
                &format!("Failed to bind server to {}: {}", addr, e),
            );
        }
    }
}
