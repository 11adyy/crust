use crust::camera::check_intersection;
use winit::event::MouseButton;

use crate::ui::menu::{MenuField, MenuHit, MenuLayout};
use crate::ui::ui::HOTBAR_SLOTS;

use super::state::State;

impl State {
    /// Translates a raw mouse-click position into a menu action.
    ///
    /// Called whenever the player clicks while the game is in
    /// [`GameState::Menu`].  A [`MenuLayout`] is constructed from the current
    /// surface dimensions so that hit regions scale correctly at any
    /// resolution, then a point-in-rect hit test determines which (if any)
    /// interactive element was clicked.
    ///
    /// # Behavior per hit result
    /// | `MenuHit` variant       | Effect                                                  |
    /// |-------------------------|---------------------------------------------------------|
    /// | `ServerAddress`         | Moves keyboard focus to the server-address text field.  |
    /// | `Username`              | Moves keyboard focus to the username text field.        |
    /// | `Connect`               | Initiates a multiplayer connection attempt.             |
    /// | `Singleplayer`          | Transitions directly to `GameState::Playing`.           |
    /// | `None` (missed all UI)  | Clears the active field so keyboard input is ignored.   |
    ///
    /// # Parameters
    /// - `x` – Horizontal cursor position in physical pixels (origin = top-left).
    /// - `y` – Vertical cursor position in physical pixels.
    pub fn handle_menu_click(&mut self, x: f32, y: f32) {
        let layout = MenuLayout::new(self.config.width, self.config.height);

        match layout.hit_test(x, y) {
            Some(MenuHit::ServerAddress) => self.menu_state.select_field(MenuField::ServerAddress),
            Some(MenuHit::Username) => self.menu_state.select_field(MenuField::Username),
            Some(MenuHit::Connect) => self.connect_to_server(),
            Some(MenuHit::Singleplayer) => self.game_state = crate::ui::menu::GameState::Playing,
            
            
            None => self.menu_state.select_field(MenuField::None),
        }
    }

    /// Processes a mouse-button press or release event.
    ///
    /// This method has two responsibilities:
    ///
    /// 1. **Always**: mirror the button state into [`InputState`] so that
    ///    continuous per-frame logic (e.g., left-click mining) can poll it
    ///    without re-processing events.
    ///
    /// 2. **Only when the mouse is captured** (i.e., the player is in-game
    ///    with the cursor locked): handle the right-click block-placement
    ///    action, including all placement guards.
    ///
    /// # Block placement guards (right-click)
    /// Placement is skipped when any of the following is true:
    /// - The raycast does not hit a surface within reach (5 blocks).
    /// - The target placement position overlaps the player's own AABB —
    ///   prevents the player from trapping themselves inside a block.
    /// - The target position overlaps a remote player's AABB — prevents
    ///   griefing by walling another player in.
    ///
    /// When all guards pass, the block currently selected in the hotbar is
    /// written to the world and the affected chunk is mark`ed dirty so its
    /// mesh is rebuilt on the next frame.
    ///
    /// # Parameters
    /// - `button`  – Which mouse button changed state.
    /// - `pressed` – `true` on press, `false` on release.
    pub fn handle_mouse_input(&mut self, button: MouseButton, pressed: bool) {
        
        match button {
            MouseButton::Left => self.input.left_mouse = pressed,
            MouseButton::Right => self.input.right_mouse = pressed,
            _ => {}
        }

        
        
        
        if !self.mouse_captured {
            return;
        }

        if button == MouseButton::Right && pressed {
            
            
            
            
            
            let target = self.camera.raycast(&*self.world.read(), 5.0);
            if let Some((_, _, _, px, py, pz)) = target {
                
                if self.camera.intersects_block(px, py, pz) {
                    return;
                }

                
                
                
                for player in self.remote_players.values() {
                    let player_pos = glam::Vec3::new(player.x, player.y, player.z);
                    if check_intersection(player_pos, px, py, pz) {
                        return;
                    }
                }

                
                let block_to_place = HOTBAR_SLOTS[self.hotbar_slot];
                self.world
                    .write()
                    .set_block_player(px, py, pz, block_to_place);

                
                if let Some(tx) = &self.network_tx {
                    let _ = tx.send(crate::multiplayer::protocol::Packet::BlockChange {
                        x: px,
                        y: py,
                        z: pz,
                        block_type: block_to_place as u8,
                    });
                }

                
                
                self.mark_chunk_dirty(px, py, pz);
            }
        }
    }

    /// Initiates an asynchronous connection to the multiplayer server.
    ///
    /// Reads the server address and username from [`MenuState`] and spawns
    /// the networking tasks on the shared Tokio runtime.  On success the
    /// `network_rx` / `network_tx` channels are populated so that
    /// `update_network` can exchange packets with the server each frame.
    ///
    /// The [`MenuState`] status message is updated in-place by
    /// `connect_to_server` to reflect connection progress or any error that
    /// occurs (e.g., DNS failure, refused connection).
    ///
    /// # Note
    /// This method is a thin forwarding shim that exists so menu click
    /// handling (`handle_menu_click`) does not need to borrow individual
    /// fields of `self` separately — `&mut self` here satisfies the borrow
    /// checker cleanly.
    pub fn connect_to_server(&mut self) {
        use crate::multiplayer::network::connect_to_server;
        connect_to_server(
            &mut self.menu_state,
            &mut self.game_state,
            &self.network_runtime,
            &mut self.network_rx,
            &mut self.network_tx,
        );
    }
}
