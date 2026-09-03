#[cfg(target_os = "windows")]
use std::ffi::c_void;
use std::time::Instant;

use clap::Parser;
use winit::{
    dpi::PhysicalPosition,
    event::{DeviceEvent, ElementState, Event, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Fullscreen, Window},
};

use crust::{
    CHUNK_SIZE, DEFAULT_WORLD_FILE, SUBCHUNK_HEIGHT, SavedWorld, World, load_world, save_world,
};

use crate::logger::{LogLevel, log};
use crate::ui::menu::GameState;

use super::performance::process_cpu_time_ms;
use super::render::RenderError;
use super::server::run_dedicated_server;
use super::state::State;





/// Command-line arguments parsed by [`clap`] at startup.
///
/// Running with `--server` skips the windowed game entirely and starts a
/// headless TCP server instead.  The port can be overridden with `--port`.
///
/// # Examples
/// ```text
/// # Start a headless server on the default port
/// crust --server
///
/// # Start a headless server on a custom port
/// crust --server --port 12345
///
/// # Start the windowed game (default when no flags are given)
/// crust
/// ```
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run as a headless dedicated server instead of opening a game window.
    #[arg(long, default_value_t = false)]
    server: bool,

    /// TCP port the dedicated server listens on.
    #[arg(long, default_value_t = 25565)]
    port: u16,
}

#[cfg(target_os = "windows")]
fn read_clipboard_text() -> Option<String> {
    const CF_UNICODETEXT: u32 = 13;

    #[link(name = "user32")]
    unsafe extern "system" {
        #[link_name = "OpenClipboard"]
        fn open_clipboard(hwnd_new_owner: *mut c_void) -> i32;
        #[link_name = "CloseClipboard"]
        fn close_clipboard() -> i32;
        #[link_name = "GetClipboardData"]
        fn get_clipboard_data(format: u32) -> *mut c_void;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        #[link_name = "GlobalLock"]
        fn global_lock(mem: *mut c_void) -> *mut c_void;
        #[link_name = "GlobalUnlock"]
        fn global_unlock(mem: *mut c_void) -> i32;
    }

    struct ClipboardGuard;

    impl Drop for ClipboardGuard {
        fn drop(&mut self) {
            unsafe {
                close_clipboard();
            }
        }
    }

    unsafe {
        if open_clipboard(std::ptr::null_mut()) == 0 {
            return None;
        }
        let _clipboard_guard = ClipboardGuard;

        let handle = get_clipboard_data(CF_UNICODETEXT);
        if handle.is_null() {
            return None;
        }

        let ptr = global_lock(handle) as *const u16;
        if ptr.is_null() {
            return None;
        }

        let mut len = 0;
        while *ptr.add(len) != 0 {
            len += 1;
        }

        let text = String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len));
        global_unlock(handle);
        Some(text)
    }
}

#[cfg(not(target_os = "windows"))]
fn read_clipboard_text() -> Option<String> {
    None
}

#[cfg(target_os = "windows")]
fn write_clipboard_text(text: &str) -> bool {
    const CF_UNICODETEXT: u32 = 13;
    const GMEM_MOVEABLE: u32 = 0x0002;

    #[link(name = "user32")]
    unsafe extern "system" {
        #[link_name = "OpenClipboard"]
        fn open_clipboard(hwnd_new_owner: *mut c_void) -> i32;
        #[link_name = "CloseClipboard"]
        fn close_clipboard() -> i32;
        #[link_name = "EmptyClipboard"]
        fn empty_clipboard() -> i32;
        #[link_name = "SetClipboardData"]
        fn set_clipboard_data(format: u32, mem: *mut c_void) -> *mut c_void;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        #[link_name = "GlobalAlloc"]
        fn global_alloc(flags: u32, bytes: usize) -> *mut c_void;
        #[link_name = "GlobalFree"]
        fn global_free(mem: *mut c_void) -> *mut c_void;
        #[link_name = "GlobalLock"]
        fn global_lock(mem: *mut c_void) -> *mut c_void;
        #[link_name = "GlobalUnlock"]
        fn global_unlock(mem: *mut c_void) -> i32;
    }

    struct ClipboardGuard;

    impl Drop for ClipboardGuard {
        fn drop(&mut self) {
            unsafe {
                close_clipboard();
            }
        }
    }

    let wide = text
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let byte_len = wide.len() * std::mem::size_of::<u16>();

    unsafe {
        if open_clipboard(std::ptr::null_mut()) == 0 {
            return false;
        }
        let _clipboard_guard = ClipboardGuard;

        if empty_clipboard() == 0 {
            return false;
        }

        let handle = global_alloc(GMEM_MOVEABLE, byte_len);
        if handle.is_null() {
            return false;
        }

        let ptr = global_lock(handle) as *mut u16;
        if ptr.is_null() {
            global_free(handle);
            return false;
        }

        std::ptr::copy_nonoverlapping(wide.as_ptr(), ptr, wide.len());
        global_unlock(handle);

        if set_clipboard_data(CF_UNICODETEXT, handle).is_null() {
            global_free(handle);
            return false;
        }

        true
    }
}

#[cfg(not(target_os = "windows"))]
fn write_clipboard_text(_text: &str) -> bool {
    false
}





/// Parses command-line arguments and either:
///
/// - **Dedicated server mode** (`--server`): blocks the calling thread on a
///   Tokio runtime that runs [`run_dedicated_server`] until the process is
///   killed.  No window is created.
///
/// - **Client / game mode** (default): creates a winit window, initializes
///   the wgpu [`State`] synchronously via `pollster::block_on`, then drives
///   the winit event loop until the window is closed.
///
/// # Event loop overview
///
/// | Event | Action |
/// |---|---|
/// | `Resized` | Rebuilds all resolution-dependent GPU resources. |
/// | `RedrawRequested` | Measures frame time, runs `update`, calls `render`. |
/// | `KeyboardInput` | Dispatches to menu or in-game key handlers (see below). |
/// | `MouseWheel` | Scrolls the hotbar slot selection. |
/// | `MouseInput` | Captures cursor on first in-game click; dispatches block actions. |
/// | `CursorMoved` | Tracks cursor position for menu hover/hit-testing. |
/// | `DeviceEvent::MouseMotion` | Rotates the camera when the cursor is captured. |
/// | `AboutToWait` | Switches to 30 fps throttle after 30 s of inactivity. |
/// | `CloseRequested` | Exits the event loop cleanly. |
///
/// # Key bindings (in-game)
///
/// | Key | Action |
/// |---|---|
/// | W / A / S / D | Move forward / left / backward / right. |
/// | Space | Jump. |
/// | Left Shift | Sprint. |
/// | Left Ctrl | Crouch. |
/// | 1–9 | Select hotbar slot. |
/// | Escape (mouse captured) | Release cursor without leaving the game. |
/// | Escape (mouse free) | Open the main menu. |
/// | F1 | Toggle crosshair and hotbar. |
/// | F5 | Save world to disk. |
/// | F3 | Toggle FPS/chunk debug overlay. |
/// | F9 | Load world from disk. |
/// | F11 | Toggle borderless fullscreen. |
///
/// # Key bindings (menu)
///
/// | Key | Action |
/// |---|---|
/// | Tab | Move focus to the next text field. |
/// | Enter | Attempt to connect to the server. |
/// | Escape | Resume play after entering a world; no-op on the initial menu. |
/// | Backspace | Delete the last character in the active field. |
/// | Ctrl+A | Select all text in the active field. |
/// | Ctrl+C | Copy selected text from the active field. |
/// | Ctrl+V | Paste clipboard text into the active field. |
/// | F11 | Toggle borderless fullscreen. |
/// | Any printable character | Appended to the active text field. |
///
/// # Panics
/// Panics if the winit event loop or window cannot be created, or if the
/// Tokio runtime for the server cannot be initialized.
pub fn run_game() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    
    if args.server {
        let addr = format!("0.0.0.0:{}", args.port);
        log(
            LogLevel::Info,
            &format!("Starting headless server on {}...", addr),
        );
        log(
            LogLevel::Info,
            "Note: This is a console-only server. No game window will appear.",
        );
        log(
            LogLevel::Info,
            "To play the game, run the application without --server.",
        );
        log(LogLevel::Info, "Press Ctrl+C to stop the server.");
        
        
        use std::io::Write;
        let _ = std::io::stdout().flush();

        
        
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(run_dedicated_server(&addr));
        return Ok(());
    }

    

    
    
    
    let event_loop = match EventLoop::new() {
        
        
        Ok(ev) => ev,
        Err(e) => {
            log(
                LogLevel::Error,
                &format!("Failed to create event loop: {}", e),
            );
            return Err(Box::new(e) as Box<dyn std::error::Error>);
        }
    };

    
    
    let window_attributes = Window::default_attributes()
        .with_title("crust")
        .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        .with_transparent(true)
        
        .with_fullscreen(Some(Fullscreen::Borderless(None)));

    let window = match event_loop.create_window(window_attributes)
    {
        Ok(w) => w,
        Err(e) => {
            log(LogLevel::Error, &format!("Failed to create window: {}", e));
            return Err(Box::new(e) as Box<dyn std::error::Error>);
        }
    };

    
    
    
    let mut state = pollster::block_on(State::new(window));

    event_loop
        .run(move |event, elwt| {
            
            
            
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    state.resize(size);
                    state.window.request_redraw();
                }

                
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    let now = Instant::now();
                    
                    
                    state.frame_time_ms =
                        now.duration_since(state.last_redraw).as_secs_f32() * 1000.0;
                    state.last_redraw = now;
                    state.frame_count += 1;

                    
                    let elapsed = now.duration_since(state.last_fps_update).as_secs_f32();
                    if elapsed >= 0.5 {
                        state.current_fps = state.frame_count as f32 / elapsed;
                        state.frame_count = 0;
                        state.last_fps_update = now;
                    }

                    
                    
                    
                    let frame_cpu_start = process_cpu_time_ms();
                    let frame_start = Instant::now();

                    
                    state.update();

                    match state.render() {
                        Ok(_) => {}
                        
                        
                        
                        Err(RenderError::Surface(wgpu::CurrentSurfaceTexture::Lost))
                        | Err(RenderError::Surface(wgpu::CurrentSurfaceTexture::Outdated)) => {
                            state.resize(state.window.inner_size())
                        }
                        
                        
                        Err(RenderError::Surface(wgpu::CurrentSurfaceTexture::Validation)) => {
                            elwt.exit()
                        }
                        Err(e) => log(LogLevel::Error, &format!("Render error: {:?}", e)),
                    }

                    
                    
                    
                    let sample_end = Instant::now();
                    if let Some(cpu_end) = process_cpu_time_ms() {
                        state.cpu_time_sample_frames += 1;
                        match state.cpu_time_sample_start {
                            Some((cpu_start, sample_start))
                                if sample_end.duration_since(sample_start).as_millis() >= 250 =>
                            {
                                state.cpu_update_ms = (cpu_end - cpu_start).max(0.0) as f32
                                    / state.cpu_time_sample_frames as f32;
                                state.cpu_time_sample_start = Some((cpu_end, sample_end));
                                state.cpu_time_sample_frames = 0;
                            }
                            Some(_) => {}
                            None => {
                                state.cpu_time_sample_start = Some((cpu_end, sample_end));
                                state.cpu_time_sample_frames = 0;
                            }
                        }
                    } else {
                        
                        state.cpu_update_ms = frame_cpu_start
                            .map(|start| (process_cpu_time_ms().unwrap_or(start) - start) as f32)
                            .unwrap_or_else(|| frame_start.elapsed().as_secs_f32() * 1000.0);
                    }

                    
                    state.window.request_redraw();
                }

                
                Event::WindowEvent {
                    event: WindowEvent::ModifiersChanged(modifiers),
                    ..
                } => {
                    state.modifiers = modifiers.state();
                }

                Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    physical_key: PhysicalKey::Code(key),
                                    state: key_state,
                                    repeat,
                                    text,
                                    ..
                                },
                            ..
                        },
                    ..
                } => {
                    state.last_input_time = Instant::now();
                    let pressed = key_state == ElementState::Pressed;
                    let menu_control_shortcut = state.game_state == GameState::Menu
                        && pressed
                        && state.modifiers.control_key();
                    let menu_edit_shortcut = menu_control_shortcut
                        && matches!(key, KeyCode::KeyA | KeyCode::KeyC | KeyCode::KeyV);
                    let menu_edit_shortcut_once = menu_edit_shortcut && !repeat;

                    
                    
                    
                    
                    
                    if state.game_state == GameState::Menu
                        && pressed
                        && !menu_edit_shortcut
                        && state.menu_state.is_editing()
                    {
                        if let Some(ref txt) = text {
                            for ch in txt.chars() {
                                state.menu_state.handle_char(ch);
                            }
                        }
                    }

                    if state.game_state == GameState::Menu {
                        
                        if pressed {
                            match key {
                                KeyCode::KeyA if menu_edit_shortcut_once => {
                                    state.menu_state.select_all_current_field();
                                }
                                KeyCode::KeyC if menu_edit_shortcut_once => {
                                    if let Some(text) = state.menu_state.selected_text() {
                                        write_clipboard_text(text);
                                    }
                                }
                                KeyCode::KeyV if menu_edit_shortcut_once => {
                                    if let Some(text) = read_clipboard_text() {
                                        state.menu_state.handle_paste(&text);
                                    }
                                }
                                KeyCode::Enter => {
                                    state.connect_to_server();
                                }
                                KeyCode::Escape => {
                                    if state.has_entered_world {
                                        
                                        
                                        
                                        state.game_state = GameState::Playing;
                                        state.mouse_captured = true;
                                        let _ = state
                                            .window
                                            .set_cursor_grab(CursorGrabMode::Confined)
                                            .or_else(|_| {
                                                state.window.set_cursor_grab(CursorGrabMode::Locked)
                                            });
                                        state.window.set_cursor_visible(false);
                                    }
                                }
                                KeyCode::Backspace => {
                                    state.menu_state.handle_backspace();
                                }
                                KeyCode::F11 => {
                                    
                                    
                                    
                                    if state.window.fullscreen().is_some() {
                                        state.window.set_fullscreen(None);
                                    } else {
                                        state.window.set_fullscreen(Some(
                                            winit::window::Fullscreen::Borderless(None),
                                        ));
                                    }
                                }
                                _ => {}
                            }
                        }
                    } else {
                        
                        match key {
                            
                            KeyCode::KeyW => state.input.forward = pressed,
                            KeyCode::KeyS => state.input.backward = pressed,
                            KeyCode::KeyA => state.input.left = pressed,
                            KeyCode::KeyD => state.input.right = pressed,
                            KeyCode::Space => {
                                state.input.jump_held = pressed;
                                if pressed && !repeat {
                                    state.input.jump = true;
                                }
                            }
                            KeyCode::ShiftLeft => state.input.sprint = pressed,
                            KeyCode::ControlLeft => state.input.sneak = pressed,

                            KeyCode::Escape if pressed => {
                                
                                
                                
                                state.game_state = GameState::Menu;
                                state.mouse_captured = false;
                                state.input = Default::default();
                                state.digging = Default::default();
                                let _ = state.window.set_cursor_grab(CursorGrabMode::None);
                                state.window.set_cursor_visible(true);
                            }

                            KeyCode::F11 if pressed => {
                                if state.window.fullscreen().is_some() {
                                    state.window.set_fullscreen(None);
                                } else {
                                    state.window.set_fullscreen(Some(
                                        winit::window::Fullscreen::Borderless(None),
                                    ));
                                }
                            }

                            KeyCode::F1 if pressed && !repeat => {
                                state.show_crosshair = !state.show_crosshair;
                                let state_name = if state.show_crosshair { "On" } else { "Off" };
                                log(LogLevel::Info, &format!("HUD: {}", state_name));
                            }

                            KeyCode::F3 if pressed && !repeat => {
                                state.show_debug_overlay = !state.show_debug_overlay;
                                let state_name = if state.show_debug_overlay {
                                    "On"
                                } else {
                                    "Off"
                                };
                                log(LogLevel::Info, &format!("Debug overlay: {}", state_name));
                            }

                            
                            KeyCode::F5 if pressed => {
                                let world = state.world.read();
                                
                                
                                
                                
                                let saved = SavedWorld::from_world(
                                    &world.chunks,
                                    world.seed,
                                    (
                                        state.camera.position.x,
                                        state.camera.position.y,
                                        state.camera.position.z,
                                    ),
                                    (state.camera.yaw, state.camera.pitch),
                                );
                                if let Err(e) = save_world(DEFAULT_WORLD_FILE, &saved) {
                                    log(LogLevel::Error, &format!("Failed to save world: {}", e));
                                } else {
                                    log(
                                        LogLevel::Info,
                                        &format!("World saved to {}", DEFAULT_WORLD_FILE),
                                    );
                                }
                            }

                            
                            KeyCode::F9 if pressed => match load_world(DEFAULT_WORLD_FILE) {
                                Ok(saved) => {
                                    log(
                                        LogLevel::Info,
                                        &format!("Regenerating world with seed {}...", saved.seed),
                                    );

                                    
                                    
                                    
                                    
                                    {
                                        let mut world = state.world.write();
                                        *world = World::new_with_seed(saved.seed);
                                    }

                                    
                                    
                                    
                                    state.indirect_manager.clear();
                                    state.water_indirect_manager.clear();

                                    
                                    state.camera.position.x = saved.player_x;
                                    state.camera.position.y = saved.player_y;
                                    state.camera.position.z = saved.player_z;
                                    state.camera.yaw = saved.player_yaw;
                                    state.camera.pitch = saved.player_pitch;

                                    
                                    
                                    
                                    
                                    
                                    {
                                        let mut world = state.world.write();
                                        for chunk_data in &saved.chunks {
                                            let cx = chunk_data.cx;
                                            let cz = chunk_data.cz;
                                            if let Some(chunk) = world.chunks.get_mut(&(cx, cz)) {
                                                for (&sy, block_data) in &chunk_data.subchunks {
                                                    if (sy as usize) < chunk.subchunks.len() {
                                                        let subchunk =
                                                            &mut chunk.subchunks[sy as usize];
                                                        
                                                        
                                                        let mut n = 0;
                                                        for lx in 0..CHUNK_SIZE as usize {
                                                            for ly in 0..SUBCHUNK_HEIGHT as usize {
                                                                for lz in 0..CHUNK_SIZE as usize {
                                                                    if n < block_data.len() {
                                                                        subchunk.set_block_raw(
                                                                            lx as i32,
                                                                            ly as i32,
                                                                            lz as i32,
                                                                            block_data[n],
                                                                        );
                                                                        n += 1;
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        subchunk.is_empty = false;
                                                        subchunk.mark_mesh_dirty();
                                                    }
                                                }
                                                chunk.player_modified = true;
                                                chunk.rebuild_metadata();
                                            }
                                        }
                                    }

                                    
                                    
                                    
                                    {
                                        let mut world = state.world.write();
                                        for chunk in world.chunks.values_mut() {
                                            for subchunk in &mut chunk.subchunks {
                                                subchunk.mark_mesh_dirty();
                                            }
                                        }
                                    }
                                    state.enqueue_all_dirty_meshes();
                                    log(
                                        LogLevel::Info,
                                        &format!(
                                            "World loaded from {} (seed: {})",
                                            DEFAULT_WORLD_FILE, saved.seed
                                        ),
                                    );
                                }
                                Err(e) => log(LogLevel::Error, &format!("Error loading: {}", e)),
                            },

                            
                            
                            
                            KeyCode::Digit1 if pressed => {
                                state.hotbar_slot = 0;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit2 if pressed => {
                                state.hotbar_slot = 1;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit3 if pressed => {
                                state.hotbar_slot = 2;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit4 if pressed => {
                                state.hotbar_slot = 3;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit5 if pressed => {
                                state.hotbar_slot = 4;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit6 if pressed => {
                                state.hotbar_slot = 5;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit7 if pressed => {
                                state.hotbar_slot = 6;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit8 if pressed => {
                                state.hotbar_slot = 7;
                                state.hotbar_dirty = true;
                            }
                            KeyCode::Digit9 if pressed => {
                                state.hotbar_slot = 8;
                                state.hotbar_dirty = true;
                            }
                            _ => {}
                        }
                    }
                }

                Event::WindowEvent {
                    event: WindowEvent::MouseWheel { delta, .. },
                    ..
                } => {
                    if state.game_state != GameState::Menu {
                        
                        
                        
                        
                        let scroll = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                        };
                        let slots = crate::ui::ui::HOTBAR_SLOTS.len() as i32;
                        
                        
                        let new_slot = (state.hotbar_slot as i32 - scroll.signum() as i32)
                            .rem_euclid(slots) as usize;
                        if new_slot != state.hotbar_slot {
                            state.hotbar_slot = new_slot;
                            state.hotbar_dirty = true;
                        }
                    }
                }

                
                Event::WindowEvent {
                    event:
                        WindowEvent::MouseInput {
                            state: btn_state,
                            button,
                            ..
                        },
                    ..
                } => {
                    state.last_input_time = Instant::now();
                    let pressed = btn_state == ElementState::Pressed;

                    if state.game_state == GameState::Menu {
                        
                        
                        
                        if pressed && button == winit::event::MouseButton::Left {
                            if let Some((x, y)) = state.cursor_position {
                                state.handle_menu_click(x, y);
                            }
                        }
                    } else if pressed && !state.mouse_captured {
                        
                        
                        
                        
                        
                        state.mouse_captured = true;
                        let _ = state
                            .window
                            .set_cursor_grab(CursorGrabMode::Confined)
                            .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked));
                        state.window.set_cursor_visible(false);
                        
                        
                        let _ = state.window.set_cursor_position(PhysicalPosition::new(
                            state.config.width / 2,
                            state.config.height / 2,
                        ));
                    } else {
                        
                        
                        state.handle_mouse_input(button, pressed);
                    }
                }

                
                
                
                Event::WindowEvent {
                    event: WindowEvent::CursorMoved { position, .. },
                    ..
                } => {
                    state.cursor_position = Some((position.x as f32, position.y as f32));
                }

                
                
                
                
                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { delta },
                    ..
                } => {
                    state.last_input_time = Instant::now();
                    if state.mouse_captured {
                        let sensitivity = 0.002; 
                        state.camera.yaw += delta.0 as f32 * sensitivity;
                        
                        
                        
                        state.camera.pitch -= delta.1 as f32 * sensitivity;
                        
                        
                        
                        state.camera.pitch = state.camera.pitch.clamp(
                            -std::f32::consts::FRAC_PI_2 + 0.1,
                            std::f32::consts::FRAC_PI_2 - 0.1,
                        );
                    }
                }

                
                
                
                
                
                
                
                Event::AboutToWait => {
                    let is_idle = state.last_input_time.elapsed().as_secs() >= 30;
                    if is_idle {
                        let next_frame = Instant::now() + std::time::Duration::from_millis(33);
                        elwt.set_control_flow(ControlFlow::WaitUntil(next_frame));
                    } else {
                        elwt.set_control_flow(ControlFlow::Poll);
                    }
                    state.window.request_redraw();
                }

                
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => elwt.exit(),

                _ => {}
            }
        })
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    Ok(())
}
