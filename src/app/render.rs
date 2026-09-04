use std::time::{Duration, Instant};

use glam::{Mat4, Vec3};
use glyphon::{Color, Metrics, Shaping, TextArea, TextBounds};
use wgpu::util::DeviceExt;

use crust::{
    BlockType, CHUNK_SIZE, DEFAULT_FOV, MAX_MESH_BUILDS_PER_FRAME, RENDER_DISTANCE, SEA_LEVEL,
    SUN_MOVEMENT_SPEED, Uniforms, Vertex, World, build_block_outline, build_item_model,
    build_player_model, extract_frustum_planes,
};

use crate::logger::{LogLevel, log};
use crate::multiplayer::player::queue_remote_players_labels;
use crate::ui::menu::{GameState, MenuField, MenuHit, MenuLayout};

use super::init::frustum_planes_to_array;
use super::state::State;

#[derive(Debug)]
pub enum RenderError {
    Surface(wgpu::CurrentSurfaceTexture),
    Text,
}

/// Computes which faces of the highlighted block should be outlined.
///
/// The outline follows the same face-visibility rules as block meshing so the
/// overlay only draws exposed faces.
fn visible_outline_faces(world: &World, bx: i32, by: i32, bz: i32) -> [bool; 6] {
    let block = world.get_block(bx, by, bz);
    if block == BlockType::Air {
        return [false; 6];
    }

    [
        block.should_render_face_against(world.get_block(bx + 1, by, bz)),
        block.should_render_face_against(world.get_block(bx - 1, by, bz)),
        block.should_render_face_against(world.get_block(bx, by + 1, bz)),
        block.should_render_face_against(world.get_block(bx, by - 1, bz)),
        block.should_render_face_against(world.get_block(bx, by, bz + 1)),
        block.should_render_face_against(world.get_block(bx, by, bz - 1)),
    ]
}

fn append_ui_quad(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [f32; 4],
) {
    let base = vertices.len() as u32;
    let normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

    for (i, (x, y)) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        .into_iter()
        .enumerate()
    {
        vertices.push(Vertex {
            position: [x, y, 0.0],
            packed: Vertex::pack_ui(normal, color, 0, i as u8),
        });
    }

    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn build_menu_input_box(
    rect: crate::ui::menu::Rect,
    screen_width: f32,
    screen_height: f32,
    focused: bool,
    selected: bool,
) -> (Vec<Vertex>, Vec<u32>) {
    let x0 = rect.x / screen_width * 2.0 - 1.0;
    let x1 = (rect.x + rect.w) / screen_width * 2.0 - 1.0;
    let y0 = 1.0 - (rect.y + rect.h) / screen_height * 2.0;
    let y1 = 1.0 - rect.y / screen_height * 2.0;

    let border_x = 4.0 / screen_width;
    let border_y = 4.0 / screen_height;
    let border_color = if focused {
        [0.78, 0.92, 1.0, 0.95]
    } else {
        [0.38, 0.50, 0.58, 0.85]
    };
    let inner_color = if selected {
        [0.20, 0.38, 0.60, 0.86]
    } else {
        [0.10, 0.14, 0.18, 0.78]
    };

    let mut vertices = Vec::with_capacity(8);
    let mut indices = Vec::with_capacity(12);
    append_ui_quad(&mut vertices, &mut indices, x0, y0, x1, y1, border_color);
    append_ui_quad(
        &mut vertices,
        &mut indices,
        x0 + border_x,
        y0 + border_y,
        x1 - border_x,
        y1 - border_y,
        inner_color,
    );

    (vertices, indices)
}

impl State {
    fn rebuild_visible_chunk_cache(&mut self, player_cx: i32, player_cz: i32) {
        if !self.visible_chunk_columns_dirty
            && self.visible_chunk_cache_center == (player_cx, player_cz)
        {
            return;
        }

        self.visible_chunk_columns.clear();
        {
            let world = self.world.read();
            for cx in (player_cx - RENDER_DISTANCE)..=(player_cx + RENDER_DISTANCE) {
                for cz in (player_cz - RENDER_DISTANCE)..=(player_cz + RENDER_DISTANCE) {
                    if world.chunks.contains_key(&(cx, cz)) {
                        self.visible_chunk_columns.push((cx, cz));
                    }
                }
            }
        }

        self.visible_chunk_columns.sort_by_key(|&(cx, cz)| {
            let dx = cx - player_cx;
            let dz = cz - player_cz;
            dx * dx + dz * dz
        });
        self.visible_chunk_cache_center = (player_cx, player_cz);
        self.visible_chunk_columns_dirty = false;
    }

    /// Produces one complete frame and presents it to the OS window.
    ///
    /// # Render pipeline overview
    ///
    /// The frame is built from the following render / compute passes in order:
    ///
    /// 1. **Player model update** – re-builds the combined vertex/index buffers
    ///    for all visible remote players if any exist.
    /// 2. **Uniform upload** – computes the camera matrices, advances the day
    ///    cycle, and uploads the `Uniforms` struct.
    /// 3. **Mesh request** – drains a bounded dirty-subchunk queue into the
    ///    background mesher.
    /// 4. **Main cull dispatch** – GPU frustum + Hi-Z occlusion cull for both
    ///    the opaque terrain and water indirect managers.
    /// 5. **Opaque pass** – sky dome -> terrain -> remote player models -> sun/moon.
    ///    Resolves MSAA into `ssr_color_view` for later water reflections.
    /// 6. **Depth resolve compute** – resolves the multisampled depth buffer
    ///    into `ssr_depth_view` (for water refraction) and the first Hi-Z mip
    ///    level (for next-frame occlusion culling).
    /// 7. **Hi-Z generation** (compute) – downsamples the depth mip chain.
    /// 8. **Transparent pass** – water surfaces, alpha-blended on top of the
    ///    opaque result.  Resolves MSAA into `scene_color_view`.
    /// 9. **Composite pass** – post-processing blit from `scene_color_view`
    ///     to the swap-chain surface (underwater fog, vignette, etc.).
    /// 10. **UI pass** – crosshair, coordinate debug overlay, hotbar.
    /// 11. **Progress bar pass** – block-breaking progress indicator (only
    ///     when the player is actively mining).
    /// 12. **Menu / HUD** – either the main-menu overlay or remote-player
    ///     name labels, depending on `game_state`.
    /// 13. **Text pass** – all `glyphon` text areas (FPS counter, menu
    ///     labels, hotbar slot name, player name tags).
    /// 14. **Submit** – the completed command buffer is submitted and the
    ///     swap-chain texture is presented.
    ///
    /// # Errors
    /// Returns an error when the swap-chain texture cannot
    /// be acquired (e.g., the window is minimized or the surface is lost).
    /// The caller should handle `Lost` / `Outdated` by calling `resize`.
    pub fn render(&mut self) -> Result<(), RenderError> {
        
        
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(profiler) = &mut self.gpu_timestamp_profiler {
            if let Some(profile) = profiler.poll() {
                self.gpu_frame_profile = Some(profile);
            }
        }

        
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            status => return Err(RenderError::Surface(status)),
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let menu_visible = self.game_state != GameState::Playing;
        let world_has_loaded_chunks = menu_visible && !self.world.read().chunks.is_empty();
        let menu_uses_world_background =
            menu_visible && (self.has_entered_world || world_has_loaded_chunks);
        let render_world_scene = !menu_visible || menu_uses_world_background;
        let preparation_start = Instant::now();

        
        
        
        
        if (!self.remote_players.is_empty() || !self.item_entities.is_empty()) && render_world_scene {
            let mut all_vertices = Vec::with_capacity(
                self.remote_players.len() * 16 + self.item_entities.len() * 24,
            );
            let mut all_indices = Vec::with_capacity(
                self.remote_players.len() * 24 + self.item_entities.len() * 36,
            );

            for (_id, player) in &self.remote_players {
                let (vertices, indices) =
                    build_player_model(player.x, player.y, player.z, player.yaw);
                let base_idx = all_vertices.len() as u32;
                all_vertices.extend(vertices);
                
                all_indices.extend(indices.iter().map(|i| i + base_idx));
            }

            let elapsed = self.game_start_time.elapsed().as_secs_f32();
            for item in &self.item_entities {
                let Some(block) = crust::block_for_item(item.item_id) else {
                    continue;
                };
                let bob = (elapsed * 3.0 + item.id as f32 * 0.71).sin() * 0.06;
                let (vertices, indices) = build_item_model(
                    [item.position.x, item.position.y + bob, item.position.z],
                    block,
                    elapsed * 1.8 + item.id as f32 * 0.37,
                );
                let base_idx = all_vertices.len() as u32;
                all_vertices.extend(vertices);
                all_indices.extend(indices.iter().map(|i| i + base_idx));
            }

            self.player_model_num_indices = all_indices.len() as u32;

            if !all_vertices.is_empty() {
                let needed_verts = all_vertices.len() as u32;
                let needed_idxs = all_indices.len() as u32;

                
                
                if needed_verts > self.player_model_vertex_capacity
                    || self.player_model_vertex_buffer.is_none()
                {
                    let new_cap = (needed_verts * 2).max(256);
                    self.player_model_vertex_buffer =
                        Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Player Model Vertex Buffer"),
                            size: (new_cap as usize * size_of::<Vertex>()) as u64,
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }));
                    self.player_model_vertex_capacity = new_cap;
                }
                
                if needed_idxs > self.player_model_index_capacity
                    || self.player_model_index_buffer.is_none()
                {
                    let new_cap = (needed_idxs * 2).max(512);
                    self.player_model_index_buffer =
                        Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Player Model Index Buffer"),
                            size: (new_cap as usize * size_of::<u32>()) as u64,
                            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        }));
                    self.player_model_index_capacity = new_cap;
                }

                self.queue.write_buffer(
                    self.player_model_vertex_buffer
                        .as_ref()
                        .expect("Player model vertex buffer should be initialized"),
                    0,
                    bytemuck::cast_slice(&all_vertices),
                );
                self.queue.write_buffer(
                    self.player_model_index_buffer
                        .as_ref()
                        .expect("Player model index buffer should be initialized"),
                    0,
                    bytemuck::cast_slice(&all_indices),
                );
            }
        } else {
            
            self.player_model_num_indices = 0;
        }
        self.frame_profile.remote_players_ms = preparation_start.elapsed().as_secs_f32() * 1000.0;

        
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        
        
        let timestamp_query_set = self
            .gpu_timestamp_profiler
            .as_ref()
            .map(|p| p.query_set.clone());
        let timestamp_readback_slot = if render_world_scene {
            self.gpu_timestamp_profiler
                .as_ref()
                .and_then(|profiler| profiler.acquire_readback())
        } else {
            None
        };

        
        let section_start = Instant::now();
        let aspect = self.config.width as f32 / self.config.height as f32;
        
        
        let far_plane = (RENDER_DISTANCE as f32 * CHUNK_SIZE as f32 * 1.5).max(400.0);
        let proj = Mat4::perspective_rh(DEFAULT_FOV, aspect, 0.1, far_plane);
        let view_mat = self.camera.view_matrix();
        
        let view_proj = proj * view_mat;
        let view_proj_array: [[f32; 4]; 4] = view_proj.to_cols_array_2d();
        self.frame_profile.camera_matrices_ms = section_start.elapsed().as_secs_f32() * 1000.0;

        
        let time = self.game_start_time.elapsed().as_secs_f32();

        
        
        let day_cycle_speed = SUN_MOVEMENT_SPEED;
        
        
        let sun_angle = time * day_cycle_speed + std::f32::consts::FRAC_PI_2;
        let sun_x = 0.0;
        let sun_y = sun_angle.sin(); 
        let sun_z = sun_angle.cos();
        let sun_dir = Vec3::new(sun_x, sun_y, sun_z).normalize();
        let moon_intensity = (-sun_dir.y).clamp(0.0, 1.0);

        
        let moon_position = [-sun_dir.x, -sun_dir.y, -sun_dir.z];

        
        let player_cx = (self.camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (self.camera.position.z / CHUNK_SIZE as f32).floor() as i32;
        let section_start = Instant::now();
        self.rebuild_visible_chunk_cache(player_cx, player_cz);
        self.frame_profile.visible_cache_ms = section_start.elapsed().as_secs_f32() * 1000.0;

        
        
        let inv_view_proj = view_proj.inverse();
        let inv_view_proj_array: [[f32; 4]; 4] = inv_view_proj.to_cols_array_2d();

        let eye_pos = self.camera.eye_position();
        let is_underwater = self.is_underwater;

        
        let section_start = Instant::now();
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                view_proj: view_proj_array,
                inv_view_proj: inv_view_proj_array,
                camera_pos: eye_pos.to_array(),
                time,
                sun_position: [sun_x, sun_y, sun_z],
                is_underwater,
                screen_size: [self.config.width as f32, self.config.height as f32],
                water_level: SEA_LEVEL as f32 - 1.0,
                reflection_mode: self.reflection_mode as f32,
                moon_position,
                _pad1_moon: 0.0,
                moon_intensity,
                wind_dir: [0.8, 0.6],
                wind_speed: 1.0,
                rain_factor: 0.0,
                sky_visibility: self.sky_visibility,
                menu_blur: if menu_visible { 1.0 } else { 0.0 },
                _pad_uniforms: 0.0,
            }]),
        );
        self.frame_profile.uniform_upload_ms = section_start.elapsed().as_secs_f32() * 1000.0;

        
        
        
        let frustum_planes = extract_frustum_planes(&view_proj);

        
        
        
        
        let section_start = Instant::now();
        for _ in 0..MAX_MESH_BUILDS_PER_FRAME {
            let Some(key) = self.pop_dirty_mesh() else {
                break;
            };
            
            
            
            let still_dirty = self
                .world
                .read()
                .chunks
                .get(&(key.chunk_x, key.chunk_z))
                .and_then(|chunk| chunk.subchunks.get(key.subchunk_y as usize))
                .is_some_and(|subchunk| subchunk.mesh_dirty && !subchunk.is_empty);
            if !still_dirty {
                continue;
            }
            if !self
                .mesh_loader
                .request_mesh(key.chunk_x, key.chunk_z, key.subchunk_y)
            {
                
                
                self.requeue_dirty_mesh_front(key);
                break;
            }
        }
        self.frame_profile.render_chunk_scan_ms = section_start.elapsed().as_secs_f32() * 1000.0;
        self.frame_profile.mesh_request_submit_ms = 0.0;
        self.frame_profile.frame_preparation_ms =
            preparation_start.elapsed().as_secs_f32() * 1000.0;

        
        
        
        let day_factor = sun_dir.y.max(0.0).min(1.0); 
        let night_factor = (-sun_dir.y).max(0.0).min(1.0); 
        let sunset_factor = 1.0 - sun_dir.y.abs(); 

        let day_sky = (0.53, 0.81, 0.98); 
        let sunset_sky = (1.0, 0.5, 0.2); 
        let night_sky = (0.001, 0.001, 0.005); 

        let sky_r: f32 = (day_sky.0 * day_factor
            + sunset_sky.0 * sunset_factor * 0.5
            + night_sky.0 * night_factor)
            .min(1.0);
        let sky_g: f32 = (day_sky.1 * day_factor
            + sunset_sky.1 * sunset_factor * 0.5
            + night_sky.1 * night_factor)
            .min(1.0);
        let sky_b: f32 = (day_sky.2 * day_factor
            + sunset_sky.2 * sunset_factor * 0.5
            + night_sky.2 * night_factor)
            .min(1.0);

        
        
        
        let frustum_planes_array = frustum_planes_to_array(&frustum_planes);
        let hiz_size_f = [self.hiz_size[0] as f32, self.hiz_size[1] as f32];

        self.indirect_manager.dispatch_culling(
            &mut encoder,
            &self.queue,
            timestamp_query_set.as_ref(),
            0,
            1,
            &self.water_indirect_manager,
            &view_proj,
            &frustum_planes_array,
            self.camera.position.into(),
            hiz_size_f,
            [self.config.width as f32, self.config.height as f32],
        );

        
        
        
        
        {
            let mut opaque_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Opaque Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    
                    
                    resolve_target: Some(&self.ssr_color_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: sky_r as f64,
                            g: sky_g as f64,
                            b: sky_b as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        
                        
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(2),
                        end_of_pass_write_index: Some(3),
                    }
                }),
                ..Default::default()
            });

            
            
            
            opaque_pass.set_pipeline(&self.sky_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);

            
            
            
            opaque_pass.set_pipeline(&self.render_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_bind_group(1, &self.terrain_quad_bind_group, &[]);
            if self.supports_indirect_count {
                opaque_pass.multi_draw_indirect_count(
                    self.indirect_manager.draw_commands(),
                    0,
                    self.indirect_manager.visible_count_buffer(),
                    0,
                    self.indirect_manager.active_count(),
                );
            } else {
                opaque_pass.multi_draw_indirect(
                    self.indirect_manager.draw_commands(),
                    0,
                    self.indirect_manager.active_count(),
                );
            }

            
            
            
            if self.player_model_num_indices > 0 {
                if let (Some(vb), Some(ib)) = (
                    &self.player_model_vertex_buffer,
                    &self.player_model_index_buffer,
                ) {
                    opaque_pass.set_pipeline(&self.player_model_pipeline);
                    opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    opaque_pass.set_vertex_buffer(0, vb.slice(..));
                    opaque_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    opaque_pass.draw_indexed(0..self.player_model_num_indices, 0, 0..1);
                }
            }

            
            
            
            opaque_pass.set_pipeline(&self.sun_pipeline);
            opaque_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            opaque_pass.set_vertex_buffer(0, self.sun_vertex_buffer.slice(..));
            opaque_pass
                .set_index_buffer(self.sun_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            opaque_pass.draw_indexed(0..6, 0, 0..1);
        }

        
        
        
        
        if render_world_scene {
            let mut depth_resolve_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Depth Resolve Compute Pass"),
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(4),
                        end_of_pass_write_index: Some(5),
                    }
                }),
            });
            depth_resolve_pass.set_pipeline(&self.depth_resolve_pipeline);
            depth_resolve_pass.set_bind_group(0, &self.depth_resolve_bind_group, &[]);
            depth_resolve_pass.dispatch_workgroups(
                (self.config.width + 15) / 16,
                (self.config.height + 15) / 16,
                1,
            );
            drop(depth_resolve_pass);

            for i in 0..self.hiz_bind_groups.len() {
                let mut hiz_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Hi-Z Generation Pass Level"),
                    timestamp_writes: timestamp_query_set.as_ref().and_then(|query_set| {
                        if i == 0 || i + 1 == self.hiz_bind_groups.len() {
                            Some(wgpu::ComputePassTimestampWrites {
                                query_set,
                                beginning_of_pass_write_index: (i == 0).then_some(6),
                                end_of_pass_write_index: (i + 1 == self.hiz_bind_groups.len())
                                    .then_some(7),
                            })
                        } else {
                            None
                        }
                    }),
                });
                hiz_pass.set_pipeline(&self.hiz_pipeline);
                hiz_pass.set_bind_group(0, &self.hiz_bind_groups[i], &[]);
                let div = 1 << (i + 1);
                let mip_width = (self.hiz_size[0] / div).max(1);
                let mip_height = (self.hiz_size[1] / div).max(1);
                hiz_pass.dispatch_workgroups((mip_width + 15) / 16, (mip_height + 15) / 16, 1);
            }
        }

        
        
        
        
        {
            let mut transparent_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Transparent Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    resolve_target: Some(&self.scene_color_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, 
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load, 
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(8),
                        end_of_pass_write_index: Some(9),
                    }
                }),
                ..Default::default()
            });

            transparent_pass.set_pipeline(&self.water_pipeline);
            transparent_pass.set_bind_group(0, &self.water_bind_group, &[]);
            transparent_pass.set_bind_group(1, &self.water_quad_bind_group, &[]);
            if self.supports_indirect_count {
                transparent_pass.multi_draw_indirect_count(
                    self.water_indirect_manager.draw_commands(),
                    0,
                    self.water_indirect_manager.visible_count_buffer(),
                    0,
                    self.water_indirect_manager.active_count(),
                );
            } else {
                transparent_pass.multi_draw_indirect(
                    self.water_indirect_manager.draw_commands(),
                    0,
                    self.water_indirect_manager.active_count(),
                );
            }
        }

        
        
        
        
        
        if self.game_state == GameState::Playing {
            let mut outline_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block Outline Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_texture_view,
                    resolve_target: Some(&self.scene_color_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if let Some((bx, by, bz)) = self.highlighted_block {
                let visible_faces = {
                    let world = self.world.read();
                    visible_outline_faces(&*world, bx, by, bz)
                };
                let (outline_vertices, outline_indices) =
                    build_block_outline(bx, by, bz, visible_faces);
                if !outline_vertices.is_empty() && !outline_indices.is_empty() {
                    let outline_vb =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Block Outline VB"),
                                contents: bytemuck::cast_slice(&outline_vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    let outline_ib =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Block Outline IB"),
                                contents: bytemuck::cast_slice(&outline_indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });
                    outline_pass.set_pipeline(&self.outline_pipeline);
                    outline_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    outline_pass.set_vertex_buffer(0, outline_vb.slice(..));
                    outline_pass.set_index_buffer(outline_ib.slice(..), wgpu::IndexFormat::Uint32);
                    outline_pass.draw_indexed(0..outline_indices.len() as u32, 0, 0..1);
                }
            }
        }

        
        
        
        
        
        {
            let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, 
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None, 
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(10),
                        end_of_pass_write_index: Some(11),
                    }
                }),
                ..Default::default()
            });

            composite_pass.set_pipeline(&self.composite_pipeline);
            let composite_bind_group = if menu_visible && !menu_uses_world_background {
                &self.menu_composite_bind_group
            } else {
                &self.composite_bind_group
            };
            composite_pass.set_bind_group(0, composite_bind_group, &[]);
            composite_pass.draw(0..3, 0..1); 
        }

        
        
        
        
        
        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, 
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(12),
                        end_of_pass_write_index: Some(13),
                    }
                }),
                ..Default::default()
            });

            ui_pass.set_pipeline(&self.crosshair_pipeline);
            ui_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

            if menu_visible {
                if self.menu_state.server_address_input_visible {
                    let layout = MenuLayout::new(self.config.width, self.config.height);
                    let focused = self.menu_state.selected_field == MenuField::ServerAddress;
                    let selected = focused && self.menu_state.selected_all;
                    let (vertices, indices) = build_menu_input_box(
                        layout.server_address_input,
                        self.config.width as f32,
                        self.config.height as f32,
                        focused,
                        selected,
                    );
                    let input_vb =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Menu Server Address Input VB"),
                                contents: bytemuck::cast_slice(&vertices),
                                usage: wgpu::BufferUsages::VERTEX,
                            });
                    let input_ib =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Menu Server Address Input IB"),
                                contents: bytemuck::cast_slice(&indices),
                                usage: wgpu::BufferUsages::INDEX,
                            });

                    ui_pass.set_vertex_buffer(0, input_vb.slice(..));
                    ui_pass.set_index_buffer(input_ib.slice(..), wgpu::IndexFormat::Uint32);
                    ui_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
                }
            } else {
                
                if self.inventory_open {
                    let (vb, ib, count) = crate::ui::inventory::build(
                        &self.device,
                        &self.inventory,
                        self.hotbar_slot,
                        self.cursor_position,
                        self.config.width,
                        self.config.height,
                    );
                    ui_pass.set_vertex_buffer(0, vb.slice(..));
                    ui_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    ui_pass.draw_indexed(0..count, 0, 0..1);
                }

                
                if self.show_crosshair && !self.inventory_open {
                    ui_pass.set_vertex_buffer(0, self.crosshair_vertex_buffer.slice(..));
                    ui_pass.set_index_buffer(
                        self.crosshair_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    ui_pass.draw_indexed(0..self.num_crosshair_indices, 0, 0..1);
                }

                
                
                if let (Some(vb), Some(ib)) =
                    (&self.coords_vertex_buffer, &self.coords_index_buffer)
                {
                    if self.coords_num_indices > 0 {
                        ui_pass.set_vertex_buffer(0, vb.slice(..));
                        ui_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        ui_pass.draw_indexed(0..self.coords_num_indices, 0, 0..1);
                    }
                }

                
                if self.show_crosshair {
                    if self.hotbar_dirty || self.hotbar_vertex_buffer.is_none() {
                        let aspect = self.config.width as f32 / self.config.height as f32;
                        let (vb, ib, count) =
                            crate::ui::ui::build_hotbar(&self.device, self.hotbar_slot, aspect);
                        self.hotbar_vertex_buffer = Some(vb);
                        self.hotbar_index_buffer = Some(ib);
                        self.hotbar_num_indices = count;
                        self.hotbar_dirty = false;
                    }
                    if let (Some(vb), Some(ib)) =
                        (&self.hotbar_vertex_buffer, &self.hotbar_index_buffer)
                    {
                        if self.hotbar_num_indices > 0 {
                            ui_pass.set_vertex_buffer(0, vb.slice(..));
                            ui_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            ui_pass.draw_indexed(0..self.hotbar_num_indices, 0, 0..1);
                        }
                    }
                }
            }
        }

        
        
        
        
        if self.digging.target.is_some() && self.digging.break_time > 0.0 {
            let progress = (self.digging.progress / self.digging.break_time).min(1.0);

            
            
            let bar_width = 0.15;
            let bar_height = 0.015;
            let bar_y = -0.05;

            let bg_color = [0.2, 0.2, 0.2];
            
            let prog_color = [1.0 - progress, progress, 0.0];
            let normal_idx = Vertex::pack_normal([0.0, 0.0, 1.0]);

            
            let mut vertices = Vec::with_capacity(8);
            for (i, (x, y)) in [
                (-bar_width, bar_y - bar_height),
                (bar_width, bar_y - bar_height),
                (bar_width, bar_y + bar_height),
                (-bar_width, bar_y + bar_height),
            ]
            .into_iter()
            .enumerate()
            {
                vertices.push(Vertex {
                    position: [x, y, 0.0],
                    packed: Vertex::pack_ui(
                        normal_idx,
                        [bg_color[0], bg_color[1], bg_color[2], 1.0],
                        0,
                        i as u8,
                    ),
                });
            }

            
            
            let prog_width = bar_width * 2.0 * progress - bar_width;
            let fg_corners = [
                (-bar_width + 0.005, bar_y - bar_height + 0.003),
                (prog_width - 0.005, bar_y - bar_height + 0.003),
                (prog_width - 0.005, bar_y + bar_height - 0.003),
                (-bar_width + 0.005, bar_y + bar_height - 0.003),
            ];
            for (i, (x, y)) in fg_corners.into_iter().enumerate() {
                vertices.push(Vertex {
                    position: [x, y, 0.0],
                    packed: Vertex::pack_ui(
                        normal_idx,
                        [prog_color[0], prog_color[1], prog_color[2], 1.0],
                        0,
                        i as u8,
                    ),
                });
            }

            
            let indices: [u32; 12] = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];

            
            
            if self.progress_bar_vertex_buffer.is_none() {
                self.progress_bar_vertex_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Progress Bar VB"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
                
                self.progress_bar_index_buffer = Some(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("Progress Bar IB"),
                        contents: bytemuck::cast_slice(&indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                ));
            } else {
                
                self.queue.write_buffer(
                    self.progress_bar_vertex_buffer
                        .as_ref()
                        .expect("Progress bar vertex buffer should be initialized"),
                    0,
                    bytemuck::cast_slice(&vertices),
                );
            }

            let progress_vb = self
                .progress_bar_vertex_buffer
                .as_ref()
                .expect("Progress bar vertex buffer should be initialized");
            let progress_ib = self
                .progress_bar_index_buffer
                .as_ref()
                .expect("Progress bar index buffer should be initialized");

            let mut progress_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Progress Bar Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            progress_pass.set_pipeline(&self.crosshair_pipeline);
            progress_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            progress_pass.set_vertex_buffer(0, progress_vb.slice(..));
            progress_pass.set_index_buffer(progress_ib.slice(..), wgpu::IndexFormat::Uint32);
            progress_pass.draw_indexed(0..12, 0, 0..1);
        }

        
        if menu_visible {
            self.render_menu(&mut encoder, &view);
        } else {
            
            
            self.render_remote_players(
                &view_proj,
                self.config.width as f32,
                self.config.height as f32,
            );
        }

        
        
        
        
        
        {
            
            
            
            
            
            if !menu_visible
                && self.show_debug_overlay
                && self.last_debug_text_update.elapsed() >= Duration::from_millis(100)
            {
                let profile = &self.frame_profile;
                let update_children_ms = profile.update_children_ms();
                let gpu_text = if let Some(gpu) = self.gpu_frame_profile {
                    format!(
                        "GPU frame: {:.2} ms\n  Cull: {:.2} ms\n  Opaque: {:.2} ms\n  Depth resolve: {:.2} ms\n  Hi-Z: {:.2} ms\n  Water: {:.2} ms\n  Composite: {:.2} ms\n  UI: {:.2} ms\n  Text: {:.2} ms\n\n",
                        gpu.frame_ms,
                        gpu.cull_ms,
                        gpu.opaque_ms,
                        gpu.depth_resolve_ms,
                        gpu.hiz_ms,
                        gpu.water_ms,
                        gpu.composite_ms,
                        gpu.ui_ms,
                        gpu.text_ms,
                    )
                } else if self.gpu_timestamp_profiler.is_some() {
                    "GPU frame: collecting timestamp queries...\n\n".to_owned()
                } else {
                    "GPU frame: unavailable (TIMESTAMP_QUERY unsupported)\n\n".to_owned()
                };
                let fps_text = format!(
                    "FPS: {:.0}\nFrame avg: {:.2} ms\nCPU time avg: {:.2} ms\n\n{}Update: {:.2} ms\n  Network: {:.2}\n  Chunk poll: {:.2}\n  Physics/snapshot: {:.2}\n  Requests: {:.2}\n  Chunk commit: {:.2}\n  Mesh commit: {:.2}\n  Children sum: {:.2}\n  Unaccounted: {:.2}\n\nFrame preparation: {:.2} ms\n  Camera/matrices: {:.2}\n  Visible cache: {:.2}\n  Uniform upload: {:.2}\n  Dirty mesh queue: {:.2}\n  Mesh requests: {:.2}\n  Remote players: {:.2}\n\nChunks: {}\nSubchunks: {}",
                    self.current_fps,
                    self.frame_time_ms,
                    self.cpu_update_ms,
                    gpu_text,
                    profile.update_ms,
                    profile.network_ms,
                    profile.chunk_poll_ms,
                    profile.physics_snapshot_ms,
                    profile.chunk_requests_ms,
                    profile.chunk_commit_ms,
                    profile.mesh_commit_ms,
                    update_children_ms,
                    profile.update_ms - update_children_ms,
                    profile.frame_preparation_ms,
                    profile.camera_matrices_ms,
                    profile.visible_cache_ms,
                    profile.uniform_upload_ms,
                    profile.render_chunk_scan_ms,
                    profile.mesh_request_submit_ms,
                    profile.remote_players_ms,
                    self.chunks_rendered,
                    self.subchunks_rendered
                );
                self.fps_buffer.set_text(
                    &fps_text,
                    &self.ui_font.attrs(),
                    Shaping::Advanced,
                    None,
                );
                self.fps_buffer
                    .shape_until_scroll(&mut self.font_system, false);
                self.fps_buffer.set_size(
                    Some(self.config.width as f32),
                    Some(self.config.height as f32),
                );
                self.last_debug_text_update = std::time::Instant::now();
            }

            
            if !menu_visible && self.last_hotbar_slot != self.hotbar_slot {
                let block = crate::ui::ui::HOTBAR_SLOTS[self.hotbar_slot];
                let label = block.display_name();
                self.hotbar_label_buffer.set_text(
                    label,
                    &self.ui_font.colored(Color::rgb(255, 238, 200)),
                    Shaping::Advanced,
                    None,
                );
                self.hotbar_label_buffer
                    .shape_until_scroll(&mut self.font_system, false);
                self.hotbar_label_buffer.set_size(
                    Some(self.config.width as f32),
                    Some(self.config.height as f32),
                );
                
                
                
                let font_size = 22.0;
                let char_width = font_size * 0.6;
                self.hotbar_label_width = label.chars().count() as f32 * char_width;
                self.last_hotbar_slot = self.hotbar_slot;
            }

            
            
            
            
            let labels = if menu_visible {
                self.prepare_menu_text();
                Vec::new() 
            } else {
                let labels = queue_remote_players_labels(
                    &self.remote_players,
                    &view_proj,
                    self.config.width as f32,
                    self.config.height as f32,
                );
                
                
                while self.player_label_buffers.len() < labels.len() {
                    self.player_label_buffers.push(glyphon::Buffer::new(
                        &mut self.font_system,
                        Metrics::new(24.0, 32.0),
                    ));
                }
                for (i, label) in labels.iter().enumerate() {
                    let buffer = &mut self.player_label_buffers[i];
                    buffer.set_text(
                        &label.username,
                        &self.ui_font.colored(Color::rgb(76, 255, 76)), 
                        Shaping::Advanced,
                        None,
                    );
                    buffer.shape_until_scroll(&mut self.font_system, false);
                    buffer.set_size(
                        Some(self.config.width as f32),
                        Some(self.config.height as f32),
                    );
                }
                labels
            };

            
            
            
            let mut text_areas = Vec::with_capacity(4);

            if !menu_visible && self.show_debug_overlay {
                text_areas.push(TextArea {
                    buffer: &self.fps_buffer,
                    left: 10.0,
                    top: 10.0,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: self.config.width as i32,
                        bottom: self.config.height as i32,
                    },
                    default_color: Color::rgb(255, 255, 255),
                    custom_glyphs: &[],
                });
            }

            if menu_visible {
                let layout = MenuLayout::new(self.config.width, self.config.height);
                let hovered = self
                    .cursor_position
                    .and_then(|(x, y)| layout.hit_test(x, y));
                let new_world_hovered = matches!(hovered, Some(MenuHit::NewWorld));
                let multiplayer_hovered = matches!(hovered, Some(MenuHit::Multiplayer));
                let render_mode_hovered = matches!(hovered, Some(MenuHit::RenderMode));
                let hover_text_color = Color::rgb(255, 255, 255);
                let menu_text_top_offset = 3.0;
                let menu_text_bounds = TextBounds {
                    left: 0,
                    top: 0,
                    right: self.config.width as i32,
                    bottom: self.config.height as i32,
                };
                let new_world_color = if new_world_hovered {
                    hover_text_color
                } else {
                    Color::rgb(238, 241, 236)
                };
                let multiplayer_color = if multiplayer_hovered {
                    hover_text_color
                } else {
                    Color::rgb(211, 226, 238)
                };
                let render_mode_color = if render_mode_hovered {
                    hover_text_color
                } else {
                    Color::rgb(184, 205, 220)
                };

                text_areas.push(TextArea {
                    buffer: &self.menu_render_mode_button_buffer,
                    left: layout.render_mode_text.x,
                    top: layout.render_mode_text.y + menu_text_top_offset,
                    scale: 1.0,
                    bounds: menu_text_bounds,
                    default_color: render_mode_color,
                    custom_glyphs: &[],
                });

                text_areas.push(TextArea {
                    buffer: &self.menu_singleplayer_button_buffer,
                    left: layout.new_world_text.x,
                    top: layout.new_world_text.y + menu_text_top_offset,
                    scale: 1.0,
                    bounds: menu_text_bounds,
                    default_color: new_world_color,
                    custom_glyphs: &[],
                });
                text_areas.push(TextArea {
                    buffer: &self.menu_connect_button_buffer,
                    left: layout.multiplayer_text.x,
                    top: layout.multiplayer_text.y + menu_text_top_offset,
                    scale: 1.0,
                    bounds: menu_text_bounds,
                    default_color: multiplayer_color,
                    custom_glyphs: &[],
                });
                if self.menu_state.server_address_input_visible {
                    let input = layout.server_address_input;
                    let input_focused = self.menu_state.selected_field == MenuField::ServerAddress;
                    let input_text_color = if input_focused {
                        Color::rgb(255, 255, 255)
                    } else {
                        Color::rgb(215, 230, 238)
                    };

                    text_areas.push(TextArea {
                        buffer: &self.menu_server_address_input_buffer,
                        left: input.x + 12.0,
                        top: input.y + 8.0,
                        scale: 1.0,
                        bounds: TextBounds {
                            left: input.x as i32,
                            top: input.y as i32,
                            right: (input.x + input.w) as i32,
                            bottom: (input.y + input.h) as i32,
                        },
                        default_color: input_text_color,
                        custom_glyphs: &[],
                    });
                }
            } else {
                

                if self.show_crosshair {
                    
                    
                    let label_width = self.hotbar_label_width.min(self.config.width as f32);
                    let label_left = (self.config.width as f32 - label_width) * 0.5;
                    
                    let label_top = (self.config.height as f32 - 170.0).max(0.0);
                    text_areas.push(TextArea {
                        buffer: &self.hotbar_label_buffer,
                        left: label_left,
                        top: label_top,
                        scale: 1.0,
                        bounds: TextBounds {
                            left: 0,
                            top: 0,
                            right: self.config.width as i32,
                            bottom: self.config.height as i32,
                        },
                        default_color: Color::rgb(255, 255, 255),
                        custom_glyphs: &[],
                    });
                }

                
                for (i, label) in labels.iter().enumerate() {
                    text_areas.push(TextArea {
                        buffer: &self.player_label_buffers[i],
                        left: label.screen_x,
                        top: label.screen_y,
                        scale: 1.0,
                        bounds: TextBounds {
                            left: 0,
                            top: 0,
                            right: self.config.width as i32,
                            bottom: self.config.height as i32,
                        },
                        default_color: Color::rgb(255, 255, 255),
                        custom_glyphs: &[],
                    });
                }
            }

            
            self.text_renderer
                .prepare(
                    &self.device,
                    &self.queue,
                    &mut self.font_system,
                    &mut self.text_atlas,
                    &self.viewport,
                    text_areas,
                    &mut self.swash_cache,
                )
                .map_err(|e| {
                    log(LogLevel::Error, &format!("Failed to prepare text: {:?}", e));
                    RenderError::Text
                })?;

            
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Text Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                timestamp_writes: timestamp_query_set.as_ref().map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(14),
                        end_of_pass_write_index: Some(15),
                    }
                }),
                ..Default::default()
            });
            self.text_renderer
                .render(&self.text_atlas, &self.viewport, &mut pass)
                .map_err(|e| {
                    log(LogLevel::Error, &format!("Failed to render text: {:?}", e));
                    RenderError::Text
                })?;
        }

        
        if let (Some(profiler), Some(slot)) =
            (&self.gpu_timestamp_profiler, timestamp_readback_slot)
        {
            profiler.encode_resolve(&mut encoder, slot);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        if let (Some(profiler), Some(slot)) =
            (&mut self.gpu_timestamp_profiler, timestamp_readback_slot)
        {
            profiler.request_readback(slot);
        }
        self.queue.present(output);
        Ok(())
    }

    /// Updates the main-menu labels and the optional server-address input.
    pub fn prepare_menu_text(&mut self) {
        let layout = MenuLayout::new(self.config.width, self.config.height);

        self.menu_connect_button_buffer.set_text(
            "MULTIPLAYER",
            &self.ui_font.attrs(),
            Shaping::Advanced,
            None,
        );
        self.menu_connect_button_buffer
            .shape_until_scroll(&mut self.font_system, false);
        self.menu_connect_button_buffer.set_size(
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        self.menu_singleplayer_button_buffer.set_text(
            "NEW WORLD",
            &self.ui_font.attrs(),
            Shaping::Advanced,
            None,
        );
        self.menu_singleplayer_button_buffer
            .shape_until_scroll(&mut self.font_system, false);
        self.menu_singleplayer_button_buffer.set_size(
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        let render_mode_text = match self.config.present_mode {
            wgpu::PresentMode::Fifo => "RENDER MODE: VSYNC",
            _ => "RENDER MODE: INSTANT",
        };
        self.menu_render_mode_button_buffer.set_text(
            render_mode_text,
            &self.ui_font.attrs(),
            Shaping::Advanced,
            None,
        );
        self.menu_render_mode_button_buffer
            .shape_until_scroll(&mut self.font_system, false);
        self.menu_render_mode_button_buffer.set_size(
            Some(self.config.width as f32),
            Some(self.config.height as f32),
        );

        let show_cursor = self.menu_state.selected_field == MenuField::ServerAddress
            && !self.menu_state.selected_all;
        let server_address_text = if show_cursor {
            format!("{}_", self.menu_state.server_address)
        } else {
            self.menu_state.server_address.clone()
        };
        self.menu_server_address_input_buffer.set_text(
            &server_address_text,
            &self.ui_font.attrs(),
            Shaping::Advanced,
            None,
        );
        self.menu_server_address_input_buffer
            .shape_until_scroll(&mut self.font_system, false);
        self.menu_server_address_input_buffer.set_size(
            Some((layout.server_address_input.w - 24.0).max(32.0)),
            Some(layout.server_address_input.h),
        );
    }

    /// Menu blur is applied in the composite shader; menu labels are drawn in the text pass.
    pub fn render_menu(&mut self, _encoder: &mut wgpu::CommandEncoder, _view: &wgpu::TextureView) {}

    /// Projects remote player world positions into screen space for name-tag
    /// rendering.
    ///
    /// Currently a stub; the actual projection logic is handled by
    /// [`queue_remote_players_labels`] in the multiplayer player module and
    /// the results are consumed directly in `render`.  This method exists as
    /// a hook for future per-player rendering work (e.g., health bars, custom
    /// skins) that would require a dedicated render pass rather than a text
    /// overlay.
    ///
    /// # Parameters
    /// - `_view_proj` – Combined view-projection matrix (unused by the stub).
    /// - `_width`     – Surface width in pixels (unused by the stub).
    /// - `_height`    – Surface height in pixels (unused by the stub).
    pub fn render_remote_players(&mut self, _view_proj: &glam::Mat4, _width: f32, _height: f32) {}
}
