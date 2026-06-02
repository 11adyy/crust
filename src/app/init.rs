use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use bytemuck;
use glam::{Mat4, Vec4};
use glyphon::{
    Cache, FontSystem, Metrics, Resolution, SwashCache, TextAtlas, TextRenderer, Viewport,
};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::app::texture_cache;
use crate::logger::{LogLevel, log};
use crate::ui::menu::{GameState, MenuState};
use crust::chunk_loader::ChunkLoader;
use crust::{
    Camera, DiggingState, IndirectManager, InputState, OutlineVertex, SEA_LEVEL, Uniforms, Vertex,
    WORLD_HEIGHT, World, build_crosshair,
};

use super::state::State;

fn create_menu_background_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let image = image::load_from_memory(include_bytes!("../../assets/menu.png"))
        .expect("Failed to decode assets/menu.png")
        .to_rgba8();
    let (width, height) = image.dimensions();

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Menu Background Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        image.as_raw(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Converts an array of six frustum planes from `glam::Vec4` into
/// a plain `[[f32; 4]; 6]` that can be sent directly to a GPU buffer.
///
/// # Safety
/// `glam::Vec4` has the same memory layout as `[f32; 4]`
/// (four tightly-packed 32-bit floats), so the `transmute` is sound.
///
/// # Parameters
/// - `planes` – Six frustum planes (left, right, top, bottom, near, far) in
///   world space, each encoded as `(nx, ny, nz, d)` where `nx·x + ny·y + nz·z + d = 0`.
///
/// # Returns
/// The same data as a raw `[[f32; 4]; 6]` array ready for `bytemuck::cast_slice`.
#[inline(always)]
pub fn frustum_planes_to_array(planes: &[Vec4; 6]) -> [[f32; 4]; 6] {
    unsafe { std::mem::transmute(*planes) }
}

impl State {
    /// Initializes the complete rendering state for the application.
    ///
    /// This is a large, one-shot async constructor that performs every wgpu
    /// setup step in sequence:
    ///
    /// 1. **Surface & adapter selection** – creates the OS window surface,
    ///    picks the highest-performance GPU adapter, and logs its name and backend.
    /// 2. **Device & queue** – requests a logical device, enabling
    ///    `MULTI_DRAW_INDIRECT_COUNT` when the adapter supports it so the
    ///    indirect draw manager can cull invisible chunks on the GPU.
    /// 3. **Swap-chain configuration** – prefers an sRGB surface format and
    ///    `PresentMode::Immediate` (uncapped frame rate) with 4× MSAA.
    /// 4. **Shader compilation** – compiles all WGSL shaders (terrain, water,
    ///    sky, sun, UI, Hi-Z, depth-resolve, composite).
    /// 5. **Buffers & textures** – allocates the uniform buffer,
    ///    SSR color/depth targets, MSAA resolve targets, and the
    ///    hierarchical-Z (Hi-Z) mip chain.
    /// 6. **Bind group layouts & bind groups** – wires textures, samplers, and
    ///    buffers to the correct shader bindings for each pipeline.
    /// 7. **Render pipelines** – builds one `RenderPipeline` per pass:
    ///    terrain, water (alpha-blended), crosshair UI, sun billboard,
    ///    sky dome, depth-resolve, and the final composite blit.
    /// 8. **Compute pipelines** – builds the Hi-Z downsampling compute pipeline
    ///    with one bind group per adjacent mip level pair.
    /// 9. **World & camera** – constructs the voxel `World`, finds a safe spawn
    ///    point, and positions the `Camera` there.
    /// 10. **Text rendering** – initialises `glyphon` with a bundled Google Sans
    ///     font and pre-allocates `Buffer` objects for every piece of on-screen
    ///     text (FPS counter, menu labels, hotbar slot name, etc.).
    /// 11. **Indirect draw managers** – creates `IndirectManager` instances for
    ///     opaque terrain and water, and wires them to the Hi-Z texture so GPU
    ///     occlusion culling works correctly.
    ///
    /// # Panics
    /// Panics if:
    /// - No compatible GPU adapter is found.
    /// - The logical device cannot be created.
    /// - The window surface cannot be created.
    /// - The Tokio runtime for networking cannot be created.
    pub async fn new(window: Window) -> Self {
        let window = Arc::new(window);
        let size = window.inner_size();

        
        
        

        
        
        let backend = wgpu::Backends::all();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: backend,
            ..Default::default()
        });

        
        
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        
        
        

        
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        let info = adapter.get_info();
        log(
            LogLevel::Info,
            &format!(
                "Selected adapter: {} on {:?} backend",
                info.name, info.backend
            ),
        );

        
        
        

        
        
        
        
        
        let adapter_features = adapter.features();
        let mut requested_features = wgpu::Features::empty();
        let supports_indirect_count =
            adapter_features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        if supports_indirect_count {
            requested_features |= wgpu::Features::MULTI_DRAW_INDIRECT_COUNT;
            log(LogLevel::Info, "Adapter supports MULTI_DRAW_INDIRECT_COUNT");
        }

        let supports_shader_f16 = adapter_features.contains(wgpu::Features::SHADER_F16);
        if supports_shader_f16 {
            requested_features |= wgpu::Features::SHADER_F16;
            log(LogLevel::Info, "Adapter supports SHADER_F16");
        }

        
        
        

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: requested_features,
                
                
                required_limits: adapter.limits(),
                memory_hints: Default::default(),
                experimental_features: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create GPU device");

        
        
        

        let surface_caps = surface.get_capabilities(&adapter);
        
        
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            
            
            present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps
                .alpha_modes
                .iter()
                .copied()
                .find(|mode| matches!(mode, wgpu::CompositeAlphaMode::PreMultiplied))
                .or_else(|| {
                    surface_caps
                        .alpha_modes
                        .iter()
                        .copied()
                        .find(|mode| matches!(mode, wgpu::CompositeAlphaMode::PostMultiplied))
                })
                .or_else(|| {
                    surface_caps
                        .alpha_modes
                        .iter()
                        .copied()
                        .find(|mode| matches!(mode, wgpu::CompositeAlphaMode::Inherit))
                })
                .unwrap_or(surface_caps.alpha_modes[0]),
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        
        
        

        
        
        
        
        let msaa_sample_count: u32 = 4;

        
        
        
        
        let depth_texture = Self::create_depth_texture(&device, &config, msaa_sample_count);
        let msaa_texture_view =
            Self::create_msaa_texture(&device, &config, surface_format, msaa_sample_count);

        
        
        

        
        

        /// Compiles a WGSL shader from a string literal embedded in the binary.
        
        let hiz_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hi-Z Shader"),
            
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hiz.wgsl").into()),
        });
        let terrain_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shader"),
            
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/terrain.wgsl").into()),
        });
        let water_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Shader"),
            
            
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/water.wgsl").into()),
        });
        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            
            
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ui.wgsl").into()),
        });
        let outline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Outline Shader"),
            
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/outline.wgsl").into()),
        });
        let sun_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sun Shader"),
            
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sun.wgsl").into()),
        });
        let sky_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Shader"),
            
            
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky.wgsl").into()),
        });

        
        
        

        
        
        
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[Uniforms {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [0.0, 0.0, 0.0],
                time: 0.0,
                sun_position: [0.4, -0.2, 0.3],
                is_underwater: 0.0,
                screen_size: [1920.0, 1080.0],
                
                water_level: SEA_LEVEL as f32 - 1.0,
                
                reflection_mode: 1.0,
                moon_position: [-0.4, 0.2, -0.3],
                _pad1_moon: 0.0,
                moon_intensity: 0.0,
                wind_dir: [0.8, 0.6],
                wind_speed: 1.0,
                rain_factor: 0.0,
                sky_visibility: 1.0,
                menu_blur: 1.0,
                _pad_uniforms: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        
        
        

        
        
        
        let (texture_atlas, texture_view, _atlas_width, _atlas_height) =
            texture_cache::load_or_generate_atlas(&device, &queue);

        
        
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        
        
        

        
        
        
        
        
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX
                            | wgpu::ShaderStages::FRAGMENT
                            | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        
        
        

        
        
        
        let ssr_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR Color Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssr_color_view = ssr_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let ssr_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSR Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssr_depth_view = ssr_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        
        
        let ssr_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSR Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        
        
        let flow_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Flow Map Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &flow_map_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[128, 128, 128, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let flow_map_view = flow_map_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let flow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Flow Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        
        
        

        
        
        
        
        
        
        let water_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("water_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    
                    
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let water_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &water_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&ssr_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&ssr_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&ssr_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&flow_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Sampler(&flow_sampler),
                },
            ],
            label: Some("water_bind_group"),
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
            label: Some("uniform_bind_group"),
        });

        
        
        

        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            immediate_size: 0,
        });

        let water_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Water Pipeline Layout"),
                bind_group_layouts: &[&water_bind_group_layout],
                immediate_size: 0,
            });

        
        
        

        
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &terrain_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &terrain_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        
        
        let terrain_depth_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Terrain Depth Pipeline"),
                layout: Some(&pipeline_layout),
                cache: None,
                vertex: wgpu::VertexState {
                    module: &terrain_shader,
                    entry_point: Some("vs_depth"),
                    compilation_options: Default::default(),
                    buffers: &[Vertex::desc()],
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: msaa_sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
            });

        
        
        
        
        let water_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Pipeline"),
            layout: Some(&water_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &water_shader,
                entry_point: Some("vs_water"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &water_shader,
                entry_point: Some("fs_water"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, 
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, 
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        
        
        
        let outline_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Outline Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &outline_shader,
                entry_point: Some("vs_outline"),
                compilation_options: Default::default(),
                buffers: &[OutlineVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &outline_shader,
                entry_point: Some("fs_outline"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        
        
        
        
        let crosshair_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &ui_shader,
                entry_point: Some("vs_ui"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ui_shader,
                entry_point: Some("fs_ui"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None, 
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        
        
        
        let sun_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sun Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &sun_shader,
                entry_point: Some("vs_sun"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sun_shader,
                entry_point: Some("fs_sun"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        
        
        
        
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &sky_shader,
                entry_point: Some("vs_sky"),
                compilation_options: Default::default(),
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &sky_shader,
                entry_point: Some("fs_sky"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                
                
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
        });

        
        
        

        
        
        let sun_normal = Vertex::pack_normal([0.0, 0.0, 1.0]);

        let sun_vertices = vec![
            Vertex {
                position: [-1.0, -1.0, 0.0],
                packed: Vertex::pack(sun_normal, [1.0, 1.0, 1.0], 0, 0, 1, 1),
            },
            Vertex {
                position: [1.0, -1.0, 0.0],
                packed: Vertex::pack(sun_normal, [1.0, 1.0, 1.0], 0, 1, 1, 1),
            },
            Vertex {
                position: [1.0, 1.0, 0.0],
                packed: Vertex::pack(sun_normal, [1.0, 1.0, 1.0], 0, 2, 1, 1),
            },
            Vertex {
                position: [-1.0, 1.0, 0.0],
                packed: Vertex::pack(sun_normal, [1.0, 1.0, 1.0], 0, 3, 1, 1),
            },
        ];
        let sun_indices: Vec<u32> = vec![0, 1, 2, 0, 2, 3];

        let sun_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sun Vertex Buffer"),
            contents: bytemuck::cast_slice(&sun_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sun_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sun Index Buffer"),
            contents: bytemuck::cast_slice(&sun_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        
        
        
        let world = Arc::new(parking_lot::RwLock::new(World::new()));
        let camera = Camera::new((0.0, WORLD_HEIGHT as f32 - 1.0, 0.0));
        log(
            LogLevel::Info,
            "World generation deferred until New World is clicked.",
        );

        let seed = world.read().seed;
        
        
        
        let chunk_loader = ChunkLoader::new(seed);

        
        
        
        let mesh_loader =
            crust::MeshLoader::new(Arc::clone(&world), crust::get_mesh_worker_count());

        
        
        

        
        
        let (crosshair_vertices, crosshair_indices) = build_crosshair();
        let num_crosshair_indices = crosshair_indices.len() as u32;
        let crosshair_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Crosshair Vertex Buffer"),
                contents: bytemuck::cast_slice(&crosshair_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let crosshair_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Crosshair Index Buffer"),
            contents: bytemuck::cast_slice(&crosshair_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        
        
        

        
        
        
        

        let mut font_system = FontSystem::new();
        
        
        font_system.db_mut().load_font_data(
            include_bytes!("../../assets/fonts/GoogleSans_17pt-Regular.ttf").to_vec(),
        );

        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let mut text_atlas = TextAtlas::new(&device, &queue, &cache, surface_format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );
        let mut viewport = Viewport::new(&device, &cache);
        viewport.update(
            &queue,
            Resolution {
                width: config.width,
                height: config.height,
            },
        );

        
        
        

        /// FPS counter displayed in the top-left corner.
        let fps_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(40.0, 48.0));

        
        let menu_connect_button_buffer =
            glyphon::Buffer::new(&mut font_system, Metrics::new(36.0, 44.0));
        let menu_singleplayer_button_buffer =
            glyphon::Buffer::new(&mut font_system, Metrics::new(36.0, 44.0));

        
        let hotbar_label_buffer = glyphon::Buffer::new(&mut font_system, Metrics::new(22.0, 28.0));

        
        
        

        
        
        
        
        let depth_resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Depth Resolve Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/depth_resolve.wgsl").into()),
        });
        let depth_resolve_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Depth Resolve Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true, 
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let depth_resolve_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Depth Resolve Pipeline Layout"),
                bind_group_layouts: &[&depth_resolve_bind_group_layout],
                immediate_size: 0,
            });
        let depth_resolve_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Depth Resolve Pipeline"),
                layout: Some(&depth_resolve_pipeline_layout),
                cache: None,
                module: &depth_resolve_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
            });

        
        
        

        
        
        
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
        });

        
        
        let scene_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Color Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let scene_color_view =
            scene_color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let (menu_background_texture, menu_background_view) =
            create_menu_background_texture(&device, &queue);

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Bind Group Layout"),
                entries: &[
                    
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Composite Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Bind Group"),
            layout: &composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&composite_sampler),
                },
            ],
        });
        let menu_composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Menu Composite Bind Group"),
            layout: &composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&menu_background_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&composite_sampler),
                },
            ],
        });
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                immediate_size: 0,
            });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[], 
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None, 
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, 
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        });

        
        
        

        
        
        
        let mut indirect_manager = IndirectManager::new(&device);
        let mut water_indirect_manager = IndirectManager::new(&device);

        
        
        

        
        
        
        
        
        

        let hiz_size = [config.width, config.height];
        let hiz_max_dim = config.width.max(config.height);
        
        let hiz_mips_count = (hiz_max_dim as f32).log2().floor() as u32 + 1;

        let hiz_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Hi-Z Texture"),
            size: wgpu::Extent3d {
                width: hiz_size[0],
                height: hiz_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: hiz_mips_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING   
                | wgpu::TextureUsages::TEXTURE_BINDING, 
            view_formats: &[],
        });

        
        let hiz_view = hiz_texture.create_view(&wgpu::TextureViewDescriptor::default());

        
        
        let hiz_mips = (0..hiz_mips_count)
            .map(|i| {
                hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("Hi-Z Mip View {}", i)),
                    base_mip_level: i,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect::<Vec<_>>();

        
        
        
        let hiz_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Hi-Z Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            
                            
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let hiz_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hi-Z Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Hi-Z Pipeline Layout"),
                    bind_group_layouts: &[&hiz_bind_group_layout],
                    immediate_size: 0,
                }),
            ),
            module: &hiz_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        
        
        
        let hiz_bind_groups = (0..hiz_mips_count - 1)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("Hi-Z Bind Group {}", i)),
                    layout: &hiz_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&hiz_mips[i as usize]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &hiz_mips[(i + 1) as usize],
                            ),
                        },
                    ],
                })
            })
            .collect::<Vec<_>>();

        let depth_resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Depth Resolve Bind Group"),
            layout: &depth_resolve_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&hiz_mips[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ssr_depth_view),
                },
            ],
        });

        
        
        indirect_manager.update_bind_group(&device, &hiz_view);
        water_indirect_manager.update_bind_group(&device, &hiz_view);

        
        
        

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            water_pipeline,
            outline_pipeline,
            sun_pipeline,
            sky_pipeline,
            crosshair_pipeline,
            sun_vertex_buffer,
            sun_index_buffer,
            crosshair_vertex_buffer,
            crosshair_index_buffer,
            num_crosshair_indices,
            show_crosshair: true,
            uniform_buffer,
            uniform_bind_group,
            terrain_depth_pipeline,
            depth_texture,
            msaa_texture_view,
            world,
            mesh_loader,
            camera,
            highlighted_block: None,
            input: InputState::default(),
            digging: DiggingState::default(),
            placement: super::state::BlockPlacementState::default(),
            window,
            frame_count: 0,
            last_fps_update: Instant::now(),
            current_fps: 0.0,
            frame_time_ms: 0.0,
            cpu_update_ms: 0.0,
            last_redraw: Instant::now(),
            last_frame: Instant::now(),
            mouse_captured: false,
            chunks_rendered: 0,
            subchunks_rendered: 0,
            game_start_time: Instant::now(), 
            coords_vertex_buffer: None,
            coords_index_buffer: None,
            coords_num_indices: 0,
            last_coords_position: (i32::MIN, i32::MIN, i32::MIN),
            progress_bar_vertex_buffer: None,
            progress_bar_index_buffer: None,
            texture_atlas,
            texture_view,
            texture_sampler,
            game_state: GameState::Menu,
            has_entered_world: false,
            menu_state: MenuState::default(),
            reflection_mode: 1,
            is_underwater: 0.0,
            sky_visibility: 1.0,
            remote_players: HashMap::new(),
            my_player_id: 0,
            last_position_send: Instant::now(),
            network_runtime: Some(
                tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"),
            ),
            network_rx: None,
            network_tx: None,
            last_input_time: Instant::now(),
            player_model_vertex_buffer: None,
            player_model_index_buffer: None,
            player_model_num_indices: 0,
            player_model_vertex_capacity: 0,
            player_model_index_capacity: 0,
            chunk_loader,
            last_gen_player_cx: i32::MIN,
            last_gen_player_cz: i32::MIN,
            visible_chunk_columns: Vec::new(),
            visible_chunk_cache_center: (i32::MIN, i32::MIN),
            visible_chunk_columns_dirty: true,
            ssr_color_texture,
            ssr_color_view,
            ssr_depth_texture,
            ssr_depth_view,
            ssr_sampler,
            flow_map_texture,
            flow_map_view,
            flow_sampler,
            water_bind_group,
            water_bind_group_layout,
            surface_format,
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            fps_buffer,
            show_debug_overlay: true,
            menu_connect_button_buffer,
            menu_singleplayer_button_buffer,
            hotbar_label_buffer,
            hotbar_label_width: 0.0,
            last_hotbar_slot: usize::MAX,
            player_label_buffers: Vec::new(),
            composite_pipeline,
            composite_bind_group,
            menu_composite_bind_group,
            scene_color_texture,
            scene_color_view,
            menu_background_texture,
            menu_background_view,
            indirect_manager,
            water_indirect_manager,
            hiz_texture,
            hiz_view,
            hiz_mips,
            hiz_pipeline,
            hiz_bind_groups,
            hiz_bind_group_layout,
            hiz_size,
            depth_resolve_pipeline,
            depth_resolve_bind_group,
            supports_indirect_count,
            hotbar_slot: 0,
            hotbar_vertex_buffer: None,
            hotbar_index_buffer: None,
            hotbar_num_indices: 0,
            hotbar_dirty: true,
            cursor_position: None,
            modifiers: Default::default(),
        }
    }

    /// Creates a (possibly multisampled) depth texture and returns a view into it.
    ///
    /// The texture uses `Depth32Float` for maximum precision, which is
    /// required for the Hi-Z chain (which stores raw floating-point depth
    /// values rather than normalized integers).
    ///
    /// # Parameters
    /// - `device`       – Active wgpu logical device.
    /// - `config`       – Current surface configuration; width/height are read
    ///                    from here so the depth texture always matches the
    ///                    swap-chain resolution.
    /// - `sample_count` – Number of MSAA samples.  Pass `1` for a
    ///                    single-sampled texture (e.g., SSR targets) or `4`
    ///                    for the main multisampled depth buffer.
    ///
    /// # Returns
    /// A `TextureView` wrapping the newly created depth texture.
    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            
            
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Creates a multisampled color texture used as the MSAA render target.
    ///
    /// All geometry passes render into this texture.  At the end of each frame
    /// it is resolved to the single-sampled `scene_color_texture` (and
    /// ultimately to the swap-chain surface) by the wgpu resolve attachment
    /// mechanism.
    ///
    /// # Parameters
    /// - `device`       – Active wgpu logical device.
    /// - `config`       – Current surface configuration.
    /// - `format`       – Surface pixel format (sRGB if available).
    /// - `sample_count` – Number of MSAA samples (typically 4).
    ///
    /// # Returns
    /// A `TextureView` wrapping the newly created MSAA color texture.
    pub fn create_msaa_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            
            
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        msaa_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}
