use bytemuck::{Pod, Zeroable};

/// Optimized vertex layout: 32 bytes (down from 56 bytes)
///
/// Changes:
/// - `normal`: [f32; 3] (12B) → [i8; 4] (4B) using Snorm8x4 (axis-aligned normals need only ±1/0)
/// - `color`: [f32; 3] (12B) → [u8; 4] (4B) using Unorm8x4 (8-bit per channel is sufficient)
/// - Removed `roughness` and `metallic` (unused — shaders ignored them as `_padding`)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],  
    pub normal: [i8; 4],     
    pub color: [u8; 4],      
    pub uv: [f32; 2],        
    pub tex_index: f32,      
}

impl Vertex {
    /// Pack a float normal [x, y, z] into [i8; 4] for Snorm8x4 format.
    /// For axis-aligned normals (voxel faces), this is lossless.
    #[inline]
    pub fn pack_normal(n: [f32; 3]) -> [i8; 4] {
        [
            (n[0].clamp(-1.0, 1.0) * 127.0) as i8,
            (n[1].clamp(-1.0, 1.0) * 127.0) as i8,
            (n[2].clamp(-1.0, 1.0) * 127.0) as i8,
            0,
        ]
    }

    /// Pack a float color [r, g, b] into [u8; 4] (RGBA) for Unorm8x4 format.
    #[inline]
    pub fn pack_color(c: [f32; 3]) -> [u8; 4] {
        [
            (c[0].clamp(0.0, 1.0) * 255.0) as u8,
            (c[1].clamp(0.0, 1.0) * 255.0) as u8,
            (c[2].clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ]
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Snorm8x4,
                },
                
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
                
                wgpu::VertexAttribute {
                    offset: 20,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
                
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}
