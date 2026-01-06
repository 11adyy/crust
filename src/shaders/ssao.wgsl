/// Screen Space Ambient Occlusion (SSAO) Shader
///
/// Uses depth-only approach with position reconstruction from depth buffer.
/// Samples hemisphere around each pixel to detect occlusion.

struct Uniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sun_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    sun_position: vec3<f32>,
    is_underwater: f32,
    screen_size: vec2<f32>,
    water_level: f32,
    reflection_mode: f32,
};

struct SSAOParams {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    samples: array<vec4<f32>, 64>,
    noise_scale: vec2<f32>,
    radius: f32,
    bias: f32,
    intensity: f32,
    aspect_ratio: f32,
    _padding0: f32,
    _padding1: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<uniform> ssao_params: SSAOParams;

@group(0) @binding(2)
var depth_texture: texture_depth_2d;

@group(0) @binding(3)
var noise_texture: texture_2d<f32>;

@group(0) @binding(4)
var point_sampler: sampler;


var<private> positions: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
);

var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(2.0, 1.0),
    vec2<f32>(0.0, -1.0)
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

/// Reconstruct view-space position from depth
fn get_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    
    let ndc = vec4<f32>(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0,  
        depth,
        1.0
    );
    
    
    let view_pos = ssao_params.inv_proj * ndc;
    return view_pos.xyz / view_pos.w;
}

/// Calculate normal from depth buffer using cross product of partial derivatives
fn get_normal_from_depth(uv: vec2<f32>, texel_size: vec2<f32>) -> vec3<f32> {
    
    let offset_x = texel_size.x;
    let offset_y = texel_size.y;

    let depth_center = textureSample(depth_texture, point_sampler, uv);
    let depth_left = textureSample(depth_texture, point_sampler, uv - vec2<f32>(offset_x, 0.0));
    let depth_right = textureSample(depth_texture, point_sampler, uv + vec2<f32>(offset_x, 0.0));
    let depth_up = textureSample(depth_texture, point_sampler, uv - vec2<f32>(0.0, offset_y));
    let depth_down = textureSample(depth_texture, point_sampler, uv + vec2<f32>(0.0, offset_y));

    let pos_center = get_view_pos(uv, depth_center);
    let pos_left = get_view_pos(uv - vec2<f32>(offset_x, 0.0), depth_left);
    let pos_right = get_view_pos(uv + vec2<f32>(offset_x, 0.0), depth_right);
    let pos_up = get_view_pos(uv - vec2<f32>(0.0, offset_y), depth_up);
    let pos_down = get_view_pos(uv + vec2<f32>(0.0, offset_y), depth_down);
    
    
    let dx = pos_right - pos_left;
    let dy = pos_down - pos_up;

    return normalize(cross(dx, dy));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel_size = 1.0 / uniforms.screen_size;
    
    
    let depth = textureSample(depth_texture, point_sampler, in.uv);
    
    
    if depth >= 0.9999 {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    
    
    let frag_pos = get_view_pos(in.uv, depth);
    let normal = get_normal_from_depth(in.uv, texel_size);
    
    
    let noise_uv = in.uv * ssao_params.noise_scale;
    let random_vec = textureSample(noise_texture, point_sampler, noise_uv).xyz * 2.0 - 1.0;
    
    
    let tangent = normalize(random_vec - normal * dot(random_vec, normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);
    
    
    var occlusion: f32 = 0.0;
    let radius = ssao_params.radius;
    let bias = ssao_params.bias;

    for (var i: i32 = 0; i < 64; i++) {
        
        let sample_dir = tbn * ssao_params.samples[i].xyz;
        var sample_pos = frag_pos + sample_dir * radius;
        
        
        let offset = ssao_params.proj * vec4<f32>(sample_pos, 1.0);
        var sample_uv = offset.xy / offset.w;
        sample_uv = sample_uv * 0.5 + 0.5;
        sample_uv.y = 1.0 - sample_uv.y;  
        
        
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }
        
        
        let sample_depth = textureSample(depth_texture, point_sampler, sample_uv);
        let sample_view_pos = get_view_pos(sample_uv, sample_depth);
        
        
        let range_check = smoothstep(0.0, 1.0, radius / abs(frag_pos.z - sample_view_pos.z));
        
        
        let occluded = select(0.0, 1.0, sample_view_pos.z >= sample_pos.z + bias);
        occlusion += occluded * range_check;
    }
    
    
    occlusion = 1.0 - (occlusion / 64.0);
    
    
    occlusion = pow(occlusion, ssao_params.intensity);

    return vec4<f32>(occlusion, occlusion, occlusion, 1.0);
}
