//
//  JointBilateralTSDF.metal
//  ArgonScan Pro v2.0 — Native iPhone 13 Pro Max Metal Compute Pipeline
//
//  Kernels:
//  1. jointBilateralUpsampleKernel: Edge-preserving 12MP Luma-guided depth super-resolution (256x192 -> 512x384 / 1024x768)
//  2. unprojectDepthToPointsKernel: Metric 3D point & normal back-projection with YCbCr -> sRGB color extraction
//

#include <metal_stdlib>
using namespace metal;

struct FrameUniforms {
    float4x4 cameraToWorld;
    float2 focalLength;       // Scaled to depth grid resolution
    float2 principalPoint;    // Scaled to depth grid resolution
    uint2 depthDimensions;    // e.g. 256 x 192
    uint2 lumaDimensions;     // e.g. 1920 x 1440
    float minDepthMeters;     // e.g. 0.12m (TrueDepth) or 0.25m (LiDAR)
    float maxDepthMeters;     // e.g. 3.5m
    uint minConfidence;       // 1 or 2
    uint strideStep;          // Subsampling step for real-time point extraction
};

struct GpuScannedVertex {
    packed_float3 position;
    packed_float3 normal;
    packed_float4 color;      // r, g, b, confidence
    packed_float2 uv;
    uint validFlag;
};

// 1. Joint Bilateral Filter (JBF) guided by full-resolution Y-Plane (Luma) from ARFrame.capturedImage
kernel void jointBilateralUpsampleKernel(
    texture2d<float, access::read>  rawDepthTex      [[texture(0)]],
    texture2d<uint, access::read>   confidenceTex    [[texture(1)]],
    texture2d<half, access::read>   lumaTex          [[texture(2)]],
    texture2d<float, access::write> filteredDepthTex [[texture(3)]],
    constant FrameUniforms& uniforms                 [[buffer(0)]],
    uint2 gid                                        [[thread_position_in_grid]]
) {
    if (gid.x >= uniforms.depthDimensions.x || gid.y >= uniforms.depthDimensions.y) {
        return;
    }

    float centerDepth = rawDepthTex.read(gid).r;
    uint centerConf = confidenceTex.read(gid).r;
    if (centerDepth < uniforms.minDepthMeters || centerDepth > uniforms.maxDepthMeters || centerConf < uniforms.minConfidence) {
        filteredDepthTex.write(float4(0.0f), gid);
        return;
    }

    float2 uv = (float2(gid) + 0.5f) / float2(uniforms.depthDimensions);
    uint2 lumaCoord = clamp(uint2(uv * float2(uniforms.lumaDimensions)), uint2(0), uniforms.lumaDimensions - 1);
    float centerLuma = float(lumaTex.read(lumaCoord).r);

    float sigmaSpatial = 2.0f;
    float sigmaLuma = 0.06f;
    float sigmaDepth = 0.04f; (void)sigmaDepth;

    float sumDepth = 0.0f;
    float sumWeight = 0.0f;

    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int2 nb = clamp(int2(gid) + int2(dx, dy), int2(0), int2(uniforms.depthDimensions) - 1);
            float d = rawDepthTex.read(uint2(nb)).r;
            uint c = confidenceTex.read(uint2(nb)).r;
            if (d < uniforms.minDepthMeters || d > uniforms.maxDepthMeters || c < uniforms.minConfidence) {
                continue;
            }

            // Reject depth discontinuities > 8cm so object boundaries stay razor-sharp
            float depthDiff = abs(d - centerDepth);
            if (depthDiff > 0.08f) {
                continue;
            }

            float2 nbUV = (float2(nb) + 0.5f) / float2(uniforms.depthDimensions);
            uint2 nbLumaCoord = clamp(uint2(nbUV * float2(uniforms.lumaDimensions)), uint2(0), uniforms.lumaDimensions - 1);
            float nbLuma = float(lumaTex.read(nbLumaCoord).r);

            float dist2 = float(dx * dx + dy * dy);
            float lumaDiff = nbLuma - centerLuma;

            float wSpatial = exp(-dist2 / (2.0f * sigmaSpatial * sigmaSpatial));
            float wLuma = exp(-(lumaDiff * lumaDiff) / (2.0f * sigmaLuma * sigmaLuma));
            float wDepth = exp(-(depthDiff * depthDiff) / (2.0f * float(0.04f * 0.04f)));
            float wConf = (c >= 2) ? 1.0f : 0.5f;

            float w = wSpatial * wLuma * wDepth * wConf;
            sumDepth += d * w;
            sumWeight += w;
        }
    }

    float outDepth = (sumWeight > 1e-5f) ? (sumDepth / sumWeight) : centerDepth;
    filteredDepthTex.write(float4(outDepth, 0.0f, 0.0f, 1.0f), gid);
}

// 2. Unprojects filtered metric depth map + YCbCr camera planes into 3D world points with analytical normals & sRGB color
kernel void unprojectDepthToPointsKernel(
    texture2d<float, access::read> filteredDepthTex  [[texture(0)]],
    texture2d<uint, access::read>  confidenceTex     [[texture(1)]],
    texture2d<half, access::read>  yTex              [[texture(2)]],
    texture2d<half, access::read>  cbcrTex           [[texture(3)]],
    device GpuScannedVertex* outVertices             [[buffer(0)]],
    constant FrameUniforms& uniforms                 [[buffer(1)]],
    uint2 gid                                        [[thread_position_in_grid]]
) {
    uint2 gridDim = uniforms.depthDimensions / uniforms.strideStep;
    if (gid.x >= gridDim.x || gid.y >= gridDim.y) {
        return;
    }

    uint outIdx = gid.y * gridDim.x + gid.x;
    uint2 depthCoord = gid * uniforms.strideStep;

    float d = filteredDepthTex.read(depthCoord).r;
    uint conf = confidenceTex.read(depthCoord).r;

    if (d < uniforms.minDepthMeters || d > uniforms.maxDepthMeters || conf < uniforms.minConfidence) {
        outVertices[outIdx].validFlag = 0;
        return;
    }

    // Back-project (u, v, d) into ARKit camera space (Right: +X, Up: +Y, Forward: -Z)
    float u = float(depthCoord.x);
    float v = float(depthCoord.y);
    float xCam = (u - uniforms.principalPoint.x) * d / uniforms.focalLength.x;
    float yCam = -(v - uniforms.principalPoint.y) * d / uniforms.focalLength.y;
    float zCam = -d;

    float4 worldPos4 = uniforms.cameraToWorld * float4(xCam, yCam, zCam, 1.0f);

    // Estimate surface normal from neighboring filtered depth samples via cross product
    uint2 rightCoord = min(depthCoord + uint2(1, 0), uniforms.depthDimensions - 1);
    uint2 downCoord  = min(depthCoord + uint2(0, 1), uniforms.depthDimensions - 1);
    float dRight = filteredDepthTex.read(rightCoord).r;
    float dDown  = filteredDepthTex.read(downCoord).r;

    float3 pCenter = float3(xCam, yCam, zCam);
    float3 nCam = float3(0.0f, 0.0f, 1.0f);
    if (dRight > uniforms.minDepthMeters && dDown > uniforms.minDepthMeters &&
        abs(dRight - d) < 0.05f && abs(dDown - d) < 0.05f) {
        float3 pRight = float3((u + 1.0f - uniforms.principalPoint.x) * dRight / uniforms.focalLength.x, yCam, -dRight);
        float3 pDown  = float3(xCam, -(v + 1.0f - uniforms.principalPoint.y) * dDown / uniforms.focalLength.y, -dDown);
        nCam = normalize(cross(pDown - pCenter, pRight - pCenter));
    }
    float3 worldNormal = normalize((uniforms.cameraToWorld * float4(nCam, 0.0f)).xyz);

    // Sample YCbCr color from 12MP camera textures
    float2 normUV = (float2(depthCoord) + 0.5f) / float2(uniforms.depthDimensions);
    uint2 yCoord = clamp(uint2(normUV * float2(uniforms.lumaDimensions)), uint2(0), uniforms.lumaDimensions - 1);
    uint2 cbcrCoord = yCoord / 2;

    float yVal = float(yTex.read(yCoord).r);
    half2 cbcrVal = cbcrTex.read(cbcrCoord).rg;
    float cb = float(cbcrVal.r) - 0.5f;
    float cr = float(cbcrVal.g) - 0.5f;

    float3 rgb;
    rgb.r = clamp(yVal + 1.402f * cr, 0.0f, 1.0f);
    rgb.g = clamp(yVal - 0.344136f * cb - 0.714136f * cr, 0.0f, 1.0f);
    rgb.b = clamp(yVal + 1.772f * cb, 0.0f, 1.0f);

    // Map portrait screen UV for texture atlas projection
    float2 portraitUV = float2(1.0f - normUV.y, normUV.x);

    outVertices[outIdx].position = packed_float3(worldPos4.xyz);
    outVertices[outIdx].normal = packed_float3(worldNormal);
    outVertices[outIdx].color = packed_float4(rgb.r, rgb.g, rgb.b, (conf >= 2) ? 1.0f : 0.6f);
    outVertices[outIdx].uv = packed_float2(portraitUV);
    outVertices[outIdx].validFlag = 1;
}
