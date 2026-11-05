//
//  HybridSensorFusionManager.swift
//  ArgonScan Pro v2.1 — High-Precision iPhone 13 Pro Max 3D Reconstruction Engine
//
//  Key Fixes in v2.1:
//  1. Exact Sensor-Space Intrinsics Projection (fx, fy, cx, cy + simd_inverse(camera.transform))
//     with normal-incidence weighting and 1-to-4 triangle midpoint subdivision for accurate multi-view color baking.
//  2. Near-Field TrueDepth (640x480) Organized Depth-Grid Surface Triangulation:
//     Fuses multi-frame TrueDepth 32-bit float depth maps and triangulates contiguous foreground surfaces into a real 3D Mesh with UV texture.
//  3. Zero-Jitter 3D Preview Studio State: Pauses background sensor streams during 3D inspection and tracks explicit modelRevision.
//

import Foundation
import ARKit
import SceneKit
import UIKit
import AVFoundation
import CoreVideo
import CoreMotion
import Metal
import CoreHaptics

public enum ScanRangeMode: String, CaseIterable {
    case nearFieldTrueDepthMacro = "近场 TrueDepth(15-45cm)"
    case midFieldLiDARPhotogrammetry = "中场高精 LiDAR(0.3-1.8m)"
    case farFieldRoomGaussianSplat = "远场空间(1.0-5.0m)"
}

public enum PreviewShaderMode: String, CaseIterable {
    case bakedTexture = "高清烘焙材质"
    case texturedWireframe = "材质+拓扑网"
    case clayMatCap = "高精雕塑白模"
    case pointCloud = "稠密体素点云"
}

public struct FrameUniforms {
    public var cameraToWorld: simd_float4x4
    public var focalLength: SIMD2<Float>
    public var principalPoint: SIMD2<Float>
    public var depthDimensions: SIMD2<UInt32>
    public var lumaDimensions: SIMD2<UInt32>
    public var minDepthMeters: Float
    public var maxDepthMeters: Float
    public var minConfidence: UInt32
    public var strideStep: UInt32
}

public struct GpuScannedVertex {
    public var px: Float
    public var py: Float
    public var pz: Float
    public var nx: Float
    public var ny: Float
    public var nz: Float
    public var r: Float
    public var g: Float
    public var b: Float
    public var a: Float
    public var u: Float
    public var v: Float
    public var validFlag: UInt32
}

public struct SpatialVoxelKey: Hashable {
    public let ix: Int32
    public let iy: Int32
    public let iz: Int32
}

public struct FusedSurfacePoint {
    public var position: SIMD3<Float>
    public var normal: SIMD3<Float>
    public var color: SIMD3<Float>
    public var uv: SIMD2<Float>
    public var weight: Float
}

public struct BakedSubmeshChunk {
    public let anchorIdentifier: UUID
    public var positions: [SIMD3<Float>]
    public var normals: [SIMD3<Float>]
    public var uvs: [SIMD2<Float>]
    public var colors: [SIMD4<Float>] // r, g, b, accumulatedConfidenceWeight
    public var triangleIndices: [UInt32]
    public var dedicatedTexture: UIImage?
}

public struct HardwareTelemetry {
    public var gpuFrameTimeMs: Double = 4.2
    public var memoryUsedMB: Double = 380.0
    public var activeVerticesCount: Int = 0
    public var denseVoxelPointsCount: Int = 0
    public var activeMeshFacesCount: Int = 0
    public var keyframesCaptured: Int = 0
    public var bakedTextureCount: Int = 0
    public var trueDepthCenterDistanceCM: Double = 0.0
    public var boundingBoxMM: SIMD3<Float> = .zero
    public var imuAngularVelocityRadSec: Double = 0.03
    public var statusMessage: String = "正在启动 iPhone 13 Pro Max 硬件与 Metal 管线..."
}

public final class HybridSensorFusionManager: NSObject, ObservableObject {
    @Published public var activeMode: ScanRangeMode = .midFieldLiDARPhotogrammetry
    @Published public var isCapturingMesh: Bool = false
    @Published public var isPreviewRunning: Bool = false
    @Published public var liveTextureOverlayEnabled: Bool = true
    @Published public var isBakingInProgress: Bool = false
    @Published public var hasBakedModelReady: Bool = false
    @Published public var voxelPitchMM: Double = 2.0
    @Published public var maxScanDepthMeters: Double = 1.8
    @Published public var latestKeyframeSnapshot: UIImage?
    @Published public var telemetry = HardwareTelemetry()
    // Incremented ONLY when mesh geometry is baked or smoothed, preventing 3D preview jitter!
    @Published public var modelRevision: Int = 0

    public let arSession = ARSession()
    public let trueDepthAVSession = AVCaptureSession()

    private let motionManager = CMMotionManager()
    private let metalDevice: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private var textureCache: CVMetalTextureCache?
    private var jbfPipeline: MTLComputePipelineState?
    private var unprojectPipeline: MTLComputePipelineState?
    private var gpuVertexBuffer: MTLBuffer?
    private let maxGpuPointsPerFrame = 256 * 192

    private let trueDepthQueue = DispatchQueue(label: "com.argonscan.truedepthQueue", qos: .userInitiated)
    private var videoDataOutput: AVCaptureVideoDataOutput?
    private var depthDataOutput: AVCaptureDepthDataOutput?
    private var outputSynchronizer: AVCaptureDataOutputSynchronizer?

    private let stateLock = NSLock()
    private var latestOmegaRadSec: Double = 0.0
    private var lastTelemetryUpdateTime: TimeInterval = 0
    private var lastTextureSampleTime: TimeInterval = 0
    private var lastMetalDepthPassTime: TimeInterval = 0
    private var lastTrueDepthFrameTime: TimeInterval = 0

    private var spatialVoxelGrid: [SpatialVoxelKey: FusedSurfacePoint] = [:]
    private var bakedChunksByAnchor: [UUID: BakedSubmeshChunk] = [:]
    private var bestKeyframeImage: UIImage?

    // TrueDepth Multi-Frame Accumulated Depth + Color Grid (320 x 240) for Mesh Triangulation
    private let tdGridW = 320
    private let tdGridH = 240
    private var tdAccumDepth: [Float] = []
    private var tdAccumWeight: [Float] = []
    private var tdAccumColor: [SIMD3<Float>] = []
    private var tdLatestIntrinsics: simd_float3x3 = matrix_identity_float3x3
    private var tdLatestRefDims: CGSize = CGSize(width: 640, height: 480)
    private var tdLatestTextureImage: UIImage?
    private let trueDepthChunkID = UUID()

    public override init() {
        let dev = MTLCreateSystemDefaultDevice()
        self.metalDevice = dev
        self.commandQueue = dev?.makeCommandQueue()
        self.tdAccumDepth = Array(repeating: 0.0, count: 320 * 240)
        self.tdAccumWeight = Array(repeating: 0.0, count: 320 * 240)
        self.tdAccumColor = Array(repeating: SIMD3<Float>(0.7, 0.75, 0.8), count: 320 * 240)
        super.init()

        arSession.delegate = self
        setupMetalResources()
    }

    private func setupMetalResources() {
        guard let device = metalDevice else { return }
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        gpuVertexBuffer = device.makeBuffer(
            length: maxGpuPointsPerFrame * MemoryLayout<GpuScannedVertex>.stride,
            options: .storageModeShared
        )

        guard let library = device.makeDefaultLibrary() else { return }
        if let jbfFunc = library.makeFunction(name: "jointBilateralUpsampleKernel") {
            jbfPipeline = try? device.makeComputePipelineState(function: jbfFunc)
        }
        if let unprojFunc = library.makeFunction(name: "unprojectDepthToPointsKernel") {
            unprojectPipeline = try? device.makeComputePipelineState(function: unprojFunc)
        }
    }

    private func startIMUStreamIfAvailable() {
        guard motionManager.isDeviceMotionAvailable, !motionManager.isDeviceMotionActive else { return }
        motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
        let imuQueue = OperationQueue()
        imuQueue.name = "com.argonscan.imuQueue"
        imuQueue.qualityOfService = .userInteractive

        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: imuQueue) { [weak self] motion, _ in
            guard let self = self, let rot = motion?.rotationRate else { return }
            let omega = sqrt(rot.x * rot.x + rot.y * rot.y + rot.z * rot.z)
            self.stateLock.lock()
            self.latestOmegaRadSec = omega
            self.stateLock.unlock()
        }
    }

    public func ensureCameraPreviewRunning() {
        let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
        switch authStatus {
        case .authorized:
            runSensorPipeline(for: activeMode, resetAnchors: false)
        case .notDetermined:
            updateStatus("请允许相机权限以启动 LiDAR 与 TrueDepth 精密扫描...")
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    if granted {
                        self.runSensorPipeline(for: self.activeMode, resetAnchors: true)
                    } else {
                        self.updateStatus("相机权限被拒绝：请前往系统设置开启相机权限")
                    }
                }
            }
        case .denied, .restricted:
            updateStatus("相机权限未开启：请在「设置 -> ArgonScan Pro」中开启相机")
        @unknown default:
            break
        }
    }

    /// Pauses camera & IMU streams while inspecting the model inside the 3D Preview Studio so the viewport never jitters
    public func pauseSensorStreamsForPreviewStudio() {
        isCapturingMesh = false
        arSession.pause()
        if trueDepthAVSession.isRunning {
            trueDepthQueue.async { [weak self] in
                self?.trueDepthAVSession.stopRunning()
            }
        }
        motionManager.stopDeviceMotionUpdates()
    }

    /// Resumes the active sensor pipeline when returning from the 3D Preview Studio
    public func resumeSensorStreamsFromPreviewStudio() {
        runSensorPipeline(for: activeMode, resetAnchors: false)
    }

    public func switchMode(_ newMode: ScanRangeMode) {
        activeMode = newMode
        isCapturingMesh = false
        switch newMode {
        case .nearFieldTrueDepthMacro:
            voxelPitchMM = 0.8
            maxScanDepthMeters = 0.45
        case .midFieldLiDARPhotogrammetry:
            voxelPitchMM = 1.5
            maxScanDepthMeters = 1.80
        case .farFieldRoomGaussianSplat:
            voxelPitchMM = 5.0
            maxScanDepthMeters = 4.50
        }
        runSensorPipeline(for: newMode, resetAnchors: true)
    }

    public func toggleCapture() {
        if !isPreviewRunning {
            ensureCameraPreviewRunning()
            isCapturingMesh = true
            return
        }
        isCapturingMesh.toggle()
        if isCapturingMesh {
            if activeMode == .nearFieldTrueDepthMacro {
                updateStatus("近场 TrueDepth 采集中：请将物体置于屏幕前 15–45cm 保持稳定...")
            } else {
                updateStatus("中远场采集中：正在按相机真实内参投影烘焙多视角 RGB 材质...")
            }
        } else {
            updateStatus("已暂停采集 · 点击「烘焙并 3D 预览」生成平滑 3D 网格")
        }
    }

    public func clearScanAndReset() {
        stateLock.lock()
        bakedChunksByAnchor.removeAll()
        spatialVoxelGrid.removeAll()
        bestKeyframeImage = nil
        tdAccumDepth = Array(repeating: 0.0, count: tdGridW * tdGridH)
        tdAccumWeight = Array(repeating: 0.0, count: tdGridW * tdGridH)
        tdAccumColor = Array(repeating: SIMD3<Float>(0.7, 0.75, 0.8), count: tdGridW * tdGridH)
        tdLatestTextureImage = nil
        stateLock.unlock()

        DispatchQueue.main.async {
            self.hasBakedModelReady = false
            self.latestKeyframeSnapshot = nil
            self.telemetry.activeVerticesCount = 0
            self.telemetry.denseVoxelPointsCount = 0
            self.telemetry.activeMeshFacesCount = 0
            self.telemetry.keyframesCaptured = 0
            self.telemetry.bakedTextureCount = 0
            self.telemetry.boundingBoxMM = .zero
            self.modelRevision += 1
            self.telemetry.statusMessage = "已清空历史点云与网格，重新开始扫描"
        }
        runSensorPipeline(for: activeMode, resetAnchors: true)
    }

    private func runSensorPipeline(for mode: ScanRangeMode, resetAnchors: Bool) {
        startIMUStreamIfAvailable()
        let options: ARSession.RunOptions = resetAnchors ? [.resetTracking, .removeExistingAnchors] : []

        switch mode {
        case .nearFieldTrueDepthMacro:
            arSession.pause()
            startTrueDepthObjectScanner()

        case .midFieldLiDARPhotogrammetry, .farFieldRoomGaussianSplat:
            if trueDepthAVSession.isRunning {
                trueDepthQueue.async { [weak self] in
                    self?.trueDepthAVSession.stopRunning()
                }
            }
            guard ARWorldTrackingConfiguration.isSupported else {
                updateStatus("当前设备不支持 ARWorldTrackingConfiguration")
                return
            }
            let config = ARWorldTrackingConfiguration()
            config.planeDetection = [.horizontal, .vertical]
            config.environmentTexturing = .automatic
            config.isLightEstimationEnabled = true

            if let bestFormat = ARWorldTrackingConfiguration.supportedVideoFormats.max(by: {
                ($0.imageResolution.width * $0.imageResolution.height) < ($1.imageResolution.width * $1.imageResolution.height)
            }) {
                config.videoFormat = bestFormat
            }

            if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
                config.frameSemantics.insert(.sceneDepth)
            }
            if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
                config.frameSemantics.insert(.smoothedSceneDepth)
            }
            if ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification) {
                config.sceneReconstruction = .meshWithClassification
            } else if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
                config.sceneReconstruction = .mesh
            }

            arSession.run(config, options: options)
            DispatchQueue.main.async {
                self.isPreviewRunning = true
                self.telemetry.statusMessage = "后置 LiDAR + 12MP 广角已就绪 · 点击 Start Capture 开始捕捉"
            }
        }
    }

    // MARK: - Front TrueDepth 640x480 Arbitrary Small-Object Scanner & Surface Mesh Triangulator

    private func startTrueDepthObjectScanner() {
        trueDepthQueue.async { [weak self] in
            guard let self = self else { return }
            self.trueDepthAVSession.beginConfiguration()
            self.trueDepthAVSession.inputs.forEach { self.trueDepthAVSession.removeInput($0) }
            self.trueDepthAVSession.outputs.forEach { self.trueDepthAVSession.removeOutput($0) }
            self.trueDepthAVSession.sessionPreset = .vga640x480

            guard let device = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.trueDepthAVSession.canAddInput(input) else {
                self.trueDepthAVSession.commitConfiguration()
                self.updateStatus("前置 TrueDepth 初始化失败，已回退至后置 LiDAR")
                return
            }
            self.trueDepthAVSession.addInput(input)

            let vOutput = AVCaptureVideoDataOutput()
            vOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            vOutput.alwaysDiscardsLateVideoFrames = true

            let dOutput = AVCaptureDepthDataOutput()
            dOutput.isFilteringEnabled = true
            dOutput.alwaysDiscardsLateDepthData = true

            guard self.trueDepthAVSession.canAddOutput(vOutput),
                  self.trueDepthAVSession.canAddOutput(dOutput) else {
                self.trueDepthAVSession.commitConfiguration()
                return
            }
            self.trueDepthAVSession.addOutput(vOutput)
            self.trueDepthAVSession.addOutput(dOutput)

            if let conn = dOutput.connection(with: .depthData) {
                conn.isEnabled = true
            }

            self.videoDataOutput = vOutput
            self.depthDataOutput = dOutput
            self.outputSynchronizer = AVCaptureDataOutputSynchronizer(dataOutputs: [vOutput, dOutput])
            self.outputSynchronizer?.setDelegate(self, queue: self.trueDepthQueue)

            self.trueDepthAVSession.commitConfiguration()
            self.trueDepthAVSession.startRunning()

            DispatchQueue.main.async {
                self.isPreviewRunning = true
                self.telemetry.statusMessage = "前置 TrueDepth 取景已开启 · 将物体置于屏幕前 15–45cm 点击 Start Capture"
            }
        }
    }

    /// Triangulates the accumulated 320x240 TrueDepth depth & color grid into a contiguous, smooth 3D Triangle Mesh!
    private func triangulateTrueDepthAccumulatedGrid() {
        stateLock.lock()
        let depthGrid = tdAccumDepth
        let weightGrid = tdAccumWeight
        let colorGrid = tdAccumColor
        let intrinsics = tdLatestIntrinsics
        let refDims = tdLatestRefDims
        let texImage = tdLatestTextureImage
        stateLock.unlock()

        let w = tdGridW
        let h = tdGridH
        guard depthGrid.count == w * h else { return }

        // 1. Bilateral smoothing pass on the 2D depth grid to eliminate IR quantization ripples
        var smoothedD = depthGrid
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let idx = y * w + x
                let cDepth = depthGrid[idx]
                guard weightGrid[idx] > 0.1, cDepth >= 0.12, cDepth <= Float(maxScanDepthMeters) else { continue }
                var sumD: Float = 0.0
                var sumW: Float = 0.0
                for dy in -1...1 {
                    for dx in -1...1 {
                        let nIdx = (y + dy) * w + (x + dx)
                        let nDepth = depthGrid[nIdx]
                        guard weightGrid[nIdx] > 0.1, abs(nDepth - cDepth) < 0.012 else { continue }
                        let wSpatial: Float = (dx == 0 && dy == 0) ? 1.0 : 0.6
                        sumD += nDepth * wSpatial
                        sumW += wSpatial
                    }
                }
                if sumW > 0 {
                    smoothedD[idx] = sumD / sumW
                }
            }
        }

        // 2. Unproject valid grid cells into 3D vertices & build index map
        let scaleX = Float(w) / Float(max(1.0, refDims.width))
        let scaleY = Float(h) / Float(max(1.0, refDims.height))
        let fx = max(100.0, intrinsics[0][0] * scaleX)
        let fy = max(100.0, intrinsics[1][1] * scaleY)
        let cx = intrinsics[2][0] * scaleX
        let cy = intrinsics[2][1] * scaleY

        var vertexIndexMap = Array(repeating: Int32(-1), count: w * h)
        var positions = [SIMD3<Float>]()
        var normals = [SIMD3<Float>]()
        var uvs = [SIMD2<Float>]()
        var colors = [SIMD4<Float>]()

        for y in 0..<h {
            for x in 0..<w {
                let idx = y * w + x
                let z = smoothedD[idx]
                guard weightGrid[idx] > 0.1, z >= 0.12, z <= Float(maxScanDepthMeters) else { continue }

                // Mirror X for natural front-camera TrueDepth coordinate alignment
                let x3D = -(Float(x) - cx) * z / fx
                let y3D = -(Float(y) - cy) * z / fy
                let z3D = -z

                vertexIndexMap[idx] = Int32(positions.count)
                positions.append(SIMD3<Float>(x3D, y3D, z3D))
                normals.append(SIMD3<Float>(0, 0, 1))
                // Portrait-aligned UV coordinates
                let u = Float(y) / Float(h - 1)
                let v = Float(x) / Float(w - 1)
                uvs.append(SIMD2<Float>(u, v))
                let c = colorGrid[idx]
                colors.append(SIMD4<Float>(c.x, c.y, c.z, 1.0))
            }
        }

        guard positions.count >= 3 else { return }

        // 3. Connect adjacent 2x2 valid grid cells into triangles if depth step < 1.8cm
        var triangleIndices = [UInt32]()
        let maxStepMeters: Float = 0.018

        for y in 0..<(h - 1) {
            for x in 0..<(w - 1) {
                let i00 = y * w + x
                let i10 = y * w + (x + 1)
                let i01 = (y + 1) * w + x
                let i11 = (y + 1) * w + (x + 1)

                let v00 = vertexIndexMap[i00]
                let v10 = vertexIndexMap[i10]
                let v01 = vertexIndexMap[i01]
                let v11 = vertexIndexMap[i11]

                let d00 = smoothedD[i00]
                let d10 = smoothedD[i10]
                let d01 = smoothedD[i01]
                let d11 = smoothedD[i11]

                if v00 >= 0 && v10 >= 0 && v01 >= 0 {
                    if abs(d00 - d10) < maxStepMeters && abs(d00 - d01) < maxStepMeters && abs(d10 - d01) < maxStepMeters {
                        triangleIndices.append(UInt32(v00))
                        triangleIndices.append(UInt32(v01))
                        triangleIndices.append(UInt32(v10))
                    }
                }
                if v10 >= 0 && v11 >= 0 && v01 >= 0 {
                    if abs(d10 - d11) < maxStepMeters && abs(d10 - d01) < maxStepMeters && abs(d11 - d01) < maxStepMeters {
                        triangleIndices.append(UInt32(v10))
                        triangleIndices.append(UInt32(v01))
                        triangleIndices.append(UInt32(v11))
                    }
                }
            }
        }

        guard triangleIndices.count >= 3 else { return }

        // 4. Compute smooth area-weighted vertex normals from the generated triangles
        var accumNormals = Array(repeating: SIMD3<Float>.zero, count: positions.count)
        let triCount = triangleIndices.count / 3
        for t in 0..<triCount {
            let ia = Int(triangleIndices[t * 3])
            let ib = Int(triangleIndices[t * 3 + 1])
            let ic = Int(triangleIndices[t * 3 + 2])
            let pa = positions[ia]
            let pb = positions[ib]
            let pc = positions[ic]
            let fn = simd_cross(pb - pa, pc - pa)
            accumNormals[ia] += fn
            accumNormals[ib] += fn
            accumNormals[ic] += fn
        }
        for i in 0..<normals.count {
            let len = simd_length(accumNormals[i])
            normals[i] = (len > 1e-6) ? (accumNormals[i] / len) : SIMD3<Float>(0, 0, 1)
        }

        let tdChunk = BakedSubmeshChunk(
            anchorIdentifier: trueDepthChunkID,
            positions: positions,
            normals: normals,
            uvs: uvs,
            colors: colors,
            triangleIndices: triangleIndices,
            dedicatedTexture: texImage
        )

        stateLock.lock()
        bakedChunksByAnchor[trueDepthChunkID] = tdChunk
        stateLock.unlock()

        DispatchQueue.main.async {
            self.telemetry.activeVerticesCount = positions.count
            self.telemetry.activeMeshFacesCount = triCount
            self.telemetry.bakedTextureCount = 1
            self.hasBakedModelReady = true
        }
    }

    // MARK: - Metal GPU Joint Bilateral Depth Super-Resolution

    private func executeMetalDepthSuperResolutionAndFusion(frame: ARFrame) {
        guard let device = metalDevice,
              let queue = commandQueue,
              let cache = textureCache,
              let jbfPSO = jbfPipeline,
              let unprojPSO = unprojectPipeline,
              let outBuffer = gpuVertexBuffer,
              let sceneDepth = frame.smoothedSceneDepth ?? frame.sceneDepth,
              let confMap = sceneDepth.confidenceMap else {
            return
        }

        let depthBuf = sceneDepth.depthMap
        let rgbBuf = frame.capturedImage

        let depthW = CVPixelBufferGetWidth(depthBuf)
        let depthH = CVPixelBufferGetHeight(depthBuf)
        let lumaW = CVPixelBufferGetWidthOfPlane(rgbBuf, 0)
        let lumaH = CVPixelBufferGetHeightOfPlane(rgbBuf, 0)

        var cvDepthTex: CVMetalTexture?
        var cvConfTex: CVMetalTexture?
        var cvYTex: CVMetalTexture?
        var cvCbCrTex: CVMetalTexture?

        guard CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, depthBuf, nil, .r32Float, depthW, depthH, 0, &cvDepthTex) == kCVReturnSuccess,
              CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, confMap, nil, .r8Uint, depthW, depthH, 0, &cvConfTex) == kCVReturnSuccess,
              CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, rgbBuf, nil, .r8Unorm, lumaW, lumaH, 0, &cvYTex) == kCVReturnSuccess,
              CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, cache, rgbBuf, nil, .rg8Unorm, lumaW / 2, lumaH / 2, 1, &cvCbCrTex) == kCVReturnSuccess,
              let depthTex = cvDepthTex.flatMap({ CVMetalTextureGetTexture($0) }),
              let confTex = cvConfTex.flatMap({ CVMetalTextureGetTexture($0) }),
              let yTex = cvYTex.flatMap({ CVMetalTextureGetTexture($0) }),
              let cbcrTex = cvCbCrTex.flatMap({ CVMetalTextureGetTexture($0) }) else {
            return
        }

        let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: depthW, height: depthH, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        guard let filteredDepthTex = device.makeTexture(descriptor: desc),
              let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            return
        }

        let intrinsics = frame.camera.intrinsics
        let scaleX = Float(depthW) / Float(lumaW)
        let scaleY = Float(depthH) / Float(lumaH)

        var uniforms = FrameUniforms(
            cameraToWorld: frame.camera.transform,
            focalLength: SIMD2<Float>(intrinsics[0][0] * scaleX, intrinsics[1][1] * scaleY),
            principalPoint: SIMD2<Float>(intrinsics[2][0] * scaleX, intrinsics[2][1] * scaleY),
            depthDimensions: SIMD2<UInt32>(UInt32(depthW), UInt32(depthH)),
            lumaDimensions: SIMD2<UInt32>(UInt32(lumaW), UInt32(lumaH)),
            minDepthMeters: 0.18,
            maxDepthMeters: Float(maxScanDepthMeters),
            minConfidence: 1,
            strideStep: 2
        )

        encoder.setComputePipelineState(jbfPSO)
        encoder.setTexture(depthTex, index: 0)
        encoder.setTexture(confTex, index: 1)
        encoder.setTexture(yTex, index: 2)
        encoder.setTexture(filteredDepthTex, index: 3)
        encoder.setBytes(&uniforms, length: MemoryLayout<FrameUniforms>.stride, index: 0)

        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let jbfGroups = MTLSize(width: (depthW + 15) / 16, height: (depthH + 15) / 16, depth: 1)
        encoder.dispatchThreadgroups(jbfGroups, threadsPerThreadgroup: tgSize)

        let gridW = depthW / Int(uniforms.strideStep)
        let gridH = depthH / Int(uniforms.strideStep)
        encoder.setComputePipelineState(unprojPSO)
        encoder.setTexture(filteredDepthTex, index: 0)
        encoder.setTexture(confTex, index: 1)
        encoder.setTexture(yTex, index: 2)
        encoder.setTexture(cbcrTex, index: 3)
        encoder.setBuffer(outBuffer, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<FrameUniforms>.stride, index: 1)

        let unprojGroups = MTLSize(width: (gridW + 15) / 16, height: (gridH + 15) / 16, depth: 1)
        encoder.dispatchThreadgroups(unprojGroups, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()

        let startGPU = CACurrentMediaTime()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let gpuElapsedMs = (CACurrentMediaTime() - startGPU) * 1000.0

        let totalOut = gridW * gridH
        let ptr = outBuffer.contents().assumingMemoryBound(to: GpuScannedVertex.self)
        let pitchMeters = Float(max(0.001, voxelPitchMM / 1000.0))

        stateLock.lock()
        if spatialVoxelGrid.count < 180_000 {
            for i in 0..<totalOut {
                let v = ptr[i]
                guard v.validFlag == 1 else { continue }
                let pos = SIMD3<Float>(v.px, v.py, v.pz)
                let norm = SIMD3<Float>(v.nx, v.ny, v.nz)
                let col = SIMD3<Float>(v.r, v.g, v.b)
                let uv = SIMD2<Float>(v.u, v.v)
                let w = v.a

                let key = SpatialVoxelKey(
                    ix: Int32(floor(pos.x / pitchMeters)),
                    iy: Int32(floor(pos.y / pitchMeters)),
                    iz: Int32(floor(pos.z / pitchMeters))
                )

                if var existing = spatialVoxelGrid[key] {
                    let newW = min(64.0, existing.weight + w)
                    existing.position = (existing.position * existing.weight + pos * w) / newW
                    existing.normal = simd_normalize(existing.normal * existing.weight + norm * w)
                    existing.color = (existing.color * existing.weight + col * w) / newW
                    existing.uv = uv
                    existing.weight = newW
                    spatialVoxelGrid[key] = existing
                } else {
                    spatialVoxelGrid[key] = FusedSurfacePoint(
                        position: pos,
                        normal: norm,
                        color: col,
                        uv: uv,
                        weight: w
                    )
                }
            }
        }
        let currentVoxelCount = spatialVoxelGrid.count
        stateLock.unlock()

        DispatchQueue.main.async {
            self.telemetry.gpuFrameTimeMs = max(1.2, gpuElapsedMs)
            self.telemetry.denseVoxelPointsCount = currentVoxelCount
        }
    }

    // MARK: - High-Precision Texture Baking & Mesh Subdivision / Smoothing

    public func bakeCurrentMeshAndTextures(completion: @escaping () -> Void) {
        DispatchQueue.main.async {
            self.isBakingInProgress = true
            self.telemetry.statusMessage = "正在构建高密度 3D 三角网格并烘焙真实相机色彩..."
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            if self.activeMode == .nearFieldTrueDepthMacro {
                self.triangulateTrueDepthAccumulatedGrid()
                self.applyLaplacianMeshSmoothing(iterations: 2, lambda: 0.35)
            } else {
                if let currentFrame = self.arSession.currentFrame {
                    self.sampleAndBakeFrameTextures(frame: currentFrame, forceAllAnchors: true)
                }
                self.applyLaplacianMeshSmoothing(iterations: 2, lambda: 0.35)
            }
            self.computeModelBoundingBoxMM()

            DispatchQueue.main.async {
                self.isBakingInProgress = false
                self.hasBakedModelReady = true
                self.modelRevision += 1
                self.telemetry.statusMessage = "3D 网格与真实材质烘焙完成！"
                completion()
            }
        }
    }

    public func applyLaplacianMeshSmoothing(iterations: Int = 2, lambda: Float = 0.35) {
        stateLock.lock()
        var chunks = bakedChunksByAnchor
        stateLock.unlock()

        for (id, var chunk) in chunks {
            let vCount = chunk.positions.count
            guard vCount > 3, chunk.triangleIndices.count >= 3 else { continue }

            var neighbors = Array(repeating: [Int](), count: vCount)
            let triCount = chunk.triangleIndices.count / 3
            for t in 0..<triCount {
                let i0 = Int(chunk.triangleIndices[t * 3])
                let i1 = Int(chunk.triangleIndices[t * 3 + 1])
                let i2 = Int(chunk.triangleIndices[t * 3 + 2])
                guard i0 < vCount, i1 < vCount, i2 < vCount else { continue }
                neighbors[i0].append(i1); neighbors[i0].append(i2)
                neighbors[i1].append(i0); neighbors[i1].append(i2)
                neighbors[i2].append(i0); neighbors[i2].append(i1)
            }

            for _ in 0..<iterations {
                var smoothedPos = chunk.positions
                var smoothedCol = chunk.colors
                for v in 0..<vCount {
                    let nbs = neighbors[v]
                    guard !nbs.isEmpty else { continue }
                    var avgPos = SIMD3<Float>.zero
                    var avgCol = SIMD3<Float>.zero
                    for nb in nbs {
                        avgPos += chunk.positions[nb]
                        avgCol += SIMD3<Float>(chunk.colors[nb].x, chunk.colors[nb].y, chunk.colors[nb].z)
                    }
                    let invCount = 1.0 / Float(nbs.count)
                    avgPos *= invCount
                    avgCol *= invCount
                    smoothedPos[v] = chunk.positions[v] * (1.0 - lambda) + avgPos * lambda
                    // Mild bilateral color smoothing eliminates seam boundaries across camera views
                    let oldC = SIMD3<Float>(chunk.colors[v].x, chunk.colors[v].y, chunk.colors[v].z)
                    let blendC = oldC * 0.82 + avgCol * 0.18
                    smoothedCol[v] = SIMD4<Float>(blendC.x, blendC.y, blendC.z, chunk.colors[v].w)
                }
                chunk.positions = smoothedPos
                chunk.colors = smoothedCol
            }
            chunks[id] = chunk
        }

        stateLock.lock()
        bakedChunksByAnchor = chunks
        stateLock.unlock()

        DispatchQueue.main.async {
            self.modelRevision += 1
        }
    }

    private func computeModelBoundingBoxMM() {
        stateLock.lock()
        let chunks = Array(bakedChunksByAnchor.values)
        let voxels = Array(spatialVoxelGrid.values)
        stateLock.unlock()

        var minB = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var maxB = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        var count = 0

        for chunk in chunks {
            for p in chunk.positions {
                minB = simd_min(minB, p)
                maxB = simd_max(maxB, p)
                count += 1
            }
        }
        if count == 0 {
            for v in voxels {
                minB = simd_min(minB, v.position)
                maxB = simd_max(maxB, v.position)
                count += 1
            }
        }

        let sizeMM = (count > 0) ? (maxB - minB) * 1000.0 : SIMD3<Float>(0, 0, 0)
        DispatchQueue.main.async {
            self.telemetry.boundingBoxMM = sizeMM
        }
    }

    /// Projects 3D mesh vertices directly into ARFrame.capturedImage sensor space using exact camera.intrinsics
    /// and simd_inverse(camera.transform) — zero aspect-ratio distortion, zero texture tiling!
    private func sampleAndBakeFrameTextures(frame: ARFrame, forceAllAnchors: Bool) {
        let pixelBuffer = frame.capturedImage
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0),
              let cbcrBase = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1) else {
            return
        }

        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let yBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let cbcrBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1)

        let yPtr = yBase.assumingMemoryBound(to: UInt8.self)
        let cbcrPtr = cbcrBase.assumingMemoryBound(to: UInt8.self)

        // Exact world-to-camera transformation & sensor-space camera intrinsics (fx, fy, cx, cy)
        let worldToCamera = simd_inverse(frame.camera.transform)
        let K = frame.camera.intrinsics
        let fx = K[0][0]
        let fy = K[1][1]
        let cx = K[2][0]
        let cy = K[2][1]

        if bestKeyframeImage == nil || forceAllAnchors {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
            let context = CIContext(options: [.useSoftwareRenderer: false])
            if let cgImg = context.createCGImage(ciImage, from: ciImage.extent) {
                let uiImg = UIImage(cgImage: cgImg)
                stateLock.lock()
                bestKeyframeImage = uiImg
                stateLock.unlock()
                DispatchQueue.main.async {
                    self.latestKeyframeSnapshot = uiImg
                }
            }
        }

        let meshAnchors = frame.anchors.compactMap { $0 as? ARMeshAnchor }
        let anchorsToProcess = forceAllAnchors ? meshAnchors : Array(meshAnchors.prefix(14))

        stateLock.lock()
        var existingChunks = bakedChunksByAnchor
        stateLock.unlock()

        for anchor in anchorsToProcess {
            let geom = anchor.geometry
            let vSource = geom.vertices
            let nSource = geom.normals
            let fSource = geom.faces

            let vCount = vSource.count
            guard vCount > 0 else { continue }

            let vBuffer = vSource.buffer.contents()
            let nBuffer = nSource.buffer.contents()
            let fBuffer = fSource.buffer.contents()

            var basePositions = [SIMD3<Float>]()
            var baseNormals = [SIMD3<Float>]()
            basePositions.reserveCapacity(vCount)
            baseNormals.reserveCapacity(vCount)

            for i in 0..<vCount {
                let vOffset = vSource.offset + i * vSource.stride
                let nOffset = nSource.offset + i * nSource.stride
                let localPos = vBuffer.advanced(by: vOffset).assumingMemoryBound(to: SIMD3<Float>.self).pointee
                let localNorm = nBuffer.advanced(by: nOffset).assumingMemoryBound(to: SIMD3<Float>.self).pointee

                let worldPos4 = anchor.transform * SIMD4<Float>(localPos.x, localPos.y, localPos.z, 1.0)
                let worldNorm4 = anchor.transform * SIMD4<Float>(localNorm.x, localNorm.y, localNorm.z, 0.0)
                basePositions.append(SIMD3<Float>(worldPos4.x, worldPos4.y, worldPos4.z))
                baseNormals.append(simd_normalize(SIMD3<Float>(worldNorm4.x, worldNorm4.y, worldNorm4.z)))
            }

            var baseIndices = [UInt32]()
            let indexCount = fSource.count * fSource.indexCountPerPrimitive
            baseIndices.reserveCapacity(indexCount)
            if fSource.bytesPerIndex == 4 {
                let ptr32 = fBuffer.assumingMemoryBound(to: UInt32.self)
                for idx in 0..<indexCount { baseIndices.append(ptr32[idx]) }
            } else if fSource.bytesPerIndex == 2 {
                let ptr16 = fBuffer.assumingMemoryBound(to: UInt16.self)
                for idx in 0..<indexCount { baseIndices.append(UInt32(ptr16[idx])) }
            }

            // Subdivide coarse triangles (1-to-4 midpoint subdivision when vertex count < 6,000)
            // so the mesh has 4x higher spatial color resolution!
            var positions = basePositions
            var normals = baseNormals
            var indices = baseIndices

            if vCount < 4500 && (baseIndices.count / 3) < 8000 {
                var subPos = basePositions
                var subNorm = baseNormals
                var subIdx = [UInt32]()
                subIdx.reserveCapacity(baseIndices.count * 4)
                var edgeMidpointCache = [UInt64: UInt32]()

                func midpointIndex(_ a: UInt32, _ b: UInt32) -> UInt32 {
                    let low = min(a, b)
                    let high = max(a, b)
                    let key = (UInt64(low) << 32) | UInt64(high)
                    if let cached = edgeMidpointCache[key] {
                        return cached
                    }
                    let pa = subPos[Int(a)]
                    let pb = subPos[Int(b)]
                    let na = subNorm[Int(a)]
                    let nb = subNorm[Int(b)]
                    let midP = (pa + pb) * 0.5
                    let midN = simd_normalize(na + nb)
                    let newIdx = UInt32(subPos.count)
                    subPos.append(midP)
                    subNorm.append(midN)
                    edgeMidpointCache[key] = newIdx
                    return newIdx
                }

                let triCount = baseIndices.count / 3
                for t in 0..<triCount {
                    let i0 = baseIndices[t * 3]
                    let i1 = baseIndices[t * 3 + 1]
                    let i2 = baseIndices[t * 3 + 2]
                    guard Int(i0) < vCount, Int(i1) < vCount, Int(i2) < vCount else { continue }
                    let m01 = midpointIndex(i0, i1)
                    let m12 = midpointIndex(i1, i2)
                    let m20 = midpointIndex(i2, i0)

                    subIdx.append(contentsOf: [i0, m01, m20])
                    subIdx.append(contentsOf: [i1, m12, m01])
                    subIdx.append(contentsOf: [i2, m20, m12])
                    subIdx.append(contentsOf: [m01, m12, m20])
                }
                positions = subPos
                normals = subNorm
                indices = subIdx
            }

            let finalVCount = positions.count
            var uvs = [SIMD2<Float>]()
            var colors = [SIMD4<Float>]()
            uvs.reserveCapacity(finalVCount)
            colors.reserveCapacity(finalVCount)

            let previousChunk = existingChunks[anchor.identifier]
            let prevColors = (previousChunk?.positions.count == finalVCount) ? previousChunk?.colors : nil
            let prevUVs = (previousChunk?.positions.count == finalVCount) ? previousChunk?.uvs : nil

            for i in 0..<finalVCount {
                let worldPos = positions[i]
                let worldNorm = normals[i]

                // Transform vertex & normal into ARKit Camera Sensor Space (+X right, +Y up, -Z forward)
                let camPos4 = worldToCamera * SIMD4<Float>(worldPos.x, worldPos.y, worldPos.z, 1.0)
                let camNorm4 = worldToCamera * SIMD4<Float>(worldNorm.x, worldNorm.y, worldNorm.z, 0.0)
                let zDepth = -camPos4.z

                var sampledColor: SIMD3<Float>? = nil
                var sampleWeight: Float = 0.0
                var sampledUV: SIMD2<Float>? = nil

                if zDepth > 0.12 && zDepth <= Float(maxScanDepthMeters) {
                    // Exact pixel coordinate in frame.capturedImage (landscapeRight sensor buffer)
                    let uPx = (fx * camPos4.x / zDepth) + cx
                    let vPx = (fy * (-camPos4.y) / zDepth) + cy

                    let marginX = Float(width) * 0.03
                    let marginY = Float(height) * 0.03

                    if uPx >= marginX && uPx < (Float(width) - marginX) &&
                       vPx >= marginY && vPx < (Float(height) - marginY) {
                        // Incidence angle check: favor camera views looking perpendicular to the surface
                        let viewRay = simd_normalize(SIMD3<Float>(-camPos4.x, -camPos4.y, -camPos4.z))
                        let camNorm = simd_normalize(SIMD3<Float>(camNorm4.x, camNorm4.y, camNorm4.z))
                        let cosTheta = max(0.15, abs(simd_dot(camNorm, viewRay)))

                        let px = Int(uPx)
                        let py = Int(vPx)

                        let yVal = Float(yPtr[py * yBytesPerRow + px])
                        let cbcrOffset = (py / 2) * cbcrBytesPerRow + (px / 2) * 2
                        let cbVal = Float(cbcrPtr[cbcrOffset]) - 128.0
                        let crVal = Float(cbcrPtr[cbcrOffset + 1]) - 128.0

                        let r = clampFloat((yVal + 1.402 * crVal) / 255.0, 0.0, 1.0)
                        let g = clampFloat((yVal - 0.344136 * cbVal - 0.714136 * crVal) / 255.0, 0.0, 1.0)
                        let b = clampFloat((yVal + 1.772 * cbVal) / 255.0, 0.0, 1.0)

                        sampledColor = SIMD3<Float>(r, g, b)
                        sampleWeight = (cosTheta * cosTheta) / max(0.25, zDepth * zDepth)
                        sampledUV = SIMD2<Float>(uPx / Float(width), 1.0 - (vPx / Float(height)))
                    }
                }

                if let newRGB = sampledColor, sampleWeight > 0.01 {
                    if let oldCols = prevColors, i < oldCols.count, oldCols[i].w > 0.01 {
                        let oldW = oldCols[i].w
                        let oldRGB = SIMD3<Float>(oldCols[i].x, oldCols[i].y, oldCols[i].z)
                        let totalW = min(30.0, oldW + sampleWeight)
                        let fusedRGB = (oldRGB * oldW + newRGB * sampleWeight) / totalW
                        colors.append(SIMD4<Float>(fusedRGB.x, fusedRGB.y, fusedRGB.z, totalW))
                    } else {
                        colors.append(SIMD4<Float>(newRGB.x, newRGB.y, newRGB.z, sampleWeight))
                    }
                } else if let oldCols = prevColors, i < oldCols.count {
                    colors.append(oldCols[i])
                } else {
                    let shade = 0.52 + 0.28 * abs(worldNorm.y)
                    colors.append(SIMD4<Float>(shade * 0.82, shade * 0.86, shade * 0.90, 0.0))
                }

                if let newUV = sampledUV {
                    uvs.append(newUV)
                } else if let oldUVs = prevUVs, i < oldUVs.count {
                    uvs.append(oldUVs[i])
                } else {
                    uvs.append(SIMD2<Float>(0.5, 0.5))
                }
            }

            existingChunks[anchor.identifier] = BakedSubmeshChunk(
                anchorIdentifier: anchor.identifier,
                positions: positions,
                normals: normals,
                uvs: uvs,
                colors: colors,
                triangleIndices: indices,
                dedicatedTexture: nil
            )
        }

        stateLock.lock()
        bakedChunksByAnchor = existingChunks
        let chunkCount = bakedChunksByAnchor.count
        stateLock.unlock()

        computeModelBoundingBoxMM()

        DispatchQueue.main.async {
            self.telemetry.bakedTextureCount = chunkCount
            if chunkCount > 0 {
                self.hasBakedModelReady = true
            }
        }
    }

    // MARK: - Standalone 3D Model Builder

    public func buildStandalone3DModelNode(
        shaderMode: PreviewShaderMode,
        roughness: Float,
        metalness: Float
    ) -> SCNNode {
        let rootNode = SCNNode()

        stateLock.lock()
        let chunks = Array(bakedChunksByAnchor.values)
        let voxels = Array(spatialVoxelGrid.values)
        stateLock.unlock()

        if shaderMode == .pointCloud && !voxels.isEmpty {
            var minB = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
            var maxB = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
            for v in voxels {
                minB = simd_min(minB, v.position)
                maxB = simd_max(maxB, v.position)
            }
            let center = (minB + maxB) * 0.5

            let posVecs = voxels.map { SCNVector3($0.position.x - center.x, $0.position.y - center.y, $0.position.z - center.z) }
            let normVecs = voxels.map { SCNVector3($0.normal.x, $0.normal.y, $0.normal.z) }
            let rgbaFloats = voxels.map { SIMD4<Float>($0.color.x, $0.color.y, $0.color.z, 1.0) }

            let posSrc = SCNGeometrySource(vertices: posVecs)
            let normSrc = SCNGeometrySource(normals: normVecs)
            let colData = rgbaFloats.withUnsafeBufferPointer { Data(buffer: $0) }
            let colSrc = SCNGeometrySource(
                data: colData,
                semantic: .color,
                vectorCount: rgbaFloats.count,
                usesFloatComponents: true,
                componentsPerVector: 4,
                bytesPerComponent: 4,
                dataOffset: 0,
                dataStride: 16
            )
            let ptIndices = (0..<UInt32(voxels.count)).map { $0 }
            let idxData = ptIndices.withUnsafeBufferPointer { Data(buffer: $0) }
            let ptElem = SCNGeometryElement(
                data: idxData,
                primitiveType: .point,
                primitiveCount: voxels.count,
                bytesPerIndex: 4
            )
            ptElem.pointSize = 4.0
            ptElem.minimumPointScreenSpaceRadius = 2.0
            ptElem.maximumPointScreenSpaceRadius = 7.0

            let ptGeom = SCNGeometry(sources: [posSrc, normSrc, colSrc], elements: [ptElem])
            let ptMat = SCNMaterial()
            ptMat.lightingModel = .constant
            ptMat.diffuse.contents = UIColor.white
            ptGeom.materials = [ptMat]

            rootNode.addChildNode(SCNNode(geometry: ptGeom))
            return rootNode
        }

        if chunks.isEmpty && !voxels.isEmpty {
            return buildStandalone3DModelNode(shaderMode: .pointCloud, roughness: roughness, metalness: metalness)
        }

        if chunks.isEmpty {
            return rootNode
        }

        var minBound = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
        var maxBound = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
        for chunk in chunks {
            for p in chunk.positions {
                minBound = simd_min(minBound, p)
                maxBound = simd_max(maxBound, p)
            }
        }
        let center = (minBound + maxBound) * 0.5

        for chunk in chunks {
            guard !chunk.positions.isEmpty, !chunk.triangleIndices.isEmpty else { continue }
            let centeredPositions = chunk.positions.map { $0 - center }

            let posSource = SCNGeometrySource(vertices: centeredPositions.map { SCNVector3($0.x, $0.y, $0.z) })
            let normSource = SCNGeometrySource(normals: chunk.normals.map { SCNVector3($0.x, $0.y, $0.z) })
            let uvSource = SCNGeometrySource(textureCoordinates: chunk.uvs.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) })

            // Ensure alpha is 1.0 for SceneKit vertex color rendering
            let displayColors = chunk.colors.map { SIMD4<Float>($0.x, $0.y, $0.z, 1.0) }
            let colorData = displayColors.withUnsafeBufferPointer { Data(buffer: $0) }
            let colorSource = SCNGeometrySource(
                data: colorData,
                semantic: .color,
                vectorCount: displayColors.count,
                usesFloatComponents: true,
                componentsPerVector: 4,
                bytesPerComponent: MemoryLayout<Float>.size,
                dataOffset: 0,
                dataStride: MemoryLayout<SIMD4<Float>>.stride
            )

            let indexData = chunk.triangleIndices.withUnsafeBufferPointer { Data(buffer: $0) }
            let element = SCNGeometryElement(
                data: indexData,
                primitiveType: .triangles,
                primitiveCount: chunk.triangleIndices.count / 3,
                bytesPerIndex: MemoryLayout<UInt32>.size
            )

            let scnGeom = SCNGeometry(sources: [posSource, normSource, uvSource, colorSource], elements: [element])
            let primaryMat = makePreviewMaterial(
                shaderMode: shaderMode,
                roughness: roughness,
                metalness: metalness
            )
            scnGeom.materials = [primaryMat]
            rootNode.addChildNode(SCNNode(geometry: scnGeom))

            if shaderMode == .texturedWireframe {
                let wireGeom = SCNGeometry(sources: [posSource, normSource], elements: [element])
                let wireMat = SCNMaterial()
                wireMat.fillMode = .lines
                wireMat.lightingModel = .constant
                wireMat.diffuse.contents = UIColor.systemCyan.withAlphaComponent(0.55)
                wireMat.isDoubleSided = true
                wireGeom.materials = [wireMat]
                rootNode.addChildNode(SCNNode(geometry: wireGeom))
            }
        }

        return rootNode
    }

    private func makePreviewMaterial(
        shaderMode: PreviewShaderMode,
        roughness: Float,
        metalness: Float
    ) -> SCNMaterial {
        let mat = SCNMaterial()
        mat.isDoubleSided = true

        switch shaderMode {
        case .bakedTexture, .texturedWireframe:
            mat.lightingModel = .physicallyBased
            // CRITICAL FIX: Set diffuse.contents = UIColor.white so SceneKit renders the exact
            // multi-view fused per-vertex sRGB colors without stretching a single 2D photo across the whole room!
            mat.diffuse.contents = UIColor.white
            mat.roughness.contents = CGFloat(roughness)
            mat.metalness.contents = CGFloat(metalness)

        case .clayMatCap:
            mat.lightingModel = .physicallyBased
            mat.diffuse.contents = UIColor(white: 0.88, alpha: 1.0)
            mat.roughness.contents = CGFloat(0.38)
            mat.metalness.contents = CGFloat(0.10)

        case .pointCloud:
            mat.lightingModel = .constant
            mat.diffuse.contents = UIColor.white
        }
        return mat
    }

    public func exportScannedMeshPackage() -> [URL] {
        stateLock.lock()
        let chunks = Array(bakedChunksByAnchor.values)
        let voxels = Array(spatialVoxelGrid.values)
        let texImage = bestKeyframeImage ?? tdLatestTextureImage
        stateLock.unlock()

        let tempDir = FileManager.default.temporaryDirectory
        let objURL = tempDir.appendingPathComponent("ArgonScan_iPhone13PM.obj")
        let mtlURL = tempDir.appendingPathComponent("ArgonScan_iPhone13PM.mtl")
        let jpgURL = tempDir.appendingPathComponent("ArgonScan_Texture.jpg")

        var exportedURLs = [URL]()

        if let img = texImage, let jpgData = img.jpegData(compressionQuality: 0.92) {
            try? jpgData.write(to: jpgURL, options: .atomic)
            exportedURLs.append(jpgURL)
        }

        let mtlContent = """
        # ArgonScan Pro v2.1 Material Library
        newmtl BakedCameraMaterial
        Ka 1.000 1.000 1.000
        Kd 1.000 1.000 1.000
        Ks 0.200 0.200 0.200
        Ns 50.0
        d 1.0
        illum 2
        """
        try? mtlContent.write(to: mtlURL, atomically: true, encoding: .utf8)
        exportedURLs.append(mtlURL)

        var objLines = [String]()
        objLines.append("# ArgonScan Pro v2.1 (iPhone 13 Pro Max) High-Precision 3D Scan")
        objLines.append("mtllib ArgonScan_iPhone13PM.mtl")
        objLines.append("usemtl BakedCameraMaterial")

        if !chunks.isEmpty {
            var vertexOffset: UInt32 = 1
            for chunk in chunks {
                for i in 0..<chunk.positions.count {
                    let p = chunk.positions[i]
                    let c = chunk.colors[i]
                    objLines.append(String(format: "v %.5f %.5f %.5f %.4f %.4f %.4f", p.x, p.y, p.z, c.x, c.y, c.z))
                }
                for uv in chunk.uvs {
                    objLines.append(String(format: "vt %.5f %.5f", uv.x, uv.y))
                }
                for n in chunk.normals {
                    objLines.append(String(format: "vn %.4f %.4f %.4f", n.x, n.y, n.z))
                }
                let triCount = chunk.triangleIndices.count / 3
                for t in 0..<triCount {
                    let i0 = chunk.triangleIndices[t * 3] + vertexOffset
                    let i1 = chunk.triangleIndices[t * 3 + 1] + vertexOffset
                    let i2 = chunk.triangleIndices[t * 3 + 2] + vertexOffset
                    objLines.append("f \(i0)/\(i0)/\(i0) \(i1)/\(i1)/\(i1) \(i2)/\(i2)/\(i2)")
                }
                vertexOffset += UInt32(chunk.positions.count)
            }
        } else {
            for v in voxels {
                let p = v.position
                let c = v.color
                objLines.append(String(format: "v %.5f %.5f %.5f %.4f %.4f %.4f", p.x, p.y, p.z, c.x, c.y, c.z))
            }
        }

        let objString = objLines.joined(separator: "\n")
        try? objString.write(to: objURL, atomically: true, encoding: .utf8)
        exportedURLs.insert(objURL, at: 0)
        return exportedURLs
    }

    private func clampFloat(_ val: Float, _ low: Float, _ high: Float) -> Float {
        min(high, max(low, val))
    }

    private func updateStatus(_ message: String) {
        DispatchQueue.main.async {
            self.telemetry.statusMessage = message
        }
    }

    private func currentMemoryUsageMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return Double(info.resident_size) / (1024.0 * 1024.0)
        }
        return 380.0
    }
}

// MARK: - ARSessionDelegate (Rear dToF LiDAR + 12MP Wide Camera)

extension HybridSensorFusionManager: ARSessionDelegate {
    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = frame.timestamp

        stateLock.lock()
        let omega = latestOmegaRadSec
        stateLock.unlock()

        if isCapturingMesh && omega < 0.40 && (now - lastMetalDepthPassTime >= 0.14) {
            lastMetalDepthPassTime = now
            executeMetalDepthSuperResolutionAndFusion(frame: frame)
        }

        let meshAnchors = frame.anchors.compactMap { $0 as? ARMeshAnchor }
        if isCapturingMesh && omega < 0.38 && (now - lastTextureSampleTime >= 0.35) && !meshAnchors.isEmpty {
            lastTextureSampleTime = now
            sampleAndBakeFrameTextures(frame: frame, forceAllAnchors: false)
        }

        guard now - lastTelemetryUpdateTime >= 0.20 else { return }
        lastTelemetryUpdateTime = now

        var totalVertexCount = 0
        var totalFaceCount = 0
        for anchor in meshAnchors {
            totalVertexCount += anchor.geometry.vertices.count
            totalFaceCount += anchor.geometry.faces.count
        }

        let ramMB = currentMemoryUsageMB()
        let addKeyframe = (isCapturingMesh && omega < 0.32 && frame.sceneDepth != nil) ? 1 : 0

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.telemetry.imuAngularVelocityRadSec = omega
            self.telemetry.memoryUsedMB = ramMB
            self.telemetry.activeVerticesCount = totalVertexCount
            self.telemetry.activeMeshFacesCount = totalFaceCount
            self.telemetry.keyframesCaptured += addKeyframe
        }
    }
}

// MARK: - AVCaptureDataOutputSynchronizerDelegate (Front TrueDepth 640x480 Object Scanner & Live Mesh Builder)

extension HybridSensorFusionManager: AVCaptureDataOutputSynchronizerDelegate {
    public func dataOutputSynchronizer(
        _ synchronizer: AVCaptureDataOutputSynchronizer,
        didOutput synchronizedDataCollection: AVCaptureSynchronizedDataCollection
    ) {
        guard let vOutput = videoDataOutput,
              let dOutput = depthDataOutput,
              let syncedVideo = synchronizedDataCollection.synchronizedData(for: vOutput) as? AVCaptureSynchronizedSampleBufferData,
              let syncedDepth = synchronizedDataCollection.synchronizedData(for: dOutput) as? AVCaptureSynchronizedDepthData,
              !syncedVideo.sampleBufferWasDropped,
              !syncedDepth.depthDataWasDropped,
              let videoPixelBuffer = CMSampleBufferGetImageBuffer(syncedVideo.sampleBuffer) else {
            return
        }

        let depthData = syncedDepth.depthData.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
        let depthBuffer = depthData.depthDataMap

        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(videoPixelBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(videoPixelBuffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
        }

        guard let depthBase = CVPixelBufferGetBaseAddress(depthBuffer),
              CVPixelBufferGetPlaneCount(videoPixelBuffer) >= 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(videoPixelBuffer, 0),
              let cbcrBase = CVPixelBufferGetBaseAddressOfPlane(videoPixelBuffer, 1),
              let calibration = depthData.cameraCalibrationData else {
            return
        }

        let dWidth = CVPixelBufferGetWidth(depthBuffer)
        let dHeight = CVPixelBufferGetHeight(depthBuffer)
        let dFloatsPerRow = CVPixelBufferGetBytesPerRow(depthBuffer) / MemoryLayout<Float>.stride
        let depthPtr = depthBase.assumingMemoryBound(to: Float.self)

        // Always measure center target distance (cm) for live TrueDepth range guidance
        let centerZ = depthPtr[(dHeight / 2) * dFloatsPerRow + (dWidth / 2)]
        let centerCM = (!centerZ.isNaN && centerZ > 0.05 && centerZ < 2.0) ? Double(centerZ * 100.0) : 0.0

        guard isCapturingMesh else {
            let now = CACurrentMediaTime()
            if now - lastTrueDepthFrameTime >= 0.25 {
                lastTrueDepthFrameTime = now
                DispatchQueue.main.async {
                    self.telemetry.trueDepthCenterDistanceCM = centerCM
                }
            }
            return
        }

        let vWidth = CVPixelBufferGetWidthOfPlane(videoPixelBuffer, 0)
        let vHeight = CVPixelBufferGetHeightOfPlane(videoPixelBuffer, 0)
        let yBPR = CVPixelBufferGetBytesPerRowOfPlane(videoPixelBuffer, 0)
        let cbcrBPR = CVPixelBufferGetBytesPerRowOfPlane(videoPixelBuffer, 1)
        let yPtr = yBase.assumingMemoryBound(to: UInt8.self)
        let cbcrPtr = cbcrBase.assumingMemoryBound(to: UInt8.self)

        let refDim = calibration.intrinsicMatrixReferenceDimensions
        let scaleX = Float(dWidth) / Float(max(1.0, refDim.width))
        let scaleY = Float(dHeight) / Float(max(1.0, refDim.height))
        let fx = calibration.intrinsicMatrix[0][0] * scaleX
        let fy = calibration.intrinsicMatrix[1][1] * scaleY
        let cx = calibration.intrinsicMatrix[2][0] * scaleX
        let cy = calibration.intrinsicMatrix[2][1] * scaleY

        let pitchMeters = Float(max(0.0008, voxelPitchMM / 1000.0))
        let maxD = Float(maxScanDepthMeters)

        stateLock.lock()
        tdLatestIntrinsics = calibration.intrinsicMatrix
        tdLatestRefDims = refDim

        // Accumulate into 320x240 organized grid for high-density 3D surface mesh triangulation
        for gy in 0..<tdGridH {
            let y = min(dHeight - 1, gy * dHeight / tdGridH)
            let vy = min(vHeight - 1, gy * vHeight / tdGridH)
            for gx in 0..<tdGridW {
                let x = min(dWidth - 1, gx * dWidth / tdGridW)
                let z = depthPtr[y * dFloatsPerRow + x]
                guard !z.isNaN, z >= 0.12, z <= maxD else { continue }

                let vx = min(vWidth - 1, gx * vWidth / tdGridW)
                let yVal = Float(yPtr[vy * yBPR + vx])
                let cbcrOff = (vy / 2) * cbcrBPR + (vx / 2) * 2
                let cb = Float(cbcrPtr[cbcrOff]) - 128.0
                let cr = Float(cbcrPtr[cbcrOff + 1]) - 128.0

                let r = clampFloat((yVal + 1.402 * cr) / 255.0, 0.0, 1.0)
                let g = clampFloat((yVal - 0.344136 * cb - 0.714136 * cr) / 255.0, 0.0, 1.0)
                let b = clampFloat((yVal + 1.772 * cb) / 255.0, 0.0, 1.0)
                let rgb = SIMD3<Float>(r, g, b)

                let gIdx = gy * tdGridW + gx
                let oldW = tdAccumWeight[gIdx]
                if oldW > 0.01 && abs(tdAccumDepth[gIdx] - z) < 0.025 {
                    let newW = min(32.0, oldW + 1.0)
                    tdAccumDepth[gIdx] = (tdAccumDepth[gIdx] * oldW + z) / newW
                    tdAccumColor[gIdx] = (tdAccumColor[gIdx] * oldW + rgb) / newW
                    tdAccumWeight[gIdx] = newW
                } else {
                    tdAccumDepth[gIdx] = z
                    tdAccumColor[gIdx] = rgb
                    tdAccumWeight[gIdx] = 1.0
                }

                // Also populate voxel point cloud every 2nd grid cell
                if (gx % 2 == 0) && (gy % 2 == 0) && spatialVoxelGrid.count < 180_000 {
                    let x3D = -(Float(x) - cx) * z / fx
                    let y3D = -(Float(y) - cy) * z / fy
                    let pos = SIMD3<Float>(x3D, y3D, -z)
                    let key = SpatialVoxelKey(
                        ix: Int32(floor(pos.x / pitchMeters)),
                        iy: Int32(floor(pos.y / pitchMeters)),
                        iz: Int32(floor(pos.z / pitchMeters))
                    )
                    spatialVoxelGrid[key] = FusedSurfacePoint(
                        position: pos,
                        normal: SIMD3<Float>(0, 0, 1),
                        color: rgb,
                        uv: SIMD2<Float>(Float(gx) / Float(tdGridW), Float(gy) / Float(tdGridH)),
                        weight: 1.0
                    )
                }
            }
        }
        let voxCount = spatialVoxelGrid.count
        stateLock.unlock()

        let now = CACurrentMediaTime()
        if now - lastTrueDepthFrameTime >= 0.45 {
            lastTrueDepthFrameTime = now
            let ciImg = CIImage(cvPixelBuffer: videoPixelBuffer).oriented(.leftMirrored)
            let ctx = CIContext(options: [.useSoftwareRenderer: false])
            if let cg = ctx.createCGImage(ciImg, from: ciImg.extent) {
                let uiImg = UIImage(cgImage: cg)
                stateLock.lock()
                tdLatestTextureImage = uiImg
                stateLock.unlock()
                DispatchQueue.main.async {
                    self.latestKeyframeSnapshot = uiImg
                }
            }
            // Periodically triangulate the TrueDepth grid so live vertex/face counts update during scanning!
            triangulateTrueDepthAccumulatedGrid()
            computeModelBoundingBoxMM()

            DispatchQueue.main.async {
                self.telemetry.trueDepthCenterDistanceCM = centerCM
                self.telemetry.denseVoxelPointsCount = voxCount
                self.telemetry.keyframesCaptured += 1
            }
        }
    }
}
