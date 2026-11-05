//
//  ProMotionScannerView.swift
//  ArgonScan Pro v2.1 — Auto-Layout TrueDepth & LiDAR Preview + Zero-Jitter 3D Model Studio
//

import SwiftUI
import ARKit
import SceneKit
import AVFoundation

/// Custom UIView that automatically resizes AVCaptureVideoPreviewLayer on layoutSubviews()
/// so Front TrueDepth camera preview is always visible full-screen in Near-Field mode!
public final class TrueDepthPreviewContainerView: UIView {
    public let previewLayer: AVCaptureVideoPreviewLayer

    public init(session: AVCaptureSession) {
        self.previewLayer = AVCaptureVideoPreviewLayer(session: session)
        super.init(frame: UIScreen.main.bounds)
        self.backgroundColor = .black
        self.previewLayer.videoGravity = .resizeAspectFill
        self.layer.addSublayer(self.previewLayer)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    public override func layoutSubviews() {
        super.layoutSubviews()
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        previewLayer.frame = self.bounds
        CATransaction.commit()
    }
}

public struct LiveARMeshPreview: UIViewRepresentable {
    @ObservedObject public var fusionManager: HybridSensorFusionManager
    public let showWireframe: Bool

    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    public func makeUIView(context: Context) -> UIView {
        let container = UIView(frame: UIScreen.main.bounds)
        container.backgroundColor = .black

        let scnView = ARSCNView(frame: container.bounds)
        scnView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        scnView.scene = SCNScene()
        scnView.session = fusionManager.arSession
        scnView.delegate = context.coordinator
        scnView.automaticallyUpdatesLighting = true
        scnView.preferredFramesPerSecond = 120
        scnView.rendersCameraGrain = false
        scnView.rendersMotionBlur = false
        scnView.debugOptions = [.showFeaturePoints]
        scnView.backgroundColor = .clear
        scnView.tag = 101
        container.addSubview(scnView)

        let tdView = TrueDepthPreviewContainerView(session: fusionManager.trueDepthAVSession)
        tdView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        tdView.isHidden = true
        tdView.tag = 102
        container.addSubview(tdView)

        return container
    }

    public func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.showWireframe = showWireframe
        context.coordinator.liveTextureMapped = fusionManager.liveTextureOverlayEnabled

        let isTrueDepth = (fusionManager.activeMode == .nearFieldTrueDepthMacro)
        if let scnView = uiView.viewWithTag(101) {
            scnView.isHidden = isTrueDepth
        }
        if let tdView = uiView.viewWithTag(102) as? TrueDepthPreviewContainerView {
            tdView.isHidden = !isTrueDepth
            tdView.setNeedsLayout()
        }
    }

    public final class Coordinator: NSObject, ARSCNViewDelegate {
        public var showWireframe: Bool = true
        public var liveTextureMapped: Bool = true

        public func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
            guard let meshAnchor = anchor as? ARMeshAnchor else { return nil }
            let geom = createSCNGeometry(from: meshAnchor.geometry)
            return SCNNode(geometry: geom)
        }

        public func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            guard let meshAnchor = anchor as? ARMeshAnchor else { return }
            node.geometry = createSCNGeometry(from: meshAnchor.geometry)
        }

        private func createSCNGeometry(from meshGeometry: ARMeshGeometry) -> SCNGeometry {
            let vertices = meshGeometry.vertices
            let normals = meshGeometry.normals
            let faces = meshGeometry.faces

            let vertexSource = SCNGeometrySource(
                buffer: vertices.buffer,
                vertexFormat: vertices.format,
                semantic: .vertex,
                vertexCount: vertices.count,
                dataOffset: vertices.offset,
                dataStride: vertices.stride
            )
            let normalSource = SCNGeometrySource(
                buffer: normals.buffer,
                vertexFormat: normals.format,
                semantic: .normal,
                vertexCount: normals.count,
                dataOffset: normals.offset,
                dataStride: normals.stride
            )
            let faceData = Data(bytesNoCopy: faces.buffer.contents(), count: faces.buffer.length, deallocator: .none)
            let geometryElement = SCNGeometryElement(
                data: faceData,
                primitiveType: .triangles,
                primitiveCount: faces.count,
                bytesPerIndex: faces.bytesPerIndex
            )

            let scnGeom = SCNGeometry(sources: [vertexSource, normalSource], elements: [geometryElement])
            let material = SCNMaterial()
            material.isDoubleSided = true

            if showWireframe {
                material.fillMode = .lines
                material.diffuse.contents = liveTextureMapped
                    ? UIColor.systemGreen.withAlphaComponent(0.85)
                    : UIColor.systemCyan.withAlphaComponent(0.80)
            } else {
                material.fillMode = .fill
                material.lightingModel = .physicallyBased
                material.diffuse.contents = UIColor.systemCyan.withAlphaComponent(0.38)
            }
            scnGeom.materials = [material]
            return scnGeom
        }
    }
}

public struct ProMotionScannerView: View {
    @StateObject private var fusionManager = HybridSensorFusionManager()
    @State private var showWireframe: Bool = true
    @State private var show3DPreviewStudio: Bool = false
    @State private var exportedURLs: [URL] = []
    @State private var showShareSheet: Bool = false

    public init() {}

    public var body: some View {
        ZStack {
            LiveARMeshPreview(fusionManager: fusionManager, showWireframe: showWireframe)
                .ignoresSafeArea()

            // Center Reticle & Range Guide for Near-Field TrueDepth Mode
            if fusionManager.activeMode == .nearFieldTrueDepthMacro {
                VStack(spacing: 8) {
                    RoundedRectangle(cornerRadius: 20)
                        .strokeBorder(
                            (fusionManager.telemetry.trueDepthCenterDistanceCM >= 14.0 &&
                             fusionManager.telemetry.trueDepthCenterDistanceCM <= 45.0)
                                ? Color.green : Color.orange,
                            style: StrokeStyle(lineWidth: 2.5, dash: [8, 6])
                        )
                        .frame(width: 240, height: 240)

                    Text(
                        fusionManager.telemetry.trueDepthCenterDistanceCM > 1.0
                            ? String(format: "当前中心物距: %.1f cm (最佳 15–45 cm)", fusionManager.telemetry.trueDepthCenterDistanceCM)
                            : "请将待扫物体移入取景框 (距离屏幕 15–45 cm)"
                    )
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundColor(.green)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.75))
                    .clipShape(Capsule())
                }
                .allowsHitTesting(false)
            }

            VStack(spacing: 8) {
                // Top Hardware Telemetry Bar
                HStack(spacing: 10) {
                    Label(String(format: "GPU %.1fms", fusionManager.telemetry.gpuFrameTimeMs), systemImage: "cpu")
                    Label("网格面: \(fusionManager.telemetry.activeMeshFacesCount)", systemImage: "triangle")
                    Label("网格顶: \(fusionManager.telemetry.activeVerticesCount)", systemImage: "cube.transparent")
                    Label(String(format: "%.0fMB", fusionManager.telemetry.memoryUsedMB), systemImage: "memorychip")
                }
                .font(.system(size: 10.5, weight: .semibold, design: .monospaced))
                .lineLimit(1)
                .minimumScaleFactor(0.70)
                .foregroundColor(.cyan)
                .padding(.horizontal, 12)
                .padding(.vertical, 7)
                .background(Color.black.opacity(0.78))
                .clipShape(Capsule())

                // Physical Bounding Box Dimensions & Status Banner
                VStack(spacing: 3) {
                    HStack(spacing: 8) {
                        if let snap = fusionManager.latestKeyframeSnapshot {
                            Image(uiImage: snap)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 24, height: 24)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.cyan, lineWidth: 1))
                        }
                        Text(fusionManager.telemetry.statusMessage)
                            .font(.system(size: 11.5, weight: .medium))
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                            .foregroundColor(.white)
                    }
                    if fusionManager.telemetry.boundingBoxMM.x > 1.0 {
                        Text(String(
                            format: "物理包围盒尺寸: %.0f × %.0f × %.0f mm · 已烘焙网格块: %d",
                            fusionManager.telemetry.boundingBoxMM.x,
                            fusionManager.telemetry.boundingBoxMM.y,
                            fusionManager.telemetry.boundingBoxMM.z,
                            fusionManager.telemetry.bakedTextureCount
                        ))
                        .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                        .foregroundColor(.green)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.70))
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .padding(.horizontal, 14)

                Spacer()

                // Segmented Sensor Mode Picker
                Picker("Scan Mode", selection: $fusionManager.activeMode) {
                    ForEach(ScanRangeMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 14)
                .onChange(of: fusionManager.activeMode) { newMode in
                    fusionManager.switchMode(newMode)
                }

                // Bottom Control Card
                VStack(spacing: 10) {
                    HStack(spacing: 12) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("体素精度: \(String(format: "%.1f", fusionManager.voxelPitchMM)) mm")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.9))
                            Slider(value: $fusionManager.voxelPitchMM, in: 0.5...5.0, step: 0.5)
                                .tint(.cyan)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            Text("测距门限: \(String(format: "%.2f", fusionManager.maxScanDepthMeters)) m")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.9))
                            Slider(value: $fusionManager.maxScanDepthMeters, in: 0.25...5.0, step: 0.15)
                                .tint(.green)
                        }

                        Button(action: { showWireframe.toggle() }) {
                            Text(showWireframe ? "线框" : "实体")
                                .font(.caption.bold())
                                .foregroundColor(showWireframe ? .black : .white)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 8)
                                .background(showWireframe ? Color.cyan : Color.white.opacity(0.18))
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }

                    HStack(spacing: 10) {
                        Button(action: {
                            fusionManager.toggleCapture()
                        }) {
                            HStack(spacing: 5) {
                                Image(systemName: fusionManager.isCapturingMesh ? "pause.circle.fill" : "viewfinder")
                                Text(fusionManager.isCapturingMesh ? "暂停采集" : "Start Capture")
                                    .lineLimit(1)
                            }
                            .font(.subheadline.bold())
                            .foregroundColor(fusionManager.isCapturingMesh ? .white : .black)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 13)
                            .background(fusionManager.isCapturingMesh ? Color.red : Color.cyan)
                            .clipShape(Capsule())
                        }

                        Button(action: {
                            fusionManager.bakeCurrentMeshAndTextures {
                                show3DPreviewStudio = true
                            }
                        }) {
                            HStack(spacing: 5) {
                                Image(systemName: "cube.fill")
                                Text(fusionManager.isBakingInProgress ? "正在生成网格..." : "烘焙并 3D 预览")
                                    .lineLimit(1)
                            }
                            .font(.subheadline.bold())
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 13)
                            .background(
                                LinearGradient(
                                    colors: [Color.purple, Color.indigo],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .clipShape(Capsule())
                        }
                    }

                    HStack(spacing: 10) {
                        Button(action: {
                            fusionManager.bakeCurrentMeshAndTextures {
                                show3DPreviewStudio = true
                            }
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: "arkit")
                                Text("独立 3D 预览室")
                                    .lineLimit(1)
                            }
                            .font(.caption.bold())
                            .foregroundColor(.cyan)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 9)
                            .background(Color.white.opacity(0.14))
                            .clipShape(Capsule())
                        }

                        Button(action: {
                            exportedURLs = fusionManager.exportScannedMeshPackage()
                            if !exportedURLs.isEmpty {
                                showShareSheet = true
                            }
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: "square.and.arrow.up")
                                Text("导出 OBJ+MTL+贴图")
                                    .lineLimit(1)
                            }
                            .font(.caption.bold())
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 9)
                            .background(Color.white.opacity(0.14))
                            .clipShape(Capsule())
                        }

                        Button(action: {
                            fusionManager.clearScanAndReset()
                        }) {
                            Image(systemName: "trash")
                                .font(.caption.bold())
                                .foregroundColor(.red)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 9)
                                .background(Color.white.opacity(0.14))
                                .clipShape(Capsule())
                        }
                    }
                }
                .padding(14)
                .background(Color.black.opacity(0.84))
                .clipShape(RoundedRectangle(cornerRadius: 22))
                .padding(.horizontal, 14)
            }
            .padding(.vertical, 6)
        }
        .onAppear {
            fusionManager.ensureCameraPreviewRunning()
        }
        .fullScreenCover(isPresented: $show3DPreviewStudio) {
            Model3DPreviewStudioView(fusionManager: fusionManager)
        }
        .sheet(isPresented: $showShareSheet) {
            if !exportedURLs.isEmpty {
                ShareSheetView(activityItems: exportedURLs)
            }
        }
    }
}

// MARK: - Zero-Jitter Standalone 3D Model Preview Studio

public struct StandaloneSCNModelViewer: UIViewRepresentable {
    public let fusionManager: HybridSensorFusionManager
    public let shaderMode: PreviewShaderMode
    public let roughness: Float
    public let metalness: Float
    public let autoRotate: Bool
    public let showGridFloor: Bool
    public let modelRevision: Int

    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    public final class Coordinator {
        public var lastLoadedRevision: Int = -1
        public var lastLoadedShaderMode: PreviewShaderMode? = nil
        public var lastAutoRotate: Bool = false
    }

    public func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView(frame: .zero)
        let scene = SCNScene()
        scnView.scene = scene
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = false
        scnView.preferredFramesPerSecond = 120
        scnView.antialiasingMode = .multisampling4X
        scnView.backgroundColor = UIColor(red: 0.04, green: 0.06, blue: 0.10, alpha: 1.0)

        let keyLightNode = SCNNode()
        let keyLight = SCNLight()
        keyLight.type = .directional
        keyLight.intensity = 1100
        keyLight.temperature = 6200
        keyLightNode.light = keyLight
        keyLightNode.eulerAngles = SCNVector3(-Float.pi / 3, Float.pi / 4, 0)
        scene.rootNode.addChildNode(keyLightNode)

        let fillLightNode = SCNNode()
        let fillLight = SCNLight()
        fillLight.type = .omni
        fillLight.intensity = 450
        fillLight.temperature = 5500
        fillLightNode.light = fillLight
        fillLightNode.position = SCNVector3(-1.5, 1.0, 1.8)
        scene.rootNode.addChildNode(fillLightNode)

        let ambientNode = SCNNode()
        let ambientLight = SCNLight()
        ambientLight.type = .ambient
        ambientLight.intensity = 420
        ambientLight.color = UIColor(white: 0.92, alpha: 1.0)
        ambientNode.light = ambientLight
        scene.rootNode.addChildNode(ambientNode)

        let cameraNode = SCNNode()
        cameraNode.name = "studioCamera"
        let camera = SCNCamera()
        camera.zNear = 0.005
        camera.zFar = 50.0
        camera.fieldOfView = 46.0
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, 0.15, 0.85)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(cameraNode)

        let floorNode = createStudioGridNode()
        floorNode.name = "studioFloor"
        scene.rootNode.addChildNode(floorNode)

        let containerNode = SCNNode()
        containerNode.name = "scannedModelContainer"
        scene.rootNode.addChildNode(containerNode)

        return scnView
    }

    public func updateUIView(_ scnView: SCNView, context: Context) {
        guard let scene = scnView.scene else { return }

        if let floorNode = scene.rootNode.childNode(withName: "studioFloor", recursively: false) {
            floorNode.isHidden = !showGridFloor
        }

        // CRITICAL FIX: Only rebuild the 3D geometry when modelRevision or shaderMode actually changes!
        // Never rebuild on background telemetry updates or slider drags -> eliminates all camera jitter!
        let needsGeometryRebuild = (context.coordinator.lastLoadedRevision != modelRevision) ||
                                   (context.coordinator.lastLoadedShaderMode != shaderMode)

        if needsGeometryRebuild {
            context.coordinator.lastLoadedRevision = modelRevision
            context.coordinator.lastLoadedShaderMode = shaderMode

            if let oldContainer = scene.rootNode.childNode(withName: "scannedModelContainer", recursively: false) {
                oldContainer.removeFromParentNode()
            }

            let modelNode = fusionManager.buildStandalone3DModelNode(
                shaderMode: shaderMode,
                roughness: roughness,
                metalness: metalness
            )
            modelNode.name = "scannedModelContainer"
            scene.rootNode.addChildNode(modelNode)
            context.coordinator.lastAutoRotate = false
        }

        // Update PBR material roughness/metalness and turntable rotation in-place without rebuilding geometry
        if let container = scene.rootNode.childNode(withName: "scannedModelContainer", recursively: false) {
            container.enumerateChildNodes { child, _ in
                if let mats = child.geometry?.materials {
                    for mat in mats where mat.lightingModel == .physicallyBased {
                        mat.roughness.contents = CGFloat(roughness)
                        mat.metalness.contents = CGFloat(metalness)
                    }
                }
            }

            if context.coordinator.lastAutoRotate != autoRotate {
                context.coordinator.lastAutoRotate = autoRotate
                container.removeAction(forKey: "turntableSpin")
                if autoRotate {
                    let spin = SCNAction.repeatForever(SCNAction.rotateBy(x: 0, y: CGFloat.pi * 2, z: 0, duration: 16.0))
                    container.runAction(spin, forKey: "turntableSpin")
                }
            }
        }
    }

    private func createStudioGridNode() -> SCNNode {
        let gridRoot = SCNNode()
        let lineMat = SCNMaterial()
        lineMat.lightingModel = .constant
        lineMat.diffuse.contents = UIColor.systemCyan.withAlphaComponent(0.22)

        let extent: Float = 0.8
        let steps = 16
        let stepSize = (extent * 2.0) / Float(steps)

        for i in 0...steps {
            let offset = -extent + Float(i) * stepSize
            let boxX = SCNBox(width: CGFloat(extent * 2.0), height: 0.001, length: 0.0015, chamferRadius: 0)
            boxX.materials = [lineMat]
            let nodeX = SCNNode(geometry: boxX)
            nodeX.position = SCNVector3(0, -0.25, offset)
            gridRoot.addChildNode(nodeX)

            let boxZ = SCNBox(width: 0.0015, height: 0.001, length: CGFloat(extent * 2.0), chamferRadius: 0)
            boxZ.materials = [lineMat]
            let nodeZ = SCNNode(geometry: boxZ)
            nodeZ.position = SCNVector3(offset, -0.25, 0)
            gridRoot.addChildNode(nodeZ)
        }
        return gridRoot
    }
}

public struct Model3DPreviewStudioView: View {
    @ObservedObject public var fusionManager: HybridSensorFusionManager
    @Environment(\.dismiss) private var dismiss

    @State private var shaderMode: PreviewShaderMode = .bakedTexture
    @State private var roughness: Float = 0.30
    @State private var metalness: Float = 0.12
    // Default autoRotate to false so manual 360 orbit camera control is completely steady
    @State private var autoRotate: Bool = false
    @State private var showGridFloor: Bool = true
    @State private var exportedURLs: [URL] = []
    @State private var showShareSheet: Bool = false

    public init(fusionManager: HybridSensorFusionManager) {
        self.fusionManager = fusionManager
    }

    public var body: some View {
        ZStack {
            Color(red: 0.04, green: 0.06, blue: 0.10).ignoresSafeArea()

            StandaloneSCNModelViewer(
                fusionManager: fusionManager,
                shaderMode: shaderMode,
                roughness: roughness,
                metalness: metalness,
                autoRotate: autoRotate,
                showGridFloor: showGridFloor,
                modelRevision: fusionManager.modelRevision
            )
            .ignoresSafeArea()

            VStack(spacing: 10) {
                HStack {
                    Button(action: { dismiss() }) {
                        HStack(spacing: 4) {
                            Image(systemName: "chevron.left")
                            Text("返回相机扫描")
                        }
                        .font(.subheadline.bold())
                        .foregroundColor(.cyan)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(Color.black.opacity(0.78))
                        .clipShape(Capsule())
                    }

                    Spacer()

                    Text("3D 模型独立预览室")
                        .font(.subheadline.bold())
                        .foregroundColor(.white)

                    Spacer()

                    Button(action: { autoRotate.toggle() }) {
                        Image(systemName: autoRotate ? "arrow.triangle.2.circlepath.circle.fill" : "arrow.triangle.2.circlepath.circle")
                            .font(.title3)
                            .foregroundColor(autoRotate ? .cyan : .gray)
                            .padding(8)
                            .background(Color.black.opacity(0.78))
                            .clipShape(Circle())
                    }
                }
                .padding(.horizontal, 16)

                HStack(spacing: 12) {
                    if let thumb = fusionManager.latestKeyframeSnapshot {
                        Image(uiImage: thumb)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 36, height: 36)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.cyan, lineWidth: 1))
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text("网格面: \(fusionManager.telemetry.activeMeshFacesCount) · 网格顶: \(fusionManager.telemetry.activeVerticesCount)")
                            .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            .foregroundColor(.cyan)
                        if fusionManager.telemetry.boundingBoxMM.x > 1.0 {
                            Text(String(
                                format: "实测尺寸: %.0f × %.0f × %.0f mm",
                                fusionManager.telemetry.boundingBoxMM.x,
                                fusionManager.telemetry.boundingBoxMM.y,
                                fusionManager.telemetry.boundingBoxMM.z
                            ))
                            .font(.system(size: 10.5, weight: .bold, design: .monospaced))
                            .foregroundColor(.green)
                        } else {
                            Text("单指拖拽 360° 旋转 · 双指平移 · 捏合缩放")
                                .font(.system(size: 10))
                                .foregroundColor(.white.opacity(0.75))
                        }
                    }
                    Spacer()
                }
                .padding(10)
                .background(Color.black.opacity(0.74))
                .clipShape(RoundedRectangle(cornerRadius: 14))
                .padding(.horizontal, 16)

                Spacer()

                VStack(spacing: 12) {
                    Picker("渲染材质模式", selection: $shaderMode) {
                        ForEach(PreviewShaderMode.allCases, id: \.self) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)

                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("PBR 粗糙度: \(String(format: "%.2f", roughness))")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.85))
                            Slider(value: $roughness, in: 0.05...1.0)
                                .tint(.cyan)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            Text("PBR 金属度: \(String(format: "%.2f", metalness))")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.85))
                            Slider(value: $metalness, in: 0.0...1.0)
                                .tint(.indigo)
                        }
                    }

                    HStack(spacing: 10) {
                        Button(action: {
                            fusionManager.applyLaplacianMeshSmoothing(iterations: 2, lambda: 0.40)
                        }) {
                            HStack(spacing: 5) {
                                Image(systemName: "wand.and.stars")
                                Text("网格去噪平滑")
                                    .lineLimit(1)
                            }
                            .font(.subheadline.bold())
                            .foregroundColor(.black)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color.cyan)
                            .clipShape(Capsule())
                        }

                        Button(action: {
                            exportedURLs = fusionManager.exportScannedMeshPackage()
                            if !exportedURLs.isEmpty {
                                showShareSheet = true
                            }
                        }) {
                            HStack(spacing: 5) {
                                Image(systemName: "square.and.arrow.up")
                                Text("导出 OBJ+MTL+贴图")
                                    .lineLimit(1)
                            }
                            .font(.subheadline.bold())
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color.indigo)
                            .clipShape(Capsule())
                        }
                    }
                }
                .padding(14)
                .background(Color.black.opacity(0.86))
                .clipShape(RoundedRectangle(cornerRadius: 22))
                .padding(.horizontal, 16)
            }
            .padding(.vertical, 8)
        }
        .onAppear {
            fusionManager.pauseSensorStreamsForPreviewStudio()
        }
        .onDisappear {
            fusionManager.resumeSensorStreamsFromPreviewStudio()
        }
        .sheet(isPresented: $showShareSheet) {
            if !exportedURLs.isEmpty {
                ShareSheetView(activityItems: exportedURLs)
            }
        }
    }
}

public struct ShareSheetView: UIViewControllerRepresentable {
    public let activityItems: [Any]
    public func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    public func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
