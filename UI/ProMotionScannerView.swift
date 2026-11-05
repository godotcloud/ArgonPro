//
//  ProMotionScannerView.swift
//  ArgonScan Pro — Full-Screen Live Camera + LiDAR 3D Mesh Overlay & Responsive HUD (v1.2)
//

import SwiftUI
import ARKit
import SceneKit

/// Renders the live camera video feed from ARSession plus real-time 3D LiDAR mesh anchors
public struct LiveARMeshPreview: UIViewRepresentable {
    public let session: ARSession
    public let showWireframe: Bool

    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    public func makeUIView(context: Context) -> ARSCNView {
        let scnView = ARSCNView(frame: UIScreen.main.bounds)
        scnView.scene = SCNScene()
        scnView.session = session
        scnView.delegate = context.coordinator
        scnView.automaticallyUpdatesLighting = true
        scnView.preferredFramesPerSecond = 120
        scnView.rendersCameraGrain = false
        scnView.rendersMotionBlur = false
        scnView.debugOptions = [.showFeaturePoints]
        scnView.backgroundColor = .clear
        return scnView
    }

    public func updateUIView(_ uiView: ARSCNView, context: Context) {
        context.coordinator.showWireframe = showWireframe
    }

    public final class Coordinator: NSObject, ARSCNViewDelegate {
        public var showWireframe: Bool = true

        public func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
            if let meshAnchor = anchor as? ARMeshAnchor {
                let geom = createSCNGeometry(from: meshAnchor.geometry)
                return SCNNode(geometry: geom)
            }
            if let faceAnchor = anchor as? ARFaceAnchor,
               let device = renderer.device,
               let faceGeo = ARSCNFaceGeometry(device: device) {
                faceGeo.update(from: faceAnchor.geometry)
                faceGeo.firstMaterial?.fillMode = .lines
                faceGeo.firstMaterial?.diffuse.contents = UIColor.systemCyan
                return SCNNode(geometry: faceGeo)
            }
            return nil
        }

        public func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            if let meshAnchor = anchor as? ARMeshAnchor {
                node.geometry = createSCNGeometry(from: meshAnchor.geometry)
            } else if let faceAnchor = anchor as? ARFaceAnchor,
                      let faceGeo = node.geometry as? ARSCNFaceGeometry {
                faceGeo.update(from: faceAnchor.geometry)
            }
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
            material.fillMode = showWireframe ? .lines : .fill
            material.diffuse.contents = UIColor.systemCyan.withAlphaComponent(showWireframe ? 0.80 : 0.35)
            material.isDoubleSided = true
            scnGeom.materials = [material]
            return scnGeom
        }
    }
}

public struct ProMotionScannerView: View {
    @StateObject private var fusionManager = HybridSensorFusionManager()
    @State private var voxelSizeMM: Double = 1.5
    @State private var showWireframe: Bool = true
    @State private var exportedFileURL: URL?
    @State private var showShareSheet: Bool = false

    public init() {}

    public var body: some View {
        ZStack {
            // 1. Live Full-Screen Camera + LiDAR 3D Mesh Viewport (Replaces v1.0 Color.black placeholder)
            LiveARMeshPreview(session: fusionManager.arSession, showWireframe: showWireframe)
                .ignoresSafeArea()

            // 2. Top & Bottom HUD Controls
            VStack(spacing: 10) {
                // Single-line compact telemetry pill (never wraps onto 3 lines)
                HStack(spacing: 14) {
                    Label(String(format: "%.1fms (120Hz)", fusionManager.telemetry.gpuFrameTimeMs), systemImage: "cpu")
                    Label("\(fusionManager.telemetry.activeVerticesCount) 顶点", systemImage: "cube.transparent")
                    Label(String(format: "%.0f MB", fusionManager.telemetry.memoryUsedMB), systemImage: "memorychip")
                }
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .lineLimit(1)
                .minimumScaleFactor(0.75)
                .foregroundColor(.cyan)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.72))
                .clipShape(Capsule())

                // Live Status Banner
                Text(fusionManager.telemetry.statusMessage)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.65))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .padding(.horizontal, 16)

                Spacer()

                // Segmented Mode Picker with auto-switching camera configuration
                Picker("Scan Mode", selection: $fusionManager.activeMode) {
                    ForEach(ScanRangeMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 16)
                .onChange(of: fusionManager.activeMode) { newMode in
                    fusionManager.switchMode(newMode)
                }

                // Bottom Control Card (Full-width horizontal buttons so text never wraps vertically)
                VStack(spacing: 12) {
                    HStack(spacing: 14) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("TSDF 体素精度: \(String(format: "%.1f", voxelSizeMM)) mm · 关键帧: \(fusionManager.telemetry.keyframesCaptured)")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.9))
                                .lineLimit(1)
                            Slider(value: $voxelSizeMM, in: 0.5...5.0, step: 0.5)
                                .tint(.cyan)
                        }

                        Button(action: {
                            showWireframe.toggle()
                        }) {
                            Text(showWireframe ? "线框: 开" : "线框: 关")
                                .font(.caption.bold())
                                .foregroundColor(showWireframe ? .black : .white)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(showWireframe ? Color.cyan : Color.white.opacity(0.18))
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                        }
                    }

                    HStack(spacing: 12) {
                        Button(action: {
                            fusionManager.toggleCapture()
                        }) {
                            HStack(spacing: 6) {
                                Image(systemName: fusionManager.isCapturingMesh ? "stop.circle.fill" : "viewfinder")
                                Text(fusionManager.isCapturingMesh ? "停止并完成重构" : "Start Capture")
                                    .lineLimit(1)
                            }
                            .font(.subheadline.bold())
                            .foregroundColor(fusionManager.isCapturingMesh ? .white : .black)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(fusionManager.isCapturingMesh ? Color.red : Color.cyan)
                            .clipShape(Capsule())
                        }

                        Button(action: {
                            if let url = fusionManager.exportScannedMeshAsOBJ() {
                                exportedFileURL = url
                                showShareSheet = true
                            }
                        }) {
                            HStack(spacing: 6) {
                                Image(systemName: "square.and.arrow.up")
                                Text("导出 .OBJ")
                                    .lineLimit(1)
                            }
                            .font(.subheadline.bold())
                            .foregroundColor(.white)
                            .padding(.horizontal, 18)
                            .padding(.vertical, 14)
                            .background(Color.indigo)
                            .clipShape(Capsule())
                        }
                    }
                }
                .padding(14)
                .background(Color.black.opacity(0.80))
                .clipShape(RoundedRectangle(cornerRadius: 22))
                .padding(.horizontal, 16)
            }
            .padding(.vertical, 8)
        }
        .onAppear {
            fusionManager.ensureCameraPreviewRunning()
        }
        .sheet(isPresented: $showShareSheet) {
            if let url = exportedFileURL {
                ShareSheetView(activityItems: [url])
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
