//
//  ProMotionScannerView.swift
//  ArgonScan Pro — Live Camera + LiDAR 3D Mesh Overlay & 120Hz ProMotion HUD (v1.1)
//

import SwiftUI
import ARKit
import SceneKit

/// Wraps ARSCNView to render live camera feed + real-time LiDAR reconstructed 3D mesh wireframe
public struct LiveARMeshPreview: UIViewRepresentable {
    public let session: ARSession
    public let showWireframe: Bool

    public func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    public func makeUIView(context: Context) -> ARSCNView {
        let scnView = ARSCNView(frame: .zero)
        scnView.session = session
        scnView.delegate = context.coordinator
        scnView.automaticallyUpdatesLighting = true
        scnView.preferredFramesPerSecond = 120
        scnView.debugOptions = [.showFeaturePoints]
        scnView.backgroundColor = .black
        return scnView
    }

    public func updateUIView(_ uiView: ARSCNView, context: Context) {
        context.coordinator.showWireframe = showWireframe
    }

    public final class Coordinator: NSObject, ARSCNViewDelegate {
        public var showWireframe: Bool = true

        public func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
            guard let meshAnchor = anchor as? ARMeshAnchor else { return nil }
            let geom = createSCNGeometry(from: meshAnchor.geometry)
            let node = SCNNode(geometry: geom)
            return node
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
            material.fillMode = showWireframe ? .lines : .fill
            material.diffuse.contents = UIColor.systemCyan.withAlphaComponent(showWireframe ? 0.75 : 0.35)
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
            // Live ARKit Camera + LiDAR 3D Mesh Viewport
            LiveARMeshPreview(session: fusionManager.arSession, showWireframe: showWireframe)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                // Top Telemetry Bar (A15 Bionic Real-Time Monitor)
                HStack(spacing: 12) {
                    Label(String(format: "GPU %.1fms", fusionManager.telemetry.gpuFrameTimeMs), systemImage: "cpu")
                    Label("Vertices: \(fusionManager.telemetry.activeVerticesCount)", systemImage: "cube.transparent")
                    Label(String(format: "RAM %.0fMB", fusionManager.telemetry.memoryUsedMB), systemImage: "memorychip")
                }
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundColor(.cyan)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.72))
                .clipShape(Capsule())

                // Live Status Banner
                Text(fusionManager.telemetry.statusMessage)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.65))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                Spacer()

                // Mode Picker tailored to iPhone 13 Pro Max Sensor Suite
                Picker("Scan Mode", selection: $fusionManager.activeMode) {
                    ForEach(ScanRangeMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

                // Control Bar
                VStack(spacing: 12) {
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("TSDF Voxel: \(String(format: "%.1f", voxelSizeMM)) mm")
                                .font(.caption2)
                                .foregroundColor(.white)
                            Slider(value: $voxelSizeMM, in: 0.5...5.0, step: 0.5)
                                .tint(.cyan)
                        }

                        Toggle("网格线框", isOn: $showWireframe)
                            .font(.caption2)
                            .foregroundColor(.white)
                            .toggleStyle(.button)
                            .tint(.cyan)
                    }

                    HStack(spacing: 12) {
                        Button(action: {
                            fusionManager.toggleScanning()
                        }) {
                            HStack {
                                Image(systemName: fusionManager.isScanning ? "stop.fill" : "camera.viewfinder")
                                Text(fusionManager.isScanning ? "Stop Capture" : "Start Capture")
                            }
                            .font(.subheadline.bold())
                            .foregroundColor(.black)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 13)
                            .background(fusionManager.isScanning ? Color.red : Color.cyan)
                            .clipShape(Capsule())
                        }

                        Button(action: {
                            if let url = fusionManager.exportScannedMeshAsOBJ() {
                                exportedFileURL = url
                                showShareSheet = true
                            }
                        }) {
                            HStack {
                                Image(systemName: "square.and.arrow.up")
                                Text("导出 .OBJ")
                            }
                            .font(.subheadline.bold())
                            .foregroundColor(.white)
                            .padding(.horizontal, 18)
                            .padding(.vertical, 13)
                            .background(Color.indigo)
                            .clipShape(Capsule())
                        }
                    }
                }
                .padding(14)
                .background(Color.black.opacity(0.78))
                .clipShape(RoundedRectangle(cornerRadius: 22))
                .padding(.horizontal)
            }
            .padding(.vertical, 8)
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
