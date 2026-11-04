//
//  ProMotionScannerView.swift
//  ArgonScan Pro — 120Hz Super Retina XDR ProMotion HUD & Real-Time Voxel Renderer
//

import SwiftUI
import MetalKit

public struct ProMotionScannerView: View {
    @StateObject private var fusionManager = HybridSensorFusionManager()
    @State private var isScanning = false
    @State private var voxelSizeMM: Double = 1.5

    public init() {}

    public var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 16) {
                // Top Telemetry Bar (A15 Bionic Real-Time Monitor)
                HStack(spacing: 12) {
                    Label(String(format: "GPU %.1f ms (120Hz)", fusionManager.telemetry.gpuFrameTimeMs), systemImage: "cpu")
                    Label(String(format: "ANE %.0f%%", fusionManager.telemetry.aneUtilizationPercent), systemImage: "brain.head.profile")
                    Label(String(format: "RAM %.0f/6144 MB", fusionManager.telemetry.memoryUsedMB), systemImage: "memorychip")
                }
                .font(.system(size: 12, weight: .semibold, design: .monospaced))
                .foregroundColor(.green)
                .padding(10)
                .background(Color.white.opacity(0.08))
                .clipShape(Capsule())

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
                HStack(spacing: 24) {
                    VStack(alignment: .leading) {
                        Text("TSDF Voxel: \(String(format: "%.1f", voxelSizeMM)) mm")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Slider(value: $voxelSizeMM, in: 0.5...5.0, step: 0.5)
                            .frame(width: 140)
                    }

                    Button(action: {
                        isScanning.toggle()
                        if isScanning {
                            fusionManager.startScanning(mode: fusionManager.activeMode)
                        }
                    }) {
                        Text(isScanning ? "Stop & Run Argon Critique" : "Start Capture")
                            .font(.headline)
                            .foregroundColor(.black)
                            .padding(.horizontal, 24)
                            .padding(.vertical, 14)
                            .background(isScanning ? Color.red : Color.cyan)
                            .clipShape(Capsule())
                    }
                }
                .padding()
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 24))
            }
            .padding()
        }
    }
}
