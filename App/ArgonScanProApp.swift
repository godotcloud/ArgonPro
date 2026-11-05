//
//  ArgonScanProApp.swift
//  ArgonScan Pro — iPhone 13 Pro Max Entry Point
//

import SwiftUI

@main
struct ArgonScanProApp: App {
    var body: some Scene {
        WindowGroup {
            ProMotionScannerView()
                .preferredColorScheme(.dark)
        }
    }
}
