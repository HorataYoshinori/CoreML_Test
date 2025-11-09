//
//  ContentView.swift — presets + FPS HUD + iOS17 fixes
//  CoreML_Test
//
//  Updated: 2025/10/30
//

import SwiftUI
import UIKit
import AVFoundation
import Vision
import CoreML
import Combine

// MARK: - Presets

struct DetectionPreset: Identifiable, Equatable {
    let id = UUID()
    let title: String
    let maxBoxes: Int
    let confThresh: VNConfidence
    let minAreaPt: CGFloat         // レイヤ（ポイント）基準の最小面積
    let useNMS: Bool
    let iouThreshold: CGFloat      // NMSのIoU閾値
    static let comfort80  = DetectionPreset(title: "快適 80",  maxBoxes: 80,  confThresh: 0.15, minAreaPt: 20*20, useNMS: true,  iouThreshold: 0.60)
    static let balanced100 = DetectionPreset(title: "標準 100", maxBoxes: 100, confThresh: 0.15, minAreaPt: 20*20, useNMS: true,  iouThreshold: 0.55)
    static let aggressive120 = DetectionPreset(title: "拡張 120", maxBoxes: 120, confThresh: 0.20, minAreaPt: 22*22, useNMS: true,  iouThreshold: 0.50)
    static let all: [DetectionPreset] = [.comfort80, .balanced100, .aggressive120]
}

// MARK: - View

struct ContentView: View {
    @StateObject private var vm = CameraVM()
    @State private var nearFocusOn = true
    
    var body: some View {
        ZStack {
            CameraPreview(session: vm.session) { layer in
                vm.previewLayer = layer
            }
            .ignoresSafeArea()

            PillOverlay(boxes: vm.boxes, showLabels: true)

            // HUD & メニュー
            VStack(spacing: 8) {
                HStack {
                    // プリセット切替メニュー（左上）
                    Menu {
                        ForEach(DetectionPreset.all) { p in
                            Button {
                                vm.selectedPreset = p
                            } label: {
                                if vm.selectedPreset == p {
                                    Label(p.title, systemImage: "checkmark")
                                } else {
                                    Text(p.title)
                                }
                            }
                        }
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "slider.horizontal.3")
                            Text(vm.selectedPreset.title)
                        }
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(.ultraThinMaterial)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                    
                    // ContentView のHUD近くに

                    Button {
                        nearFocusOn.toggle()
                        vm.setNearFocus(nearFocusOn)
                    } label: {
                        Label(nearFocusOn ? "近距離AF ON" : "近距離AF OFF", systemImage: "viewfinder")
                    }
                    .buttonStyle(.bordered)
                    
                    Spacer()
                    // FPS HUD（右上）
                    HUDView(fps: vm.actualFPS, shown: vm.boxes.count, maxShown: vm.selectedPreset.maxBoxes)
                }
                .padding(.horizontal, 12).padding(.top, 12)
                Spacer()
            }

            if vm.cameraDenied {
                VStack(spacing: 12) {
                    Text("設定 > プライバシー > カメラ で本アプリを許可してください")
                        .multilineTextAlignment(.center)
                    Button("設定を開く") {
                        if let url = URL(string: UIApplication.openSettingsURLString) {
                            UIApplication.shared.open(url)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding()
            }
        }
        .onAppear {
            if AVCaptureDevice.authorizationStatus(for: .video) == .notDetermined {
                AVCaptureDevice.requestAccess(for: .video) { _ in }
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) { _ in
            vm.stop()
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
            if vm.modelReady, vm.previewLayer != nil { vm.start() }
        }
        // iOS17の onChange 新シグネチャ（2引数）
        .onChange(of: vm.modelReady) { _, ready in
            if ready, vm.previewLayer != nil {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { vm.start() }
            }
        }
        .alert("エラー", isPresented: Binding(get: { vm.errorMessage != nil },
                                           set: { if !$0 { vm.errorMessage = nil } })) {
            Button("OK", role: .cancel) { vm.errorMessage = nil }
        } message: {
            Text(vm.errorMessage ?? "")
        }
        
        
    }
}



// MARK: - HUD

struct HUDView: View {
    let fps: Double
    let shown: Int
    let maxShown: Int
    var body: some View {
        HStack(spacing: 10) {
            Label(String(format: "FPS %.1f", fps), systemImage: "speedometer")
            Text("Pills \(shown)/\(maxShown)")
        }
        .font(.caption)
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - ViewModel

final class CameraVM: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    // 出力
    @Published var boxes: [DrawBox] = []
    @Published var cameraDenied = false
    @Published var modelReady = false
    @Published var errorMessage: String?
    @Published var actualFPS: Double = 0.0
    @Published var selectedPreset: DetectionPreset = .comfort80   // ← デフォルト：80

    // カメラ
    let session = AVCaptureSession()
    fileprivate var previewLayer: AVCaptureVideoPreviewLayer?
    private let sessionQueue = DispatchQueue(label: "pill.session.queue")

    // Capture / Infer を分離
    private let captureQueue = DispatchQueue(label: "pill.capture.queue", qos: .userInitiated)
    private let inferQueue   = DispatchQueue(label: "pill.infer.queue", qos: .userInitiated)

    // Vision
    private var request: VNCoreMLRequest?
    private var didSetupRequest = false
    // スロットリング & UI
    private let inferSemaphore = DispatchSemaphore(value: 1)
    private var lastInferTime = CFAbsoluteTimeGetCurrent()
    private var targetFPS: Double = 3.0 { didSet { inferInterval = 1.0 / max(1.0, targetFPS) } }
    private var inferInterval: CFTimeInterval = 1.0 / 3.0
    private var lastUIUpdate = CFAbsoluteTimeGetCurrent()
    private let uiInterval: CFTimeInterval = 0.10
    private var lastPerfTime = CFAbsoluteTimeGetCurrent() // FPS算出用

    // 画像サイズ（回転後）
    private var lastPixelBufferSize: CGSize?
    // 使用カメラ
    private var devicePosition: AVCaptureDevice.Position = .back

    override init() {
        super.init()
        sessionQueue.async { [weak self] in
            self?.setupCamera()
        }
        setupVision()
    }

    // MARK: Camera

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            sessionQueue.async {
                if !self.session.isRunning {
                    self.session.startRunning()
                    print("[INFO] session.startRunning()")
                }
            }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
                guard let self else { return }
                if ok {
                    self.sessionQueue.async {
                        if !self.session.isRunning { self.session.startRunning() }
                        print("[INFO] session.startRunning() after permission")
                    }
                } else {
                    DispatchQueue.main.async { self.cameraDenied = true }
                }
            }
        default:
            DispatchQueue.main.async { self.cameraDenied = true }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
            print("[INFO] session.stopRunning()")
        }
    }

    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .vga640x480

        guard let cam = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: cam) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.errorMessage = "カメラ入力を初期化できませんでした。" }
            return
        }
        devicePosition = cam.position
        if session.canAddInput(input) { session.addInput(input) }

        do {
            try cam.lockForConfiguration()
            if cam.activeFormat.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= 12 }) {
                cam.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 12)
                cam.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 12)
            }
            if cam.isSmoothAutoFocusSupported { cam.isSmoothAutoFocusEnabled = true }
            if cam.isLowLightBoostSupported { cam.automaticallyEnablesLowLightBoostWhenAvailable = true }
            if cam.isFocusModeSupported(.continuousAutoFocus) { cam.focusMode = .continuousAutoFocus }
            if cam.isExposureModeSupported(.continuousAutoExposure) { cam.exposureMode = .continuousAutoExposure }
            if cam.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) { cam.whiteBalanceMode = .continuousAutoWhiteBalance }
            cam.unlockForConfiguration()
        } catch {
            print("[WARN] Camera lockForConfiguration failed: \(error.localizedDescription)")
        }

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: captureQueue)
        if session.canAddOutput(output) { session.addOutput(output) }
        if let conn = output.connection(with: .video) {
            if #available(iOS 17.0, *) {
                if conn.isVideoRotationAngleSupported(90) {
                    conn.videoRotationAngle = 90
                }
            } else if conn.isVideoOrientationSupported {
                conn.videoOrientation = .portrait
            }
        }
        session.commitConfiguration()
    }

    // MARK: Vision/CoreML

    private func urlForCompiledModel() -> URL? {
        let bundle = Bundle.main
        if let urls = bundle.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil),
           let u = urls.first {
            print("[INFO] Found .mlmodelc:", u.lastPathComponent)
            return u
        }
        print("[WARN] No .mlmodelc. Try .mlpackage → compile.")
        if let pkgs = bundle.urls(forResourcesWithExtension: "mlpackage", subdirectory: nil),
           let p = pkgs.first {
            do {
                let compiled = try MLModel.compileModel(at: p)
                print("[INFO] Compiled .mlpackage → .mlmodelc:", compiled.lastPathComponent)
                return compiled
            } catch {
                print("[ERROR] Compile .mlpackage failed:", error.localizedDescription)
                DispatchQueue.main.async { self.errorMessage = "CoreMLモデル(.mlpackage)のコンパイルに失敗しました。" }
                return nil
            }
        }
        print("[ERROR] No .mlmodelc/.mlpackage in bundle.")
        DispatchQueue.main.async { self.errorMessage = "CoreMLモデルがバンドル内に見つかりません。" }
        return nil
    }

    private func setupVision() {
        print("[DEBUG] setupVision(): staged MLModel.load start (no blocking)")
        guard let url = urlForCompiledModel() else { return }

        // 低電力モードでロード優先度を切替
        let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        let tiers: [MLComputeUnits] = lowPower
        ? [.cpuOnly, .cpuAndGPU, .all]
        : [.all, .cpuAndGPU, .cpuOnly]

        let timeoutSec: Double = 6
        var completed = false

        func tryLoad(at index: Int) {
            guard index < tiers.count, !completed else { return }
            let cfg = MLModelConfiguration(); cfg.computeUnits = tiers[index]
            MLModel.load(contentsOf: url, configuration: cfg) { [weak self] result in
                guard let self, !completed else { return }
                switch result {
                case .success(let model):
                    completed = true
                    print("[DEBUG] MLModel.load(\(tiers[index])) ✅ (lowPower:\(lowPower))")
                    self.buildVNModelAndRequest(model)
                case .failure(let err):
                    print("[WARN] load(\(tiers[index])) failed:", err.localizedDescription)
                    tryLoad(at: index + 1)
                }
            }
            DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + timeoutSec) {
                if !completed { tryLoad(at: index + 1) }
            }
        }
        tryLoad(at: 0)
    }

    private func buildVNModelAndRequest(_ model: MLModel) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let vn = try VNCoreMLModel(for: model)
                print("[DEBUG] VNCoreMLModel built ✅. Building request…")
                DispatchQueue.main.async {
                    self.buildVisionRequest(vn)
                }
            } catch {
                print("[ERROR] VNCoreMLModel build failed:", error.localizedDescription)
                DispatchQueue.main.async { self.errorMessage = "VNCoreMLModel の生成に失敗しました。" }
            }
        }
    }

    private func buildVisionRequest(_ vnModel: VNCoreMLModel) {
        if didSetupRequest { return }
        didSetupRequest = true

        let req = VNCoreMLRequest(model: vnModel) { [weak self] request, _ in
            guard let self else { return }
            let obs = (request.results as? [VNRecognizedObjectObservation]) ?? []

            let now = CFAbsoluteTimeGetCurrent()
            guard now - self.lastUIUpdate >= self.uiInterval else { return }
            self.lastUIUpdate = now

            DispatchQueue.main.async {
                guard let pl = self.previewLayer else { return }
                self.boxes = self.convertToLayerRects(obs, previewLayer: pl)
            }
        }
        // 学習に合わせて選択（letterbox なら .scaleFit）
        req.imageCropAndScaleOption = .scaleFill

        self.request = req
        print("[DEBUG] request built ✅")

        // Warm-up
        inferQueue.async { [weak self] in
            guard let self else { return }
            self.warmUp()
            DispatchQueue.main.async {
                self.modelReady = true
            }
        }
    }

    private func warmUp() {
        guard let req = self.request else { return }
        if let pb = Self.makePixelBuffer(width: 320, height: 320) {
            let h = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
            _ = try? h.perform([req])
        }
    }

    // MARK: 推論

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard modelReady, let request = self.request else { return }

        // 画像サイズ（回転に応じて入替）
        if let pixel = CMSampleBufferGetImageBuffer(sb) {
            let w = CVPixelBufferGetWidth(pixel)
            let h = CVPixelBufferGetHeight(pixel)
            if #available(iOS 17.0, *) {
                let rot = (Int(connection.videoRotationAngle) % 360 + 360) % 360
                lastPixelBufferSize = (rot == 90 || rot == 270) ? CGSize(width: h, height: w) : CGSize(width: w, height: h)
            } else {
                switch connection.videoOrientation {
                case .portrait, .portraitUpsideDown:
                    lastPixelBufferSize = CGSize(width: h, height: w)
                case .landscapeLeft, .landscapeRight:
                    lastPixelBufferSize = CGSize(width: w, height: h)
                @unknown default:
                    lastPixelBufferSize = CGSize(width: w, height: h)
                }
            }
        }

        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastInferTime >= inferInterval else { return }
        lastInferTime = now

        guard inferSemaphore.wait(timeout: .now()) == .success else { return }

        inferQueue.async { [weak self] in
            guard let self = self else { return }
            defer { self.inferSemaphore.signal() }

            // Vision のオリエンテーション（プレビューと整合のため 180°補正）
            let orient = self.cgImageOrientation(for: connection, devicePosition: self.devicePosition)

            let t0 = CFAbsoluteTimeGetCurrent()
            let handler = VNImageRequestHandler(cmSampleBuffer: sb, orientation: orient, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("[ERROR] VN perform:", error.localizedDescription)
            }
            let t1 = CFAbsoluteTimeGetCurrent()

            // 実測FPS（EMA）
            let dt = max(1e-3, t1 - self.lastPerfTime)
            self.lastPerfTime = t1
            let instFPS = 1.0 / dt
            DispatchQueue.main.async {
                self.actualFPS = self.actualFPS * 0.9 + instFPS * 0.1
            }
        }
    }

    // MARK: Orientation（iOS17: rotationAngle + 180°補正）

    private func cgImageOrientation(for connection: AVCaptureConnection, devicePosition: AVCaptureDevice.Position) -> CGImagePropertyOrientation {
        let isFront = (devicePosition == .front)
        var angle: Int
        if #available(iOS 17.0, *) {
            angle = Int(connection.videoRotationAngle) % 360
        } else {
            switch connection.videoOrientation {
            case .portrait:            angle = 90
            case .portraitUpsideDown:  angle = 270
            case .landscapeRight:      angle = 0
            case .landscapeLeft:       angle = 180
            @unknown default:          angle = 90
            }
        }
        // 180°補正（プレビューと整合）
        let adjusted = (angle + 180) % 360
        switch adjusted {
        case 0:   return isFront ? .upMirrored    : .up
        case 90:  return isFront ? .leftMirrored  : .right
        case 180: return isFront ? .downMirrored  : .down
        case 270: return isFront ? .rightMirrored : .left
        default:  return isFront ? .leftMirrored  : .right
        }
    }

    // MARK: 座標変換（Y反転→公式変換 + NMS/制限）

    private func convertToLayerRects(
        _ obs: [VNRecognizedObjectObservation],
        previewLayer pl: AVCaptureVideoPreviewLayer
    ) -> [DrawBox] {
        let preset = selectedPreset
        guard !obs.isEmpty else { return [] }

        // 1) スコア順 → まずは上位を広めに確保
        var rects: [DrawBox] = []
        for o in obs.sorted(by: { $0.confidence > $1.confidence }).prefix(max(preset.maxBoxes * 2, preset.maxBoxes)) {
            guard o.confidence >= preset.confThresh else { continue }
            // Vision 正規化（左下原点）→ メタデータ座標（左上原点）へ反転
            let v = o.boundingBox
            let meta = CGRect(x: v.minX,
                              y: 1 - v.minY - v.height,
                              width: v.width,
                              height: v.height)
            // メタデータ(0..1, 左上原点) → レイヤ座標
            let layerRect = pl.layerRectConverted(fromMetadataOutputRect: meta)
            let clipped = layerRect.intersection(pl.bounds)
            guard clipped.width >= 1, clipped.height >= 1 else { continue }
            // 最小面積
            guard (clipped.width * clipped.height) >= preset.minAreaPt else { continue }

            rects.append(DrawBox(rect: clipped,
                                 score: o.confidence,
                                 label: o.labels.first?.identifier ?? "pill"))
            if rects.count >= preset.maxBoxes * 2 { break }
        }

        // 2) 簡易NMS（オプション）
        if preset.useNMS {
            var kept: [DrawBox] = []
            for b in rects.sorted(by: { $0.score > $1.score }) {
                if kept.allSatisfy({ iou($0.rect, b.rect) < preset.iouThreshold }) {
                    kept.append(b)
                    if kept.count >= preset.maxBoxes { break }
                }
            }
            return kept
        } else {
            return Array(rects.sorted(by: { $0.score > $1.score }).prefix(preset.maxBoxes))
        }
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        if inter.isNull || inter.isEmpty { return 0 }
        let ia = inter.width * inter.height
        let ua = (a.width*a.height) + (b.width*b.height) - ia
        if ua <= 0 { return 0 }
        return ia / ua
    }

    // MARK: Utils

    private static func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue as Any,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue as Any
        ] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &pb)
        return pb
    }
    
    // CameraVM 内
    func setNearFocus(_ on: Bool) {
        sessionQueue.async { [weak self] in
            guard let self,
                  let cam = self.session.inputs.compactMap({ $0 as? AVCaptureDeviceInput }).first?.device
            else { return }
            do {
                try cam.lockForConfiguration()
                if cam.isAutoFocusRangeRestrictionSupported {
                    cam.autoFocusRangeRestriction = on ? .near : .none
                }
                if cam.isFocusModeSupported(.continuousAutoFocus) {
                    cam.focusMode = .continuousAutoFocus
                } else if cam.isFocusModeSupported(.autoFocus) {
                    cam.focusMode = .autoFocus
                }
                cam.unlockForConfiguration()
            } catch { print("[FOCUS] lock failed:", error.localizedDescription) }
        }
    }
}

// MARK: - Preview Layer

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    let onReady: (AVCaptureVideoPreviewLayer) -> Void

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        let l = v.videoPreviewLayer
        l.videoGravity = .resizeAspectFill
        l.session = session
        onReady(l)
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.videoPreviewLayer.frame = uiView.bounds
    }
}

// MARK: - Overlay（小枠はラベル省略）

struct PillOverlay: View {
    let boxes: [DrawBox]
    let showLabels: Bool
    var body: some View {
        Canvas { ctx, _ in
            let sorted = boxes.sorted { $0.score > $1.score }
            for b in sorted {
                // 枠
                let p = Path(CGRect(x: b.rect.minX, y: b.rect.minY, width: b.rect.width, height: b.rect.height))
                ctx.stroke(p, with: .color(.green), lineWidth: 2)

                // 小さい枠はラベル省略（描画負荷を抑える）
                guard showLabels, b.rect.width >= 28, b.rect.height >= 14 else { continue }
                let text = "\(b.label) \(Int(b.score * 100))%"
                let layout = Text(AttributedString(text))
                    .font(.caption)
                    .foregroundStyle(.green)
                ctx.draw(layout, at: CGPoint(x: b.rect.minX + 4, y: b.rect.minY + 12), anchor: .topLeading)
            }
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }
}

// MARK: - モデル

struct DrawBox: Hashable {
    var rect: CGRect
    var score: VNConfidence
    var label: String
}

// MARK: - Preview

#Preview {
    ContentView()
}
