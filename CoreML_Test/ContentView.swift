//
//  ContentView.swift — iOS 17 deprecations fixed
//  CoreML_Test
//
//  Updated: 2025/10/29
//

import SwiftUI
import UIKit
import AVFoundation
import Vision
import CoreML
import Combine

// MARK: - View

struct ContentView: View {
    @StateObject private var vm = CameraVM()
    
    var body: some View {
        ZStack {
            CameraPreview(session: vm.session) { layer in
                vm.previewLayer = layer
                // 起動は modelReady のみ（ここでは start しない）
            }
            .ignoresSafeArea()

            PillOverlay(boxes: vm.boxes)

            VStack {
                HStack { Spacer(); CountBadge(count: vm.boxes.count).padding() }
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
            // 初回体験向上：起動直後に権限ダイアログを出しておく（未決定のみ）
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
                // レイアウト安定のため少し遅らせる（ハング抑制）
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

// MARK: - ViewModel

final class CameraVM: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    // 出力
    @Published var boxes: [DrawBox] = []
    @Published var cameraDenied = false
    @Published var modelReady = false
    @Published var errorMessage: String?

    // カメラ
    let session = AVCaptureSession()
    fileprivate var previewLayer: AVCaptureVideoPreviewLayer?
    private let sessionQueue = DispatchQueue(label: "pill.session.queue")

    // Capture / Infer を分離
    private let captureQueue = DispatchQueue(label: "pill.capture.queue", qos: .userInitiated)
    private let inferQueue   = DispatchQueue(label: "pill.infer.queue", qos: .userInitiated)

    // Vision
    private var request: VNCoreMLRequest?
    private var didSetupRequest = false  // 二重構築ガード
    private let confThresh: VNConfidence = 0.10

    // 実行制御
    private let inferSemaphore = DispatchSemaphore(value: 1)
    private var lastInferTime = CFAbsoluteTimeGetCurrent()
    private var targetFPS: Double = 3.0 { didSet { inferInterval = 1.0 / max(1.0, targetFPS) } }
    private var inferInterval: CFTimeInterval = 1.0 / 3.0 // ≈3fps
    private var lastUIUpdate = CFAbsoluteTimeGetCurrent()
    private let uiInterval: CFTimeInterval = 0.10    // UI は最大 10fps

    // ピクセルバッファ寸法（回転後）
    private var lastPixelBufferSize: CGSize?

    // 使用カメラの向き（オリエンテーション変換で使用）
    private var devicePosition: AVCaptureDevice.Position = .back

    override init() {
        super.init()
        // カメラ構成はメインをブロックしない
        sessionQueue.async { [weak self] in
            self?.setupCamera()
        }
        // モデル非同期ロード（端末状態に合わせて計算資源を選択）
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

        // 背面広角
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
            // 12fps ロック（遅延と発熱のバランス）
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
                // iOS17+: 回転角ベース
                if conn.isVideoRotationAngleSupported(90) {
                    conn.videoRotationAngle = 90
                }
            } else if conn.isVideoOrientationSupported {
                // iOS16-: 旧API
                conn.videoOrientation = .portrait
            }
        }
        session.commitConfiguration()
    }

    // MARK: Vision/CoreML

    // バンドル内の .mlmodelc を優先。なければ .mlpackage をコンパイル
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

        // 🔁 低電力モードで tier の優先を切替
        let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        let tiers: [MLComputeUnits] = lowPower
        ? [.cpuOnly, .cpuAndGPU, .all]   // 省電力優先
        : [.all, .cpuAndGPU, .cpuOnly]   // 性能優先

        let timeoutSec: Double = 6
        var completed = false

        func tryLoad(at index: Int) {
            guard index < tiers.count, !completed else { return }
            let cfg = MLModelConfiguration()
            cfg.computeUnits = tiers[index]
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
            // タイムアウトで次 tier を試す（成功/失敗で completed が立つ）
            DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + timeoutSec) {
                if !completed { tryLoad(at: index + 1) }
            }
        }
        tryLoad(at: 0)
    }

    // VNCoreMLModel を作って Request 構築
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
        // 二重実行ガード
        if didSetupRequest {
            print("[DEBUG] buildVisionRequest skipped (already set)")
            return
        }
        didSetupRequest = true

        let req = VNCoreMLRequest(model: vnModel) { [weak self] request, _ in
            guard let self else { return }
            let obs = (request.results as? [VNRecognizedObjectObservation]) ?? []

            let now = CFAbsoluteTimeGetCurrent()
            guard now - self.lastUIUpdate >= self.uiInterval else { return }
            self.lastUIUpdate = now

            // レイヤー変換＋UI 更新はメインへ
            DispatchQueue.main.async {
                guard let pl = self.previewLayer else { return }
                self.boxes = self.convertToLayerRects(obs, previewLayer: pl)
            }
        }

        // ⚠️ usesCPUOnly は iOS17 で非推奨 → 使わない
        // 省電力対応は MLModelConfiguration.computeUnits 側で解決済み

        // YOLO の前処理に合わせて調整（学習が letterbox なら .scaleFit が近い）
        req.imageCropAndScaleOption = .scaleFill

        self.request = req
        print("[DEBUG] request built ✅")

        // ウォームアップ（BG）
        inferQueue.async { [weak self] in
            guard let self else { return }
            self.warmUp()
            print("[DEBUG] warmUp done (background)")
            DispatchQueue.main.async {
                self.modelReady = true
                print("[DEBUG] modelReady = true (after warmUp) ✅")
            }
        }
    }

    // 初回だけダミー入力で perform（Metal/BNNS 初期化）
    private func warmUp() {
        guard let req = self.request else { return }
        if let pb = Self.makePixelBuffer(width: 320, height: 320) {
            let h = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
            _ = try? h.perform([req]) // 結果は捨てる
        }
    }

    // MARK: 推論

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard modelReady, let request = self.request else { return }

        // 実フレームのピクセル寸法を更新（回転に応じて入替）
        if let pixel = CMSampleBufferGetImageBuffer(sb) {
            let w = CVPixelBufferGetWidth(pixel)
            let h = CVPixelBufferGetHeight(pixel)
            #if compiler(>=5.9)
            if #available(iOS 17.0, *) {
                // videoRotationAngle で 90/270 は縦持ち（幅高入替）
                let rot = Int(connection.videoRotationAngle) % 360
                switch rot {
                case 90, 270:
                    lastPixelBufferSize = CGSize(width: h, height: w)
                default:
                    lastPixelBufferSize = CGSize(width: w, height: h)
                }
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
            #else
            switch connection.videoOrientation {
            case .portrait, .portraitUpsideDown:
                lastPixelBufferSize = CGSize(width: h, height: w)
            case .landscapeLeft, .landscapeRight:
                lastPixelBufferSize = CGSize(width: w, height: h)
            @unknown default:
                lastPixelBufferSize = CGSize(width: w, height: h)
            }
            #endif
        }

        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastInferTime >= inferInterval else { return }
        lastInferTime = now

        guard inferSemaphore.wait(timeout: .now()) == .success else { return }

        inferQueue.async { [weak self] in
            defer { self?.inferSemaphore.signal() }
            guard let self = self else { return }

            // オリエンテーションを "回転角×カメラ位置" から決定
            let orient = self.cgImageOrientation(for: connection, devicePosition: self.devicePosition)

            // sampleBuffer から直接 handler を生成（余計なコピーを避ける）
            let handler = VNImageRequestHandler(cmSampleBuffer: sb, orientation: orient, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("[ERROR] VN perform:", error.localizedDescription)
            }
        }
    }

    // MARK: Orientation (rotation-angle based for iOS17)

    private func cgImageOrientation(for connection: AVCaptureConnection, devicePosition: AVCaptureDevice.Position) -> CGImagePropertyOrientation {
        let isFront = (devicePosition == .front)

        var rotationAngle: CGFloat = 0
        if #available(iOS 17.0, *) {
            rotationAngle = CGFloat(connection.videoRotationAngle) // 0 / 90 / 180 / 270
        } else {
            switch connection.videoOrientation {
            case .portrait:            rotationAngle = 90
            case .portraitUpsideDown:  rotationAngle = 270
            case .landscapeRight:      rotationAngle = 0
            case .landscapeLeft:       rotationAngle = 180
            @unknown default:          rotationAngle = 90
            }
        }

        switch Int(rotationAngle) % 360 {
        case 0:   return isFront ? .upMirrored    : .up
        case 90:  return isFront ? .leftMirrored  : .right
        case 180: return isFront ? .downMirrored  : .down
        case 270: return isFront ? .rightMirrored : .left
        default:  return isFront ? .leftMirrored  : .right
        }
    }

    // MARK: 座標変換（PreviewLayer API を優先）

    private func convertToLayerRects(
        _ obs: [VNRecognizedObjectObservation],
        previewLayer pl: AVCaptureVideoPreviewLayer
    ) -> [DrawBox] {
        guard lastPixelBufferSize != nil else { return [] }

        var rects: [DrawBox] = []
        for o in obs.sorted(by: { $0.confidence > $1.confidence }).prefix(40) {
            guard o.confidence >= self.confThresh else { continue }

            // Vision 正規化（左下原点）→ 左上原点へ反転
            let v = o.boundingBox
            let normTop = CGRect(x: v.minX,
                                 y: 1 - v.minY - v.height,
                                 width: v.width,
                                 height: v.height)

            // 公式APIで 0..1 → レイヤ座標へ
            let layerRect = pl.layerRectConverted(fromMetadataOutputRect: normTop)

            let area = layerRect.width * layerRect.height
            guard area >= 16 * 16 else { continue }

            rects.append(DrawBox(rect: layerRect,
                                 score: o.confidence,
                                 label: o.labels.first?.identifier ?? "pill"))
        }
        return rects
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

// MARK: - Overlay（軽量 Canvas + ラベル）

struct PillOverlay: View {
    let boxes: [DrawBox]
    var body: some View {
        Canvas { ctx, _ in
            let sorted = boxes.sorted { $0.score > $1.score }
            let maxRects = sorted.prefix(40)

            for b in maxRects where (b.rect.width * b.rect.height) >= 16 * 16 {
                // 枠
                let p = Path(CGRect(x: b.rect.minX, y: b.rect.minY,
                                    width: b.rect.width, height: b.rect.height))
                ctx.stroke(p, with: .color(.green), lineWidth: 2)

                // ラベル
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

// MARK: - カウント表示

struct CountBadge: View {
    let count: Int
    var body: some View {
        Text("Pills: \(count)")
            .font(.headline)
            .padding(10)
            .background(.ultraThinMaterial)
            .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - 描画用モデル

struct DrawBox: Hashable {
    var rect: CGRect
    var score: VNConfidence
    var label: String
}

// MARK: - Preview

#Preview {
    ContentView()
}
