//
//  ContentView.swift — rewritten
//  CoreML_Test
//
//  Updated: 2025/10/20
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
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)) { _ in
            vm.stop()
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didBecomeActiveNotification)) { _ in
            if vm.modelReady, vm.previewLayer != nil { vm.start() }
        }
        // 起動は modelReady のみ（プレビューLayerがあることを確認）
        .onChange(of: vm.modelReady) { ready in
            if ready, vm.previewLayer != nil {
                // レイアウト安定のため少し遅らせる（ハング検知抑制）
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { vm.start() }
            }
        }
    }
}

// MARK: - ViewModel

final class CameraVM: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    // 出力
    @Published var boxes: [DrawBox] = []
    @Published var cameraDenied = false
    @Published var modelReady = false

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
    private let inferInterval: CFTimeInterval = 0.30 // ≈3.3fps
    private var lastUIUpdate = CFAbsoluteTimeGetCurrent()
    private let uiInterval: CFTimeInterval = 0.10    // UI は最大 10fps

    override init() {
        super.init()
        // カメラ構成はメインをブロックしない
        sessionQueue.async { [weak self] in
            self?.setupCamera()
        }
        // モデル非同期ロード（必要なら CPU 限定のフォールバック）
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
            return
        }
        if session.canAddInput(input) { session.addInput(input) }

        do {
            try cam.lockForConfiguration()
            cam.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 12)
            cam.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 12)
            cam.unlockForConfiguration()
        } catch {}

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: captureQueue)
        if session.canAddOutput(output) { session.addOutput(output) }
        if let conn = output.connection(with: .video), conn.isVideoOrientationSupported {
            conn.videoOrientation = .portrait
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
                return nil
            }
        }
        print("[ERROR] No .mlmodelc/.mlpackage in bundle.")
        return nil
    }

    private func setupVision() {
        print("[DEBUG] setupVision(): async MLModel.load start (no blocking)")
        guard let url = urlForCompiledModel() else { return }

        // 既定は ANE を避けて軽量に初期化
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndGPU

        var completed = false
        if #available(iOS 15.0, *) {
            // 1) .cpuAndGPU で非同期ロード
            MLModel.load(contentsOf: url, configuration: cfg) { [weak self] result in
                guard let self, !completed else { return }
                switch result {
                case .success(let model):
                    completed = true
                    print("[DEBUG] MLModel.load(.cpuAndGPU) ✅")
                    self.buildVNModelAndRequest(model) // request セット→warmUp→ready は buildVisionRequest 内
                case .failure(let err):
                    print("[ERROR] MLModel.load(.cpuAndGPU) failed:", err.localizedDescription)
                    // 失敗は CPU フォールバックに任せる
                }
            }
            // 2) 6秒以内に戻らなければ .cpuOnly で非同期リトライ
            DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 6) { [weak self] in
                guard let self, !completed else { return }
                print("[WARN] .cpuAndGPU load is slow → retry with .cpuOnly")
                let cpuCfg = MLModelConfiguration()
                cpuCfg.computeUnits = .cpuOnly
                MLModel.load(contentsOf: url, configuration: cpuCfg) { [weak self] result in
                    guard let self, !completed else { return }
                    switch result {
                    case .success(let model):
                        completed = true
                        print("[DEBUG] MLModel.load(.cpuOnly) ✅")
                        self.buildVNModelAndRequest(model)
                    case .failure(let err):
                        print("[ERROR] MLModel.load(.cpuOnly) failed:", err.localizedDescription)
                    }
                }
            }
        } else {
            // iOS14 以下はバックグラウンド同期
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self else { return }
                do {
                    let model = try MLModel(contentsOf: url, configuration: cfg)
                    completed = true
                    self.buildVNModelAndRequest(model)
                } catch {
                    print("[ERROR] Legacy MLModel load failed:", error.localizedDescription)
                }
            }
        }
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
        req.imageCropAndScaleOption = .scaleFill

        self.request = req
        print("[DEBUG] request built ✅")

        // ウォームアップは重いので BG で
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

    // 初回だけダミー入力で perform（Metal/BNNS 初期化をここで済ませる）
    private func warmUp() {
        guard let req = self.request else { return }
        if let pb = Self.makePixelBuffer(width: 320, height: 320) {
            let h = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
            _ = try? h.perform([req]) // 結果は捨てる（UI 側は previewLayer が無ければ無視）
        }
    }

    // MARK: 推論

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard modelReady, let request = self.request else { return }

        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastInferTime >= inferInterval else { return }
        lastInferTime = now

        guard inferSemaphore.wait(timeout: .now()) == .success else { return }

        inferQueue.async { [weak self] in
            defer { self?.inferSemaphore.signal() }
            guard let self = self, let pixel = CMSampleBufferGetImageBuffer(sb) else { return }
            let orient = self.currentImageOrientation()
            let h = VNImageRequestHandler(cvPixelBuffer: pixel, orientation: orient, options: [:])
            do {
                try h.perform([request])
            } catch {
                print("[ERROR] VN perform: \(error)")
            }
        }
    }

    // MARK: Utils

    // 端末の現在向きから CGImagePropertyOrientation を推定
    private func currentImageOrientation() -> CGImagePropertyOrientation {
        // プレビューは常に portrait に設定している前提。
        // 背面カメラの縦持ちでは .right が自然になるケースが多い。
        switch UIDevice.current.orientation {
        case .portrait:            return .right
        case .portraitUpsideDown:  return .left
        case .landscapeLeft:       return .up   // ホームボタンが右（古い表現）
        case .landscapeRight:      return .down // ホームボタンが左（古い表現）
        default:                   return .right
        }
    }

    // 正規化座標 → プレビュー Layer 座標（メインスレッドで呼ぶ）
    private func convertToLayerRects(_ obs: [VNRecognizedObjectObservation], previewLayer pl: AVCaptureVideoPreviewLayer) -> [DrawBox] {
        var rects: [DrawBox] = []
        for o in obs.sorted(by: { $0.confidence > $1.confidence }).prefix(60) {
            guard o.confidence >= self.confThresh else { continue }
            let v = o.boundingBox
            let meta = CGRect(x: v.minX, y: 1 - v.minY - v.height, width: v.width, height: v.height)
            let layerRect = pl.layerRectConverted(fromMetadataOutputRect: meta)
            let area = layerRect.width * layerRect.height
            guard area >= 16 * 16 else { continue }
            rects.append(DrawBox(rect: layerRect,
                                 score: o.confidence,
                                 label: o.labels.first?.identifier ?? "pill"))
        }
        return rects
    }

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

// MARK: - Overlay（軽量 Canvas）

struct PillOverlay: View {
    let boxes: [DrawBox]
    var body: some View {
        Canvas { ctx, _ in
            let sorted = boxes.sorted { $0.score > $1.score }
            let maxRects = sorted.prefix(40)

            for b in maxRects where (b.rect.width * b.rect.height) >= 16 * 16 {
                let p = Path(CGRect(x: b.rect.minX, y: b.rect.minY,
                                    width: b.rect.width, height: b.rect.height))
                ctx.stroke(p, with: .color(.green), lineWidth: 2)
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
