// ios/Runner/LivePosePlugin.swift
//
// Live capture + parallel inference feed (iOS).
// - Records MP4 via AVAssetWriter.
// - Emits ~target2dFps 640x640 RGB letterboxed frames over EventChannel.
// - Provides ratio + pads = [L,T,R,B] for mapping YOLO coords back to original:
//     x_orig = (x_lb - L) / ratio,  y_orig = (y_lb - T) / ratio
//
// Channel names:
//   MethodChannel: "live_pose"           (startCapture / stopCapture)
//   EventChannel : "live_pose/frames"    (per-frame RGB + meta)
//
// NOTE: This file is self-contained; no storyboard/UI work required.
//       Requires NSCameraUsageDescription in Info.plist.

import Foundation
import AVFoundation
import UIKit
import Flutter

final class LivePosePlugin: NSObject, FlutterPlugin, FlutterStreamHandler {

  // MARK: - Flutter bootstrap

  public static func register(with registrar: FlutterPluginRegistrar) {
    let instance = LivePosePlugin(registrar: registrar)

    let method = FlutterMethodChannel(name: "live_pose", binaryMessenger: registrar.messenger())
    registrar.addMethodCallDelegate(instance, channel: method)

    let events = FlutterEventChannel(name: "live_pose/frames", binaryMessenger: registrar.messenger())
    events.setStreamHandler(instance)
  }

  // MARK: - State

  private let registrar: FlutterPluginRegistrar

  // Capture
  private let session = AVCaptureSession()
  private let sessionQueue = DispatchQueue(label: "livepose.session")
  private let videoOutput = AVCaptureVideoDataOutput()

  // Recording
  private var assetWriter: AVAssetWriter?
  private var writerInput: AVAssetWriterInput?
  private var firstPTS: CMTime?
  private var recordedURL: URL?

  // Event sink
  private var eventSink: FlutterEventSink?

  // Throttling/metrics
  private var tokenBucket = TokenBucket(targetFps: 10.0)
  private let emitQueue = DispatchQueue(label: "livepose.emit") // serialize frame emission
  private let ringDepth = 4
  private var outstandingEmits = 0

  // Counters for meta
  private var countReceived = 0
  private var countEmitted2D = 0
  private var countDropped2D = 0
  private var countWriterAppended = 0

  // Frame index
  private var frameIndex = 0

  // Letterbox constants
  private let outW = 640
  private let outH = 640
  private let padRGB: (UInt8, UInt8, UInt8) = (114, 114, 114) // 0x72

  // Init
  init(registrar: FlutterPluginRegistrar) {
    self.registrar = registrar
    super.init()
  }

  // MARK: - FlutterStreamHandler

  func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
    self.eventSink = events
    return nil
  }

  func onCancel(withArguments arguments: Any?) -> FlutterError? {
    self.eventSink = nil
    return nil
  }

  // MARK: - FlutterPlugin (MethodChannel)

  func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "startCapture":
      guard let args = call.arguments as? [String: Any] else {
        result(FlutterError(code: "bad_args", message: "Expected map args", details: nil))
        return
      }
      let savePath = (args["savePath"] as? String) ?? defaultVideoPath()
      let target2dFps = (args["target2dFps"] as? Double) ?? 10.0
      startCapture(savePath: savePath, target2dFps: target2dFps, result: result)

    case "stopCapture":
      stopCapture(result: result)

    default:
      result(FlutterMethodNotImplemented)
    }
  }

  // MARK: - Capture control

  private func startCapture(savePath: String, target2dFps: Double, result: @escaping FlutterResult) {
    sessionQueue.async {
      guard self.assetWriter == nil else {
        result(FlutterError(code: "already_running", message: "Capture already running", details: nil))
        return
      }

      self.resetCounters()
      self.tokenBucket = TokenBucket(targetFps: target2dFps)
      self.frameIndex = 0

      // Ensure parent directory exists (Dart usually passes .../run_xxx/capture.mp4)
      let saveURL = URL(fileURLWithPath: savePath)
      let parent = saveURL.deletingLastPathComponent()
      try? FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)

      // Configure session
      self.session.beginConfiguration()
      self.session.sessionPreset = .high // 1080p on many devices

      // Input (back camera; simple path)
      guard
        let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
        let input = try? AVCaptureDeviceInput(device: camera),
        self.session.canAddInput(input)
      else {
        self.session.commitConfiguration()
        DispatchQueue.main.async {
          result(FlutterError(code: "no_camera", message: "Cannot access back camera", details: nil))
        }
        return
      }
      self.session.addInput(input)

      // Video data output (BGRA for easy CoreImage/CGContext path)
      self.videoOutput.videoSettings = [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
      ]
      self.videoOutput.alwaysDiscardsLateVideoFrames = true
      let cbQueue = DispatchQueue(label: "livepose.capture")
      self.videoOutput.setSampleBufferDelegate(self, queue: cbQueue)
      guard self.session.canAddOutput(self.videoOutput) else {
        self.session.commitConfiguration()
        DispatchQueue.main.async {
          result(FlutterError(code: "no_video_out", message: "Cannot add AVCaptureVideoDataOutput", details: nil))
        }
        return
      }
      self.session.addOutput(self.videoOutput)

      // Orientation: assume portrait as per spec
      if let conn = self.videoOutput.connection(with: .video),
         conn.isVideoOrientationSupported {
        conn.videoOrientation = .portrait
      }

      self.session.commitConfiguration()

      // Prepare writer (H.264 MP4)
      do {
        self.recordedURL = saveURL
        if FileManager.default.fileExists(atPath: saveURL.path) {
          try? FileManager.default.removeItem(at: saveURL)
        }
        self.assetWriter = try AVAssetWriter(outputURL: saveURL, fileType: .mp4)
        // writerInput is created lazily on first frame when we know source size
      } catch {
        self.assetWriter = nil
        DispatchQueue.main.async {
          result(FlutterError(code: "writer_err", message: "AVAssetWriter init failed", details: error.localizedDescription))
        }
        return
      }

      // Start session
      self.firstPTS = nil
      self.session.startRunning()
      DispatchQueue.main.async { result(nil) }
    }
  }

  private func stopCapture(result: @escaping FlutterResult) {
    sessionQueue.async {
      guard self.assetWriter != nil else {
        DispatchQueue.main.async {
          result(["runDir": self.recordedURL?.deletingLastPathComponent().path ?? "",
                  "counts": self.currentCountsDict()])
        }
        return
      }

      self.session.stopRunning()

      // Finish writer
      let writer = self.assetWriter
      self.assetWriter = nil
      self.writerInput?.markAsFinished()

      writer?.finishWriting { [weak self] in
        guard let self = self else { return }
        let resp: [String: Any] = [
          "runDir": self.recordedURL?.deletingLastPathComponent().path ?? "",
          "counts": self.currentCountsDict()
        ]
        self.firstPTS = nil
        self.writerInput = nil
        self.recordedURL = nil
        DispatchQueue.main.async { result(resp) }
      }
    }
  }

  private func resetCounters() {
    countReceived = 0
    countEmitted2D = 0
    countDropped2D = 0
    countWriterAppended = 0
  }

  private func currentCountsDict() -> [String: Any] {
    return [
      "received": countReceived,
      "emitted2D": countEmitted2D,
      "dropped2D": countDropped2D,
      "writerAppended": countWriterAppended
    ]
  }

  private func defaultVideoPath() -> String {
    let dir = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true).first ?? NSTemporaryDirectory()
    let ts = Int(Date().timeIntervalSince1970)
    return (dir as NSString).appendingPathComponent("capture_\(ts).mp4")
  }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension LivePosePlugin: AVCaptureVideoDataOutputSampleBufferDelegate {
  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    countReceived &+= 1

    // Lazily init writer input once we know the first frame's size
    if writerInput == nil {
      guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
      let w = CVPixelBufferGetWidth(pb)
      let h = CVPixelBufferGetHeight(pb)
      guard let wtr = assetWriter else { return }

      let comps: [String: Any] = [
        AVVideoAverageBitRateKey: 8_000_000, // ~8 Mbps
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
      ]
      let settings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: w,
        AVVideoHeightKey: h,
        AVVideoCompressionPropertiesKey: comps
      ]
      let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
      input.expectsMediaDataInRealTime = true

      if wtr.canAdd(input) {
        wtr.add(input)
        writerInput = input
      }

      // Begin writing at first frame PTS
      firstPTS = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
      if wtr.status == .unknown {
        wtr.startWriting()
        if let s = firstPTS { wtr.startSession(atSourceTime: s) }
      }
    }

    // Append to writer
    if let input = writerInput, input.isReadyForMoreMediaData {
      let ok = input.append(sampleBuffer)
      if ok { countWriterAppended &+= 1 }
    }

    // 2D inference feed throttled by token-bucket
    guard tokenBucket.consume() else {
      countDropped2D &+= 1
      return
    }

    // Backpressure guard (drop oldest when too many outstanding emits)
    if outstandingEmits >= ringDepth {
      countDropped2D &+= 1
      return
    }

    // Extract metadata
    let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
    let tsNs = ptsToNanos(pts)
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
    let origW = CVPixelBufferGetWidth(pixelBuffer)
    let origH = CVPixelBufferGetHeight(pixelBuffer)

    // Convert & letterbox → RGB(3bpp) 640x640 + mapping
    guard let packet = makeLetterboxedRGB(from: pixelBuffer, origW: origW, origH: origH) else { return }

    // Emit
    outstandingEmits &+= 1
    countEmitted2D &+= 1
    let idx = frameIndex
    frameIndex &+= 1

    emitQueue.async { [weak self] in
      guard let self = self else { return }
      if let sink = self.eventSink {
        let map: [String: Any] = [
          "idx": idx,
          "tsNs": tsNs,
          "w": self.outW, "h": self.outH,
          "origW": origW, "origH": origH,
          "ratio": packet.ratio,
          "pads": [packet.padL, packet.padT, packet.padR, packet.padB],
          "rgb": FlutterStandardTypedData(bytes: packet.rgb)
        ]
        sink(map)
      }
      self.outstandingEmits &-= 1
    }
  }

  // MARK: - Helpers

  private func ptsToNanos(_ t: CMTime) -> Int64 {
    if t.timescale == 0 { return 0 }
    let seconds = Double(t.value) / Double(t.timescale)
    return Int64((seconds * 1_000_000_000).rounded())
  }

  /// Letterbox to 640x640 RGB8.
  /// Returns RGB data + mapping to original:
  ///   ratio = min(640/origW, 640/origH)
  ///   pads  = [L,T,R,B] in letterbox pixels.
  private func makeLetterboxedRGB(from pb: CVPixelBuffer, origW: Int, origH: Int)
    -> (rgb: Data, ratio: Double, padL: Double, padT: Double, padR: Double, padB: Double)?
  {
    // CIImage from buffer
    let ciImage = CIImage(cvPixelBuffer: pb)

    // Create CGImage (no color transforms beyond sRGB)
    let ciCtx = CIContext(options: [CIContextOption.priorityRequestLow: true])
    guard let cg = ciCtx.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: origW, height: origH)) else {
      return nil
    }

    // Compute aspect-fit rectangle inside 640x640
    let r = min(Double(outW) / Double(origW), Double(outH) / Double(origH))
    let newW = Int((Double(origW) * r).rounded(.toNearestOrAwayFromZero))
    let newH = Int((Double(origH) * r).rounded(.toNearestOrAwayFromZero))
    let padL = (outW - newW) / 2
    let padT = (outH - newH) / 2
    let padR = outW - newW - padL
    let padB = outH - newH - padT
    let destRect = CGRect(x: padL, y: padT, width: newW, height: newH)

    // RGBA8 canvas 640x640, filled with pad color
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let bytesPerRowRGBA = outW * 4
    guard let ctx = CGContext(
      data: nil,
      width: outW, height: outH,
      bitsPerComponent: 8, bytesPerRow: bytesPerRowRGBA,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    // Fill background with (114,114,114)
    ctx.setFillColor(CGColor(red: CGFloat(padRGB.0)/255.0,
                             green: CGFloat(padRGB.1)/255.0,
                             blue: CGFloat(padRGB.2)/255.0,
                             alpha: 1.0))
    ctx.fill(CGRect(x: 0, y: 0, width: outW, height: outH))

    // Draw scaled image
    ctx.interpolationQuality = .medium
    ctx.draw(cg, in: destRect)

    guard let rgbaPtr = ctx.data else { return nil }
    // Convert RGBA → RGB by dropping alpha
    let countRGB = outW * outH * 3
    var rgb = Data(count: countRGB)
    rgb.withUnsafeMutableBytes { rgbBuf in
      let src = rgbaPtr.bindMemory(to: UInt8.self, capacity: outW * outH * 4)
      let dst = rgbBuf.bindMemory(to: UInt8.self).baseAddress!
      var si = 0
      var di = 0
      let total = outW * outH
      for _ in 0..<total {
        dst[di + 0] = src[si + 0] // R
        dst[di + 1] = src[si + 1] // G
        dst[di + 2] = src[si + 2] // B
        si += 4
        di += 3
      }
    }

    return (rgb, r, Double(padL), Double(padT), Double(padR), Double(padB))
  }
}

// MARK: - Token bucket for adaptive subsampling

private final class TokenBucket {
  private let capacity: Double = 2.0       // allow small bursts
  private let targetFps: Double            // tokens per second
  private var tokens: Double
  private var lastRefill: TimeInterval

  init(targetFps: Double) {
    self.targetFps = max(1.0, targetFps)
    self.tokens = 0.0
    self.lastRefill = CACurrentMediaTime()
  }

  func consume() -> Bool {
    refill()
    if tokens >= 1.0 {
      tokens -= 1.0
      return true
    }
    return false
  }

  private func refill() {
    let now = CACurrentMediaTime()
    let dt = now - lastRefill
    lastRefill = now
    tokens = min(capacity, tokens + dt * targetFps)
  }
}
