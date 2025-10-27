// lib/core2d.dart
//
// Shared 2D engine for both offline and live pipelines.
// - Input: FramePacket { letterboxed 640x640 RGB8, origW/H, ratio, pads, tsNs, idx }
// - YOLO every `yoloStride` frames (detector cadence), hold bbox otherwise
// - RTMPose on *every* processed frame (crop from letterbox → map back to original)
// - RTM preprocess auto-tune on the first valid crop
// - Confidence-aware EMA smoothing of 2D keypoints
//
// Assets expected (same as your current project):
//   assets/models/configs/yolo.cfg
//   assets/models/configs/rtm.cfg
//   assets/models/configs/motionbert.cfg   (read here only for convenience)
//   assets/models/configs/skeletons.json   (optional)
//   YOLO/RTM ONNX files referenced in cfgs

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter/services.dart' show rootBundle, MethodChannel;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

typedef CocoXYC = List<List<double>>;

class FramePacket {
  final Uint8List rgb640x640; // RGB8, letterboxed
  final int origW, origH;
  final double ratio; // scale used before pad: min(640/w, 640/h)
  final List<double> pads; // [L,T,R,B] in letterbox pixels
  final int frameIdx;
  final int tsNs;

  const FramePacket({
    required this.rgb640x640,
    required this.origW,
    required this.origH,
    required this.ratio,
    required this.pads,
    required this.frameIdx,
    required this.tsNs,
  });
}

class _Cfg2D {
  // YOLO
  late String yoloModelAsset;
  int yoloInH = 640, yoloInW = 640;
  int personClassId = 0;
  double confTh = 0.25, iouTh = 0.45;
  String classScoreActivation = 'sigmoid';
  String yoloUnits = 'normalized'; // normalized|pixels
  String yoloCoords = 'letterbox'; // letterbox|original

  // RTM
  late String rtmModelAsset;
  int rtmInH = 256, rtmInW = 192;
  String rtmPreproc = 'rgb_255';
  List<double> rtmMean = [0, 0, 0];
  List<double> rtmStd  = [255, 255, 255];
  double simccRatio = 2.0;

  // Skeleton fallback (COCO17)
  List<String> cocoOrder = const [
    "Nose","LEye","REye","LEar","REar",
    "LShoulder","RShoulder","LElbow","RElbow",
    "LWrist","RWrist","LHip","RHip",
    "LKnee","RKnee","LAnkle","RAnkle"
  ];
}

class YDet {
  YDet(this.xyxy, this.score);
  final List<double> xyxy; // [x1,y1,x2,y2] in ORIGINAL pixels
  final double score;
}

/// Confidence-aware EMA. Lower alpha when confidence low.
class _EmaSmoother {
  _EmaSmoother({this.alphaHi = 0.35, this.alphaLo = 0.15, this.confPivot = 0.5});
  final double alphaHi, alphaLo, confPivot;
  List<List<double>>? _prev; // [17][3]

  CocoXYC apply(CocoXYC cur) {
    if (_prev == null) {
      _prev = cur.map((p) => [p[0], p[1], p[2]]).toList(growable: false);
      return cur;
    }
    final out = List.generate(17, (_) => [0.0, 0.0, 0.0], growable: false);
    for (int j = 0; j < 17; j++) {
      final c = cur[j][2].clamp(0.0, 1.0);
      final a = c >= confPivot ? alphaHi : alphaLo;
      out[j][0] = a * cur[j][0] + (1 - a) * _prev![j][0];
      out[j][1] = a * cur[j][1] + (1 - a) * _prev![j][1];
      out[j][2] = cur[j][2]; // keep current confidence
    }
    _prev = out.map((p) => [p[0], p[1], p[2]]).toList(growable: false);
    return out;
  }

  void reset() => _prev = null;
}

class Core2DEngine {
  Core2DEngine({this.yoloStride = 2});

  final int yoloStride;
  final _Cfg2D cfg = _Cfg2D();

  late final OnnxRuntime _ort;
  late OrtSession _yolo, _rtm;
  late String _yoloInName, _rtmInName;
  late List<String> _yoloOutNames, _rtmOutNames;

  // Buffers
  late final Float32List _yoloBuf; // 3*640*640
  late final Float32List _rtmBuf;  // 3*256*192

  // State
  List<double>? _prevBoxOrig; // last bbox in ORIGINAL pixels
  String? _rtmPreprocChosen;
  final List<String> _rtmTryModes = const ['rgb_255', 'bgr_255', 'rgb_ms', 'bgr_ms'];
  final double _rtmAutoThresh = 0.02;
  final _EmaSmoother _ema = _EmaSmoother();

  // -------- public API --------

  String get rtmPreprocChosen => _rtmPreprocChosen ?? cfg.rtmPreproc;

  Future<void> init() async {
    // Load cfgs
    final yoloCfg = jsonDecode(await rootBundle.loadString('assets/models/configs/yolo.cfg')) as Map;
    final rtmCfg  = jsonDecode(await rootBundle.loadString('assets/models/configs/rtm.cfg'))  as Map;
    try {
      final sk = jsonDecode(await rootBundle.loadString('assets/models/configs/skeletons.json')) as Map;
      if (sk['coco17'] is List) cfg.cocoOrder = (sk['coco17'] as List).cast<String>();
    } catch (_) {}

    // YOLO
    cfg.yoloModelAsset = (yoloCfg['model']?['path'] ?? 'assets/models/yolov8n.onnx') as String;
    final yIn = (yoloCfg['model']?['input'] ?? {}) as Map;
    cfg.yoloInH = (yIn['target_height'] ?? yIn['height'] ?? 640) as int;
    cfg.yoloInW = (yIn['target_width']  ?? yIn['width']  ?? 640) as int;
    final yOut = (yoloCfg['model']?['output'] ?? {}) as Map;
    cfg.personClassId = (yOut['classes']?['person_id'] ?? 0) as int;
    cfg.classScoreActivation = (yOut['class_score_activation'] ?? 'sigmoid') as String;
    final yDom = (yOut['domain'] ?? {}) as Map;
    cfg.yoloUnits  = (yDom['units']  ?? 'normalized') as String;
    cfg.yoloCoords = (yDom['coords'] ?? 'letterbox') as String;
    final yPost = (yoloCfg['model']?['postprocess'] ?? {}) as Map;
    cfg.confTh = (yPost['conf_threshold'] ?? 0.25).toDouble();
    cfg.iouTh  = (yPost['iou_threshold']  ?? 0.45).toDouble();

    // RTM
    cfg.rtmModelAsset = (rtmCfg['model']?['path'] ?? 'assets/models/rtmpose-m_256x192.onnx') as String;
    final rIn = (rtmCfg['model']?['input'] ?? {}) as Map;
    cfg.rtmInH = (rIn['target_height'] ?? rIn['height'] ?? 256) as int;
    cfg.rtmInW = (rIn['target_width']  ?? rIn['width']  ?? 192) as int;
    final rPre = (rIn['preprocess'] ?? {}) as Map;
    cfg.rtmPreproc = (rPre['mode'] ?? 'rgb_255') as String;
    if (cfg.rtmPreproc.endsWith('_ms')) {
      cfg.rtmMean = (rPre['mean'] ?? [123.675,116.28,103.53]).cast<num>().map((e)=>e.toDouble()).toList();
      cfg.rtmStd  = (rPre['std']  ?? [58.395,57.12,57.375]).cast<num>().map((e)=>e.toDouble()).toList();
    } else {
      cfg.rtmMean = [0,0,0];
      cfg.rtmStd  = [255,255,255];
    }
    final rOut = (rtmCfg['model']?['output'] ?? {}) as Map;
    cfg.simccRatio = ((rOut['simcc'] ?? {}) as Map)['split_ratio']?.toDouble() ?? 2.0;

    // Sessions
    _ort = OnnxRuntime();
    final providers2D = <OrtProvider>[
      OrtProvider.CORE_ML, // iOS fast path
      OrtProvider.XNNPACK,
      OrtProvider.CPU,
    ];
    final opts2D = OrtSessionOptions(providers: providers2D);

    _yolo = await _ort.createSessionFromAsset(cfg.yoloModelAsset, options: opts2D);
    _rtm  = await _ort.createSessionFromAsset(cfg.rtmModelAsset,  options: opts2D);

    _yoloInName   = _yolo.inputNames.first;
    _rtmInName    = _rtm.inputNames.first;
    _yoloOutNames = _yolo.outputNames.toList();
    _rtmOutNames  = _rtm.outputNames.toList();

    _yoloBuf = Float32List(3 * cfg.yoloInH * cfg.yoloInW);
    _rtmBuf  = Float32List(3 * cfg.rtmInH  * cfg.rtmInW);

    debugPrint('[Core2D] YOLO in=$_yoloInName outs=${_yoloOutNames.join(",")}');
    debugPrint('[Core2D] RTM  in=$_rtmInName  outs=${_rtmOutNames.join(",")}');
    debugPrint('[Core2D] EP order: [CORE_ML, XNNPACK, CPU]');
  }

  Future<void> dispose() async {
    await _tryDispose(_yolo);
    await _tryDispose(_rtm);
  }

  /// Processes one frame (letterboxed RGB8); returns bbox (normalized by original) and COCO17 XYCs in ORIGINAL pixels.
  Future<({
    List<double> bboxNorm,
    double score,
    CocoXYC cocoXYC,
  })> process(FramePacket pkt) async {
    // --- 1) YOLO (cadenced), map XYXY back to ORIGINAL
    final List<double> boxOrig;
    double bestScore = 0.0;

    if (pkt.frameIdx % yoloStride == 0 || _prevBoxOrig == null) {
      _rgbToCHW(dst: _yoloBuf, rgb: pkt.rgb640x640, w: cfg.yoloInW, h: cfg.yoloInH,
          mean: const [0,0,0], std: const [255,255,255], bgr: false);
      final inVal = await OrtValue.fromList(_yoloBuf, [1, 3, cfg.yoloInH, cfg.yoloInW]);
      final res   = await _yolo.run({_yoloInName: inVal});
      await inVal.dispose();

      final out   = res[_yoloOutNames.first]!;
      final shape = out.shape; // [1,N,C] or [1,C,N]
      final flat  = (await out.asFlattenedList()).cast<num>();
      for (final v in res.values) { await v.dispose(); }

      final dets = _yoloPostprocess(
        flat: flat,
        shape: shape,
        imgW: pkt.origW,
        imgH: pkt.origH,
        netW: cfg.yoloInW,
        netH: cfg.yoloInH,
        ratio: pkt.ratio,
        pads: pkt.pads,
        confTh: cfg.confTh,
        iouTh: cfg.iouTh,
        classId: cfg.personClassId,
        activation: cfg.classScoreActivation,
        outputsAreNormalized: cfg.yoloUnits.toLowerCase() == 'normalized',
        coordsSpaceIsLetterbox: cfg.yoloCoords.toLowerCase() == 'letterbox',
      );

      if (dets.isEmpty) {
        boxOrig = _prevBoxOrig ?? [pkt.origW*0.25, pkt.origH*0.1, pkt.origW*0.75, pkt.origH*0.9];
        bestScore = 0.0;
      } else {
        boxOrig = dets.first.xyxy;
        bestScore = dets.first.score;
        _prevBoxOrig = List<double>.from(boxOrig);
      }
    } else {
      boxOrig   = List<double>.from(_prevBoxOrig!);
      bestScore = 0.0;
    }

    final bboxNorm = [
      boxOrig[0] / pkt.origW, boxOrig[1] / pkt.origH,
      boxOrig[2] / pkt.origW, boxOrig[3] / pkt.origH,
    ];

    // --- 2) RTMPose crop: build crop rect in LETTERBOX coords, crop letterboxed RGB, run RTM, map back to ORIGINAL
    final cropLb = _cropRectInLetterbox(
      xyxyOrig: boxOrig,
      ratio: pkt.ratio,
      pads: pkt.pads,
      outW: cfg.rtmInW,
      outH: cfg.rtmInH,
      scale: 1.25,
    ); // [x,y,w,h] in letterbox px

    // Prepare an Image from the letterboxed RGB buffer for cropping & resize.
    final lbImg = img.Image.fromBytes(
      width: 640,
      height: 640,
      bytes: pkt.rgb640x640.buffer,   // <-- pass ByteBuffer
      numChannels: 3,
      order: img.ChannelOrder.rgb,    // <-- replace Format.rgb
    );

    final crop = img.copyCrop(lbImg,
      x: cropLb[0].round().clamp(0, 639),
      y: cropLb[1].round().clamp(0, 639),
      width:  cropLb[2].round().clamp(1, 640),
      height: cropLb[3].round().clamp(1, 640),
    );
    final resized = img.copyResize(crop, width: cfg.rtmInW, height: cfg.rtmInH, interpolation: img.Interpolation.linear);

    // RTM preprocess auto-tune (first crop only)
    CocoXYC cocoXYC;
    if (_rtmPreprocChosen == null) {
      final first = await _rtmOnce(resized, cropLb, pkt, cfg.rtmPreproc);
      var bestConf = first.$2; var bestMode = cfg.rtmPreproc; var bestPts = first.$1;
      if (bestConf < _rtmAutoThresh) {
        for (final m in _rtmTryModes) {
          if (m == cfg.rtmPreproc) continue;
          final r = await _rtmOnce(resized, cropLb, pkt, m);
          if (r.$2 > bestConf) { bestConf = r.$2; bestMode = m; bestPts = r.$1; }
        }
      }
      _rtmPreprocChosen = bestMode;
      debugPrint('[Core2D] RTM preprocess="$bestMode" (first mean_conf=${bestConf.toStringAsFixed(4)})');
      cocoXYC = bestPts;
    } else {
      cocoXYC = (await _rtmOnce(resized, cropLb, pkt, _rtmPreprocChosen!)).$1;
    }

    // EMA smoothing (confidence-aware)
    cocoXYC = _ema.apply(cocoXYC);

    return (bboxNorm: bboxNorm, score: bestScore, cocoXYC: cocoXYC);
  }

  // -------- internals --------

  // RTM run with given mode; returns COCO XYCs in ORIGINAL pixels (+ mean conf)
  Future<(CocoXYC, double)> _rtmOnce(
    img.Image resizedCrop,
    List<double> rectLb, // [x,y,w,h] in LETTERBOX
    FramePacket pkt,
    String mode,
  ) async {
    final (mean, std, bgr) = _rtmPreprocTripletFor(mode);

    // CHW from resized (RGB)
    _imageToCHW(dst: _rtmBuf, image: resizedCrop, mean: mean, std: std, bgr: bgr);
    final inVal = await OrtValue.fromList(_rtmBuf, [1,3,cfg.rtmInH,cfg.rtmInW]);
    final res   = await _rtm.run({_rtmInName: inVal});
    await inVal.dispose();

    final simccX = res[_rtmOutNames[0]]!;
    final simccY = res[_rtmOutNames[1]]!;
    final xShape = simccX.shape, yShape = simccY.shape;
    final xFlat  = (await simccX.asFlattenedList()).cast<double>();
    final yFlat  = (await simccY.asFlattenedList()).cast<double>();
    for (final v in res.values) { await v.dispose(); }

    // Decode to crop-space (RTM input is 192x256)
    final cropXYC = _simccDecodeFast(xFlat, xShape, yFlat, yShape, splitRatio: cfg.simccRatio);

    // Map crop-space → LETTERBOX → ORIGINAL
    // rectLb = [rx,ry,rw,rh] in letterbox; RTM inW=in 192, inH=256
    final rx = rectLb[0], ry = rectLb[1], rw = rectLb[2], rh = rectLb[3];
    final cocoLb = List.generate(17, (i) {
      final xLb = rx + cropXYC[i][0] * (rw / cfg.rtmInW);
      final yLb = ry + cropXYC[i][1] * (rh / cfg.rtmInH);
      return [xLb, yLb, cropXYC[i][2]];
    });

    final L = pkt.pads[0], T = pkt.pads[1], r = pkt.ratio.toDouble();
    final cocoOrig = cocoLb.map((p) {
      final xo = ((p[0] - L) / r).clamp(0.0, pkt.origW - 1.0);
      final yo = ((p[1] - T) / r).clamp(0.0, pkt.origH - 1.0);
      return [xo, yo, p[2]];
    }).toList(growable: false);

    final meanC = _meanConfFrame(cocoOrig);
    return (cocoOrig, meanC);
  }

  double _meanConfFrame(CocoXYC f) {
    double s = 0; for (final j in f) s += j[2]; return f.isNotEmpty ? s / f.length : 0.0;
  }

  (List<double>, List<double>, bool) _rtmPreprocTripletFor(String mode) {
    switch (mode) {
      case 'rgb_ms': return ([123.675,116.28,103.53], [58.395,57.12,57.375], false);
      case 'bgr_ms': return ([123.675,116.28,103.53], [58.395,57.12,57.375], true);
      case 'bgr_255': return ([0,0,0], [255,255,255], true);
      case 'rgb_255':
      default: return ([0,0,0], [255,255,255], false);
    }
  }

  List<double> _cropRectInLetterbox({
    required List<double> xyxyOrig,
    required double ratio,
    required List<double> pads, // [L,T,R,B]
    required int outW,
    required int outH,
    required double scale,
  }) {
    // Convert ORIG→LETTERBOX
    final L = pads[0], T = pads[1];
    double x1o = xyxyOrig[0], y1o = xyxyOrig[1], x2o = xyxyOrig[2], y2o = xyxyOrig[3];
    double cx = (x1o + x2o) / 2.0, cy = (y1o + y2o) / 2.0;
    double bw = (x2o - x1o) * scale, bh = (y2o - y1o) * scale;
    final targetAR = outW / outH; // (192/256)
    if ((bw / bh) > targetAR) bh = bw / targetAR; else bw = bh * targetAR;

    // expand around center in ORIGINAL, then map to LB
    final x1o2 = cx - bw/2.0, y1o2 = cy - bh/2.0, x2o2 = cx + bw/2.0, y2o2 = cy + bh/2.0;
    final x1 = x1o2 * ratio + L;
    final y1 = y1o2 * ratio + T;
    final x2 = x2o2 * ratio + L;
    final y2 = y2o2 * ratio + T;

    // clamp to 640x640
    final xx1 = x1.clamp(0.0, 639.0);
    final yy1 = y1.clamp(0.0, 639.0);
    final xx2 = x2.clamp(0.0, 639.0);
    final yy2 = y2.clamp(0.0, 639.0);
    final w = math.max(1.0, xx2 - xx1), h = math.max(1.0, yy2 - yy1);
    return [xx1, yy1, w, h];
  }

  void _rgbToCHW({
    required Float32List dst,
    required Uint8List rgb,
    required int w,
    required int h,
    required List<double> mean,
    required List<double> std,
    required bool bgr,
  }) {
    final plane = w * h;
    int di = 0;
    for (int i = 0; i < rgb.length; i += 3) {
      double r = rgb[i + 0].toDouble();
      double g = rgb[i + 1].toDouble();
      double b = rgb[i + 2].toDouble();
      if (bgr) { final t = r; r = b; b = t; }
      r = (r - mean[0]) / std[0];
      g = (g - mean[1]) / std[1];
      b = (b - mean[2]) / std[2];
      dst[di]            = r;
      dst[di + plane]    = g;
      dst[di + 2*plane]  = b;
      di++;
    }
  }

  void _imageToCHW({
    required Float32List dst,
    required img.Image image,
    required List<double> mean,
    required List<double> std,
    required bool bgr,
  }) {
    final w = image.width, h = image.height, plane = w * h;
    int di = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = image.getPixel(x, y);
        double r = (p.r as num).toDouble();
        double g = (p.g as num).toDouble();
        double b = (p.b as num).toDouble();
        if (bgr) { final t = r; r = b; b = t; }
        r = (r - mean[0]) / std[0];
        g = (g - mean[1]) / std[1];
        b = (b - mean[2]) / std[2];
        dst[di]            = r;
        dst[di + plane]    = g;
        dst[di + 2*plane]  = b;
        di++;
      }
    }
  }

  List<YDet> _yoloPostprocess({
    required List<num> flat,
    required List<int> shape,
    required int imgW,
    required int imgH,
    required int netW,
    required int netH,
    required double ratio,
    required List<double> pads, // [L,T,R,B]
    required double confTh,
    required double iouTh,
    required int classId,
    required String activation,
    required bool outputsAreNormalized,
    required bool coordsSpaceIsLetterbox,
  }) {
    if (shape[0] != 1) {
      throw ArgumentError('Only batch=1 supported, got $shape');
    }
    late final int n; late final int c; late final bool transposed;
    if (shape[1] > shape[2]) { n = shape[1]; c = shape[2]; transposed = false; }
    else { c = shape[1]; n = shape[2]; transposed = true; }

    final boxes = List<double>.filled(n*4, 0.0);
    final cls   = List<double>.filled(n*(c-4), 0.0);
    if (!transposed) {
      for (int i=0;i<n;i++){
        for (int j=0;j<4;j++){ boxes[i*4+j] = flat[i*c+j].toDouble(); }
        for (int j=0;j<(c-4);j++){ cls[i*(c-4)+j] = flat[i*c+4+j].toDouble(); }
      }
    } else {
      for (int i=0;i<n;i++){
        for (int j=0;j<4;j++){ boxes[i*4+j] = flat[j*n+i].toDouble(); }
        for (int j=0;j<(c-4);j++){ cls[i*(c-4)+j] = flat[(4+j)*n+i].toDouble(); }
      }
    }
    if (activation == 'sigmoid') {
      for (int i=0;i<cls.length;i++) { final v = cls[i]; cls[i] = 1.0/(1.0+math.exp(-v)); }
    }

    final L = pads[0], T = pads[1];
    final dets = <YDet>[];
    for (int i=0;i<n;i++){
      double bestScore = -1e9; int bestK = -1;
      for (int k=0;k<(c-4);k++){
        final v = cls[i*(c-4)+k];
        if (v > bestScore){ bestScore = v; bestK = k; }
      }
      if (bestK != classId) continue;
      if (bestScore < confTh) continue;

      double cx = boxes[i*4+0], cy = boxes[i*4+1];
      double bw = boxes[i*4+2], bh = boxes[i*4+3];
      if (outputsAreNormalized) {
        cx *= netW; cy *= netH; bw *= netW; bh *= netH;
      }
      final x1lb = cx - bw/2, y1lb = cy - bh/2, x2lb = cx + bw/2, y2lb = cy + bh/2;

      double x1, y1, x2, y2;
      if (coordsSpaceIsLetterbox) {
        x1 = ((x1lb - L) / ratio).clamp(0.0, imgW - 1.0);
        y1 = ((y1lb - T) / ratio).clamp(0.0, imgH - 1.0);
        x2 = ((x2lb - L) / ratio).clamp(0.0, imgW - 1.0);
        y2 = ((y2lb - T) / ratio).clamp(0.0, imgH - 1.0);
      } else {
        x1 = x1lb.clamp(0.0, imgW - 1.0);
        y1 = y1lb.clamp(0.0, imgH - 1.0);
        x2 = x2lb.clamp(0.0, imgW - 1.0);
        y2 = y2lb.clamp(0.0, imgH - 1.0);
      }
      dets.add(YDet([x1,y1,x2,y2], bestScore));
    }

    dets.sort((a,b)=>b.score.compareTo(a.score));
    final keep = <YDet>[];
    for (final d in dets) {
      bool sup = false;
      for (final k in keep) { if (_iou(d.xyxy, k.xyxy) > iouTh) { sup = true; break; } }
      if (!sup) keep.add(d);
    }
    return keep;
  }

  double _iou(List<double> a, List<double> b) {
    final ax1=a[0], ay1=a[1], ax2=a[2], ay2=a[3];
    final bx1=b[0], by1=b[1], bx2=b[2], by2=b[3];
    final ix1 = math.max(ax1, bx1), iy1 = math.max(ay1, by1);
    final ix2 = math.min(ax2, bx2), iy2 = math.min(ay2, by2);
    final iw = math.max(0, ix2 - ix1), ih = math.max(0, iy2 - iy1);
    final inter = iw * ih;
    final aArea = math.max(0, ax2 - ax1) * math.max(0, ay2 - ay1);
    final bArea = math.max(0, bx2 - bx1) * math.max(0, by2 - by1);
    return inter / (aArea + bArea - inter + 1e-6);
  }

  List<List<double>> _simccDecodeFast(
    List<double> x, List<int> xs, List<double> y, List<int> ys, {required double splitRatio}) {
    final K = xs[1], Lx = xs[2], Ly = ys[2];
    final out = List.generate(K, (_) => [0.0,0.0,0.0]);
    for (int k = 0; k < K; k++) {
      final xo = k * Lx, yo = k * Ly;
      double xmax = -1e300, ymax = -1e300;
      for (int i = 0; i < Lx; i++) xmax = math.max(xmax, x[xo + i]);
      for (int i = 0; i < Ly; i++) ymax = math.max(ymax, y[yo + i]);
      double xsum = 0, ysum = 0, xpeak = -1, ypeak = -1; int xi = 0, yi = 0;
      for (int i = 0; i < Lx; i++) { final p = math.exp(x[xo + i] - xmax); if (p > xpeak) { xpeak = p; xi = i; } xsum += p; }
      for (int i = 0; i < Ly; i++) { final p = math.exp(y[yo + i] - ymax); if (p > ypeak) { ypeak = p; yi = i; } ysum += p; }
      out[k][0] = xi / splitRatio;
      out[k][1] = yi / splitRatio;
      final px = (xsum > 0 ? xpeak / xsum : 0.0);
      final py = (ysum > 0 ? ypeak / ysum : 0.0);
      out[k][2] = 0.5 * (px + py);
    }
    return out;
  }

  static List<List<double>> coco17ToH36M17(List<List<double>> c) {
    final nose=c[0], leye=c[1], reye=c[2];
    final lsho=c[5], rsho=c[6], lelb=c[7], relb=c[8], lwri=c[9], rwri=c[10];
    final lhip=c[11], rhip=c[12], lknee=c[13], rknee=c[14], lank=c[15], rank=c[16];
    final pelvis=[(lhip[0]+rhip[0])/2,(lhip[1]+rhip[1])/2,(lhip[2]+rhip[2])/2];
    final neck  =[(lsho[0]+rsho[0])/2,(lsho[1]+rsho[1])/2,(lsho[2]+rsho[2])/2];
    final spine1=[(pelvis[0]+neck[0])/2,(pelvis[1]+neck[1])/2,(pelvis[2]+neck[2])/2];
    final head  =(leye[2]>0 && reye[2]>0)
        ? [(leye[0]+reye[0])/2,(leye[1]+reye[1])/2,(leye[2]+reye[2])/2]
        : [nose[0],nose[1],nose[2]];
    final site  =[nose[0],nose[1],nose[2]];

    final h = List.generate(17, (_)=>[0.0,0.0,0.0]);
    h[0]=pelvis; h[1]=rhip; h[2]=rknee; h[3]=rank; h[4]=lhip; h[5]=lknee; h[6]=lank;
    h[7]=spine1; h[8]=neck; h[9]=head; h[10]=site;
    h[11]=lsho; h[12]=lelb; h[13]=lwri; h[14]=rsho; h[15]=relb; h[16]=rwri;
    return h;
  }

  Future<void> _tryDispose(Object? sess) async {
    if (sess == null) return;
    try { final r = (sess as dynamic).dispose(); if (r is Future) await r; return; } catch (_) {}
    try { final r = (sess as dynamic).release(); if (r is Future) await r; return; } catch (_) {}
    try { final r = (sess as dynamic).close();   if (r is Future) await r; return; } catch (_) {}
  }
}
