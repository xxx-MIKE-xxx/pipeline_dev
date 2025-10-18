// Output-equivalent pipeline to your Python run_rgb_to_motionbert3d_patched_v5.py
// - Reads JSON cfgs from assets/models/configs
// - YOLO -> RTMPose (SimCC) -> COCO->H36M -> MotionBERT
// - Saves yolo_out.json, rtm_out.json, motionbert_out.json to Documents/run_<ts>/
//
// Requires pubspec deps:
//   flutter_onnxruntime, image, ffmpeg_kit_flutter_new, path_provider, image_picker, share_plus
//
// iOS tip: if you’re on iOS 18+, prefer --profile / --release for now due to a
// temporary debug-mode issue in Flutter (see Flutter issue tracker).

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';

typedef ProgressCb = void Function(double progress, String stage);

class _Cfg {
  // YOLO
  late String yoloModelAsset;
  int yoloInH = 640, yoloInW = 640;
  int personClassId = 0;
  double confTh = 0.25, iouTh = 0.45;
  String classScoreActivation = 'sigmoid'; // or 'none'
  String yoloUnits = 'normalized'; // or 'pixels'
  String yoloCoords = 'letterbox'; // or 'original'

  // RTM
  late String rtmModelAsset;
  int rtmInH = 256, rtmInW = 192;
  // preprocess: one of: rgb_255, bgr_255, rgb_ms, bgr_ms
  String rtmPreproc = 'rgb_255';
  List<double> rtmMean = [0, 0, 0];
  List<double> rtmStd = [255, 255, 255];
  double simccRatio = 2.0;

  // MotionBERT
  late String mbModelAsset;
  int T = 243;
  bool wrapPad = true;
  bool rootRel = false;

  // skeleton orders (fallback defaults)
  List<String> cocoOrder = const [
    "Nose","LEye","REye","LEar","REar",
    "LShoulder","RShoulder","LElbow","RElbow",
    "LWrist","RWrist","LHip","RHip",
    "LKnee","RKnee","LAnkle","RAnkle"
  ];
  List<String> h36mOrder = const [
    "Pelvis","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle",
    "Spine1","Neck","Head","Site","LShoulder","LElbow","LWrist",
    "RShoulder","RElbow","RWrist"
  ];
}

class OutputEquivRunner {
  final _Cfg cfg = _Cfg();
  late final OnnxRuntime _ort;
  late final OrtSession _yolo, _rtm, _mb;

  late final String _yoloInName, _rtmInName, _mbInName;
  late final List<String> _yoloOutNames, _rtmOutNames, _mbOutNames;

  List<double>? _prevBox; // xyxy pixels from last frame

  Future<void> initFromAssets({ProgressCb? onProgress}) async {
    onProgress?.call(0.01, 'Loading configs');

    // Load cfgs
    final yoloCfg = jsonDecode(await rootBundle.loadString('assets/models/configs/yolo.cfg')) as Map;
    final rtmCfg  = jsonDecode(await rootBundle.loadString('assets/models/configs/rtm.cfg')) as Map;
    final mbCfg   = jsonDecode(await rootBundle.loadString('assets/models/configs/motionbert.cfg')) as Map;

    // skeletons.json (optional orders)
    try {
      final sk = jsonDecode(await rootBundle.loadString('assets/models/configs/skeletons.json')) as Map;
      if (sk['coco17'] is List) {
        cfg.cocoOrder = (sk['coco17'] as List).cast<String>();
      }
      if (sk['h36m17'] is List) {
        cfg.h36mOrder = (sk['h36m17'] as List).cast<String>();
      }
    } catch (_) {/* keep defaults */}

    // YOLO cfg
    cfg.yoloModelAsset = (yoloCfg['model']?['path'] ?? 'assets/models/yolo.onnx') as String;
    final yIn = (yoloCfg['model']?['input'] ?? {}) as Map;
    cfg.yoloInH = (yIn['target_height'] ?? yIn['height'] ?? 640) as int;
    cfg.yoloInW = (yIn['target_width']  ?? yIn['width']  ?? 640) as int;
    final yOut = (yoloCfg['model']?['output'] ?? {}) as Map;
    cfg.personClassId = (yOut['classes']?['person_id'] ?? 0) as int;
    cfg.classScoreActivation = (yOut['class_score_activation'] ?? 'sigmoid') as String;
    final yDom = (yOut['domain'] ?? {}) as Map;
    cfg.yoloUnits = (yDom['units'] ?? 'normalized') as String;
    cfg.yoloCoords= (yDom['coords'] ?? 'letterbox') as String;
    final yPost = (yoloCfg['postprocess'] ?? {}) as Map;
    cfg.confTh = (yPost['conf_threshold'] ?? 0.25).toDouble();
    cfg.iouTh  = (yPost['iou_threshold']  ?? 0.45).toDouble();

    // RTM cfg
    cfg.rtmModelAsset = (rtmCfg['model']?['path'] ?? 'assets/models/rtmpose.onnx') as String;
    final rIn = (rtmCfg['model']?['input'] ?? {}) as Map;
    cfg.rtmInH = (rIn['target_height'] ?? rIn['height'] ?? 256) as int;
    cfg.rtmInW = (rIn['target_width']  ?? rIn['width']  ?? 192) as int;
    final rPre = (rIn['preprocess'] ?? {}) as Map;
    cfg.rtmPreproc = (rPre['mode'] ?? 'rgb_255') as String;
    if (cfg.rtmPreproc.endsWith('_ms')) {
      cfg.rtmMean = (rPre['mean'] ?? [123.675, 116.28, 103.53]).cast<num>().map((e) => e.toDouble()).toList();
      cfg.rtmStd  = (rPre['std']  ?? [58.395, 57.12, 57.375]).cast<num>().map((e) => e.toDouble()).toList();
    } else {
      cfg.rtmMean = [0,0,0];
      cfg.rtmStd  = [255,255,255];
    }
    final rOut = (rtmCfg['model']?['output'] ?? {}) as Map;
    cfg.simccRatio = ((rOut['simcc'] ?? {}) as Map)['split_ratio']?.toDouble() ?? 2.0;

    // MB cfg
    cfg.mbModelAsset = (mbCfg['model']?['path'] ?? 'assets/models/motionbert.onnx') as String;
    cfg.T = (mbCfg['model']?['input']?['sequence_length'] ?? 243) as int;
    cfg.rootRel = (mbCfg['model']?['output']?['root_relative'] ?? false) as bool;
    cfg.wrapPad = (mbCfg['runtime']?['wrap_pad_sequence'] ?? true) as bool;

    // Create sessions
    onProgress?.call(0.06, 'Creating ONNX sessions');
    _ort = OnnxRuntime();
    final opts = OrtSessionOptions(providers: const [OrtProvider.XNNPACK]);
    _yolo = await _ort.createSessionFromAsset(cfg.yoloModelAsset, options: opts);
    _rtm  = await _ort.createSessionFromAsset(cfg.rtmModelAsset,  options: opts);
    _mb   = await _ort.createSessionFromAsset(cfg.mbModelAsset,   options: opts);

    _yoloInName  = _yolo.inputNames.first;
    _rtmInName   = _rtm.inputNames.first;
    _mbInName    = _mb.inputNames.first;
    _yoloOutNames= _yolo.outputNames.toList();
    _rtmOutNames = _rtm.outputNames.toList(); // expect 2
    _mbOutNames  = _mb.outputNames.toList();
  }

  /// Runs the full pipeline on a video file path. Returns output folder path.
  Future<String> runOnVideo(File videoFile, {ProgressCb? onProgress}) async {
    final docs = await getApplicationDocumentsDirectory();
    final outDir = Directory('${docs.path}/run_${_ts()}');
    await outDir.create(recursive: true);

    // Extract frames to temp
    onProgress?.call(0.08, 'Extracting frames');
    final framesDir = await _extractFrames(videoFile);
    final frameFiles = (await framesDir.list().toList())
      ..sort((a,b)=>a.path.compareTo(b.path));
    final pngs = <img.Image>[];
    for (final e in frameFiles) {
      if (e is File && e.path.endsWith('.png')) {
        final im = img.decodePng(await e.readAsBytes());
        if (im != null) pngs.add(im);
      }
    }
    if (pngs.isEmpty) {
      throw Exception('No frames decoded from video.');
    }

    final total = pngs.length;
    onProgress?.call(0.12, 'Running YOLO/RTMPose');

    // Globals
    final inW = pngs.first.width, inH = pngs.first.height;

    // Accumulators for JSON parity
    final yoloBboxesNorm = <List<double>>[];
    final yoloScores = <double>[];
    final seqCocoXYC = <List<List<double>>>[];  // [F][17][3]
    final seqH36MXYC = <List<List<double>>>[];  // [F][17][3]

    _prevBox = null;

    // Loop frames
    for (int fi = 0; fi < pngs.length; fi++) {
      final frame = pngs[fi];

      // --- YOLO letterbox -> infer -> undo letterbox ---
      final (lb, ratio, pads) = _letterbox(
        frame, newH: cfg.yoloInH, newW: cfg.yoloInW, padColor: img.ColorRgb8(114,114,114));
      final yoloIn = _toCHWFloat(lb, mean: const [0,0,0], std: const [255,255,255], bgr: false);
      final yoloFeeds = {_yoloInName: await OrtValue.fromList(yoloIn, [1,3,cfg.yoloInH,cfg.yoloInW])};
      final yoloRes = await _yolo.run(yoloFeeds);
      final yoloOut = yoloRes[_yoloOutNames.first]!;
      final yShape = yoloOut.shape; // [1,N,C] or [1,C,N]
      final yData = (await yoloOut.asFlattenedList()).cast<num>();
      for (final v in yoloRes.values) { await v.dispose(); }

      final dets = _yoloPostprocess(
        flat: yData, shape: yShape, imgW: frame.width, imgH: frame.height,
        ratio: ratio, pads: pads, confTh: cfg.confTh, iouTh: cfg.iouTh,
        classId: cfg.personClassId, activation: cfg.classScoreActivation);

      // Best or fallback
      List<double> box;
      double score = 0.0;
      if (dets.isEmpty) {
        if (_prevBox != null) {
          box = List<double>.from(_prevBox!);
        } else {
          box = [inW*0.25, inH*0.1, inW*0.75, inH*0.9];
        }
      } else {
        box = dets.first.xyxy;
        score = dets.first.score;
        _prevBox = List<double>.from(box);
      }
      // normalized bbox (by original W/H)
      yoloBboxesNorm.add([box[0]/inW, box[1]/inH, box[2]/inW, box[3]/inH]);
      yoloScores.add(score);

      // --- Crop to 256x192 → RTMPose (SimCC decode) ---
      final cropRes = _cropToAspect(frame, box, outH: cfg.rtmInH, outW: cfg.rtmInW, scale: 1.25);
      final crop = cropRes.$1; 
      final rect = cropRes.$2; // [rx,ry,rw,rh] in original pixels

      if (crop == null || rect == null) {
        if (seqCocoXYC.isNotEmpty) {
          seqCocoXYC.add(_deepCopy2D(seqCocoXYC.last));
          seqH36MXYC.add(_deepCopy2D(seqH36MXYC.last));
        } else {
          seqCocoXYC.add(_zeros173());
          seqH36MXYC.add(_zeros173());
        }
      } else {
        final (mean, std, bgr) = _rtmPreprocTriplet();
        final rtmInput = _toCHWFloat(crop, mean: mean, std: std, bgr: bgr);
        final rtmFeeds = {_rtmInName: await OrtValue.fromList(rtmInput, [1,3,cfg.rtmInH,cfg.rtmInW])};
        final rtmRes = await _rtm.run(rtmFeeds);

        // Expect two outputs: simcc_x [1,K,Lx], simcc_y [1,K,Ly]
        final simccX = rtmRes[_rtmOutNames[0]]!;
        final simccY = rtmRes[_rtmOutNames[1]]!;
        final xShape = simccX.shape; 
        final yShape2 = simccY.shape;
        final xFlat = (await simccX.asFlattenedList()).cast<double>();
        final yFlat = (await simccY.asFlattenedList()).cast<double>();
        for (final v in rtmRes.values) { await v.dispose(); }

        final cocoCrop = _simccDecode(xFlat, xShape, yFlat, yShape2, splitRatio: cfg.simccRatio); // [17][3] in crop

        // Map crop→image pixels via rect [rx,ry,rw,rh]
        final rx = rect[0], ry = rect[1], rw = rect[2], rh = rect[3];
        final cocoXYC = List.generate(17, (i) {
          final x = rx + cocoCrop[i][0] * (rw / cfg.rtmInW);
          final y = ry + cocoCrop[i][1] * (rh / cfg.rtmInH);
          return [x, y, cocoCrop[i][2]];
        });

        // COCO→H36M mapping
        final h36mXYC = _coco17ToH36M17(cocoXYC);

        seqCocoXYC.add(cocoXYC);
        seqH36MXYC.add(h36mXYC);
      }

      // progress window: 12%..90%
      final p = 0.12 + 0.78 * (fi + 1) / total;
      onProgress?.call(p, 'Frames ${fi + 1}/$total');
    }

    // --- Build MotionBERT input sequence: H36M XY normalized to [-1,1], conf untouched ---
    onProgress?.call(0.91, 'Preparing MotionBERT input');
    final mbSeq = _buildMbSequence(seqH36MXYC, inW, inH, T: cfg.T, wrapPad: cfg.wrapPad);

    // --- MotionBERT inference ---
    onProgress?.call(0.94, 'Running MotionBERT');
    final mbFlat = <double>[];
    for (int t = 0; t < cfg.T; t++) {
      for (int j = 0; j < 17; j++) {
        mbFlat.addAll(mbSeq[t][j]); // [xn, yn, c]
      }
    }
    final mbFeeds = {_mbInName: await OrtValue.fromList(mbFlat, [1, cfg.T, 17, 3])};
    final mbRes = await _mb.run(mbFeeds);
    final mbTensor = mbRes[_mbOutNames.first]!;
    final mbShape = mbTensor.shape; // [1,T,17,3]
    final mbData = (await mbTensor.asFlattenedList()).cast<double>();
    for (final v in mbRes.values) { await v.dispose(); }
    final coords3d = _reshape3d(mbData, mbShape[1], mbShape[2], mbShape[3]); // [T][17][3]
    if (cfg.rootRel) {
      for (int t = 0; t < coords3d.length; t++) {
        coords3d[t][0] = [0.0,0.0,0.0];
      }
    }

    // --- Write JSON artifacts ---
    onProgress?.call(0.98, 'Writing JSONs');
    final videoBase = videoFile.uri.pathSegments.last;
    await _writeJson(outDir.path, 'yolo_out.json', {
      "video": videoBase,
      "frames": yoloBboxesNorm.length,
      "bboxes_norm": yoloBboxesNorm,
      "scores": yoloScores,
      "yolo_output_units": "normalized",
      "yolo_output_coords": cfg.yoloCoords,
      "normalized_by": ["width","height"]
    });

    await _writeJson(outDir.path, 'rtm_out.json', {
      "video": videoBase,
      "frames": seqCocoXYC.length,
      "coco_order": cfg.cocoOrder,
      "coords_2d": seqCocoXYC
    });

    await _writeJson(outDir.path, 'motionbert_out.json', {
      "video": videoBase,
      "T": cfg.T,
      "h36m_order": cfg.h36mOrder,
      "coords_3d": coords3d
    });

    onProgress?.call(1.0, 'Done');
    return outDir.path;
  }

  // ---------- Helpers ----------

  Future<Directory> _extractFrames(File video) async {
    final tmp = await getTemporaryDirectory();
    final dir = Directory('${tmp.path}/frames_${_ts()}');
    await dir.create(recursive: true);
    final cmd = ['-hide_banner','-y','-i', video.path, '${dir.path}/frame_%05d.png'].join(' ');
    await FFmpegKit.execute(cmd);
    return dir;
  }

  (img.Image, double, List<double>) _letterbox(
    img.Image src, {required int newH, required int newW, required img.Color padColor}) {
    final h = src.height, w = src.width;
    final r = math.min(newW / w, newH / h);
    final nh = (h * r).round(), nw = (w * r).round();

    final resized = img.copyResize(src, width: nw, height: nh, interpolation: img.Interpolation.linear);
    final canvas = img.Image(width: newW, height: newH);
    img.fill(canvas, color: padColor);
    final dw = ((newW - nw) / 2).floor();
    final dh = ((newH - nh) / 2).floor();
    img.compositeImage(canvas, resized, dstX: dw, dstY: dh);

    final pads = [dw.toDouble(), dh.toDouble(), (newW - nw - dw).toDouble(), (newH - nh - dh).toDouble()];
    return (canvas, r, pads);
  }

  Float32List _toCHWFloat(img.Image image, {required List<double> mean, required List<double> std, required bool bgr}) {
    final h = image.height, w = image.width;
    final out = Float32List(3 * h * w);
    var dx = 0;
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
        final idx = dx;
        out[idx] = r;
        out[idx + h * w] = g;
        out[idx + 2 * h * w] = b;
        dx++;
      }
    }
    return out;
  }

  (img.Image?, List<double>?) _cropToAspect(
    img.Image src, List<double> boxXYXY, {required int outH, required int outW, double scale = 1.25}) {
    final w = src.width.toDouble(), h = src.height.toDouble();
    double x1 = boxXYXY[0], y1 = boxXYXY[1], x2 = boxXYXY[2], y2 = boxXYXY[3];
    final cx = (x1 + x2) / 2.0, cy = (y1 + y2) / 2.0;
    double bw = (x2 - x1) * scale, bh = (y2 - y1) * scale;

    final targetAR = outW / outH; // 192/256
    if (bw / bh > targetAR) bh = bw / targetAR; else bw = bh * targetAR;

    final nx1 = (cx - bw / 2).round().clamp(0, w - 1).toInt();
    final ny1 = (cy - bh / 2).round().clamp(0, h - 1).toInt();
    final nx2 = (cx + bw / 2).round().clamp(0, w - 1).toInt();
    final ny2 = (cy + bh / 2).round().clamp(0, h - 1).toInt();
    final cw = math.max(1, nx2 - nx1), ch = math.max(1, ny2 - ny1);
    if (cw <= 1 || ch <= 1) return (null, null);

    final crop = img.copyCrop(src, x: nx1, y: ny1, width: cw, height: ch);
    final resized = img.copyResize(crop, width: outW, height: outH, interpolation: img.Interpolation.linear);
    return (resized, [nx1.toDouble(), ny1.toDouble(), cw.toDouble(), ch.toDouble()]);
  }

  List<YDet> _yoloPostprocess({
    required List<num> flat,
    required List<int> shape,
    required int imgW,
    required int imgH,
    required double ratio,
    required List<double> pads, // [L,T,R,B]
    required double confTh,
    required double iouTh,
    required int classId,
    required String activation, // 'sigmoid' or 'none'
  }) {
    if (shape[0] != 1) { throw ArgumentError('Only batch=1 supported, got $shape'); }

    late final int n; late final int c; late final bool transposed;
    if (shape[1] > shape[2]) { n = shape[1]; c = shape[2]; transposed = false; }
    else { c = shape[1]; n = shape[2]; transposed = true; }

    final boxes = List<double>.filled(n*4, 0);
    final cls   = List<double>.filled(n*(c-4), 0);

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
      for (int i=0;i<cls.length;i++){
        final v = cls[i];
        cls[i] = 1.0 / (1.0 + math.exp(-v));
      }
    }

    final dets = <YDet>[];
    final padL = pads[0], padT = pads[1];

    for (int i=0;i<n;i++){
      // best class
      double bestScore = -1e9; int bestK = -1;
      for (int k=0;k<(c-4);k++){
        final v = cls[i*(c-4)+k];
        if (v > bestScore){ bestScore = v; bestK = k; }
      }
      if (bestK != classId) continue;
      if (bestScore < confTh) continue;

      final cx = boxes[i*4+0], cy = boxes[i*4+1];
      final bw = boxes[i*4+2], bh = boxes[i*4+3];
      final x1lb = cx - bw/2, y1lb = cy - bh/2, x2lb = cx + bw/2, y2lb = cy + bh/2;

      // undo letterbox
      final x1 = _clampD((x1lb - padL) / ratio, 0.0, imgW - 1.0);
      final y1 = _clampD((y1lb - padT) / ratio, 0.0, imgH - 1.0);
      final x2 = _clampD((x2lb - padL) / ratio, 0.0, imgW - 1.0);
      final y2 = _clampD((y2lb - padT) / ratio, 0.0, imgH - 1.0);

      dets.add(YDet([x1,y1,x2,y2], bestScore));
    }

    // NMS
    dets.sort((a,b)=>b.score.compareTo(a.score));
    final keep = <YDet>[];
    for (final d in dets){
      bool sup = false;
      for (final k in keep){
        if (_iou(d.xyxy, k.xyxy) > iouTh){ sup = true; break; }
      }
      if (!sup) keep.add(d);
    }
    return keep;
  }

  List<List<double>> _simccDecode(List<double> x, List<int> xs, List<double> y, List<int> ys, {required double splitRatio}) {
    // x: [1,K,Lx], y: [1,K,Ly]
    final K = xs[1], Lx = xs[2], Ly = ys[2];
    final out = List.generate(K, (_) => [0.0,0.0,0.0]);
    for (int k=0;k<K;k++){
      // softmax + argmax for x
      final xo = k*Lx;
      double xmax=-1e30; for(int i=0;i<Lx;i++){ if (x[xo+i]>xmax) xmax=x[xo+i]; }
      double xsum=0.0; for(int i=0;i<Lx;i++){ xsum += math.exp(x[xo+i]-xmax); }
      int xi=0; double xbest=0.0;
      for(int i=0;i<Lx;i++){
        final p = math.exp(x[xo+i]-xmax)/xsum;
        if (p> xbest){ xbest=p; xi=i; }
      }

      // softmax + argmax for y
      final yo = k*Ly;
      double ymax=-1e30; for(int i=0;i<Ly;i++){ if (y[yo+i]>ymax) ymax=y[yo+i]; }
      double ysum=0.0; for(int i=0;i<Ly;i++){ ysum += math.exp(y[yo+i]-ymax); }
      int yi=0; double ybest=0.0;
      for(int i=0;i<Ly;i++){
        final p = math.exp(y[yo+i]-ymax)/ysum;
        if (p> ybest){ ybest=p; yi=i; }
      }
      final xx = xi.toDouble()/splitRatio;
      final yy = yi.toDouble()/splitRatio;
      out[k][0]=xx; out[k][1]=yy; out[k][2]=math.sqrt(xbest*ybest);
    }
    return out;
  }

  List<List<double>> _coco17ToH36M17(List<List<double>> cocoXYC) {
    // indices (COCO):
    // 0 nose,1 leye,2 reye,3 lear,4 rear,5 lsho,6 rsho,7 lelb,8 relb,9 lwri,10 rwri,
    // 11 lhip,12 rhip,13 lknee,14 rknee,15 lank,16 rank
    List<double> nose = cocoXYC[0],
        leye = cocoXYC[1], reye = cocoXYC[2];
    List<double> lsho = cocoXYC[5], rsho = cocoXYC[6];
    List<double> lelb = cocoXYC[7], relb = cocoXYC[8];
    List<double> lwri = cocoXYC[9], rwri = cocoXYC[10];
    List<double> lhip = cocoXYC[11], rhip = cocoXYC[12];
    List<double> lknee = cocoXYC[13], rknee = cocoXYC[14];
    List<double> lank = cocoXYC[15], rank = cocoXYC[16];

    final pelvis = [(lhip[0] + rhip[0])/2.0, (lhip[1] + rhip[1])/2.0, (lhip[2] + rhip[2])/2.0];
    final neck   = [(lsho[0] + rsho[0])/2.0, (lsho[1] + rsho[1])/2.0, (lsho[2] + rsho[2])/2.0];
    final spine1 = [(pelvis[0] + neck[0])/2.0, (pelvis[1] + neck[1])/2.0, (pelvis[2] + neck[2])/2.0];
    final head   = (leye[2]>0 && reye[2]>0)
        ? [(leye[0] + reye[0])/2.0, (leye[1] + reye[1])/2.0, (leye[2] + reye[2])/2.0]
        : [nose[0], nose[1], nose[2]];
    final site = [nose[0], nose[1], nose[2]];

    final h = List.generate(17, (_) => [0.0,0.0,0.0]);
    h[0]  = pelvis;
    h[1]  = rhip;  h[2]  = rknee; h[3]  = rank;
    h[4]  = lhip;  h[5]  = lknee; h[6]  = lank;
    h[7]  = spine1;h[8]  = neck;  h[9]  = head; h[10] = site;
    h[11] = lsho;  h[12] = lelb;  h[13] = lwri;
    h[14] = rsho;  h[15] = relb;  h[16] = rwri;
    return h;
  }

  List<List<List<double>>> _buildMbSequence(
    List<List<List<double>>> seqH36MXYC, int inW, int inH, {required int T, required bool wrapPad}) {
    final F = seqH36MXYC.length;
    final seq = <List<List<double>>>[];
    for (int t=0;t<F;t++){
      final frame = <List<double>>[];
      final s = math.min(inW,inH)/2.0;
      final cx = inW/2.0, cy = inH/2.0;
      for (int j=0;j<17;j++){
        final x = seqH36MXYC[t][j][0], y = seqH36MXYC[t][j][1], c = seqH36MXYC[t][j][2];
        final xn = (x - cx) / s;
        final yn = (y - cy) / s;
        frame.add([xn, yn, c]);
      }
      seq.add(frame);
    }
    if (wrapPad) {
      if (seq.length < T) {
        final last = seq.isNotEmpty ? _deepCopy2D(seq.last) : _zeros173();
        while (seq.length < T) seq.add(_deepCopy2D(last));
      } else if (seq.length > T) {
        while (seq.length > T) { seq.removeLast(); }
      }
    } else {
      if (seq.length != T) {
        throw Exception('MotionBERT expects T=$T, got ${seq.length}. Enable wrapPad in cfg to pad/truncate.');
      }
    }
    return seq;
  }

  List<List<List<double>>> _reshape3d(List<double> flat, int T, int J, int C){
    final out = List.generate(T, (_) => List.generate(J, (_)=>List.filled(C,0.0)));
    int idx=0;
    for (int t=0;t<T;t++){
      for (int j=0;j<J;j++){
        for (int c=0;c<C;c++){
          out[t][j][c] = flat[idx++];
        }
      }
    }
    return out;
  }

  Future<void> _writeJson(String dir, String name, Map<String,Object?> payload) async {
    final f = File('$dir/$name');
    await f.writeAsString(const JsonEncoder.withIndent('  ').convert(payload));
  }

  String _ts(){
    final n = DateTime.now();
    String two(int v)=>v.toString().padLeft(2,'0');
    return '${n.year}${two(n.month)}${two(n.day)}_${two(n.hour)}${two(n.minute)}${two(n.second)}';
  }

  double _clampD(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);

  double _iou(List<double> a, List<double> b) {
    final ax1=a[0], ay1=a[1], ax2=a[2], ay2=a[3];
    final bx1=b[0], by1=b[1], bx2=b[2], by2=b[3];
    final interX1 = math.max(ax1, bx1), interY1 = math.max(ay1, by1);
    final interX2 = math.min(ax2, bx2), interY2 = math.min(ay2, by2);
    final interW = math.max(0, interX2 - interX1), interH = math.max(0, interY2 - interY1);
    final inter = interW * interH;
    final aArea = math.max(0, ax2 - ax1) * math.max(0, ay2 - ay1);
    final bArea = math.max(0, bx2 - bx1) * math.max(0, by2 - by1);
    return inter / (aArea + bArea - inter + 1e-6);
  }

  (List<double>, List<double>, bool) _rtmPreprocTriplet(){
    switch (cfg.rtmPreproc) {
      case 'rgb_ms': return (cfg.rtmMean, cfg.rtmStd, false);
      case 'bgr_ms': return (cfg.rtmMean, cfg.rtmStd, true);
      case 'bgr_255': return ([0,0,0], [255,255,255], true);
      case 'rgb_255':
      default: return ([0,0,0], [255,255,255], false);
    }
  }

  List<List<double>> _deepCopy2D(List<List<double>> src) =>
      List<List<double>>.generate(src.length, (i) => List<double>.from(src[i]), growable: false);

  List<List<double>> _zeros173() =>
      List<List<double>>.generate(17, (_) => [0.0, 0.0, 0.0], growable: false);
}

class YDet {
  YDet(this.xyxy, this.score);
  final List<double> xyxy;
  final double score;
}

// -------------------------------
// Public API convenience:
// -------------------------------

/// Initialize the runner (reads cfgs and creates ORT sessions)
Future<OutputEquivRunner> createOutputEquivRunner({ProgressCb? onProgress}) async {
  final r = OutputEquivRunner();
  await r.initFromAssets(onProgress: onProgress);
  return r;
}

/// One-shot convenience.
Future<String> runPipelineOnVideo(File videoFile, {ProgressCb? onProgress}) async {
  final r = await createOutputEquivRunner(onProgress: onProgress);
  return r.runOnVideo(videoFile, onProgress: onProgress);
}
