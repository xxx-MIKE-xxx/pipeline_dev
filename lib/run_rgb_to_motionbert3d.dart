// run_rgb_to_motionebert3d.dart
//
// Disk-backed, low-memory pipeline:
// - Extracts JPG frames to /tmp (fps=15, pre-letterboxed 640x640)
// - YOLO every N frames, holds bbox between detections
// - RTMPose (SimCC) on crops
// - COCO->H36M mapping
// - MotionBERT on a padded/truncated sequence
// Writes yolo_out.json, rtm_out.json, motionbert_out.json to Documents/run_<ts>/
//
// Requires pubspec:
//   flutter_onnxruntime, image, ffmpeg_kit_flutter_new, path_provider
//
// Change in this file:
// - All logging and progress callbacks are now wrapped in _sLog/_sProgress to
//   avoid user callback exceptions propagating into native code or isolates.
// - NEW: Auto-tune RTMPose preprocessing (rgb_255 / bgr_255 / rgb_ms / bgr_ms)
//   on the first crop; then reuse the best mode for the whole run.
// - NEW: Prefer Core ML EP on Apple by ordering providers [CORE_ML, XNNPACK, CPU].
// - NEW: `runPipelineOnVideoInIsolate` helper to run the whole pipeline
//        on a background isolate (logs via debugPrint inside the isolate).

import 'dart:async';
import 'dart:convert';
import 'dart:io' show Directory, File;
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:isolate';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/ffprobe_kit.dart';

typedef ProgressCb = void Function(double progress, String stage);
typedef LogCb = void Function(String line);

class _Cfg {
  // YOLO
  late String yoloModelAsset;
  int yoloInH = 640, yoloInW = 640;
  int personClassId = 0;
  double confTh = 0.25, iouTh = 0.45;
  String classScoreActivation = 'sigmoid';
  String yoloUnits = 'normalized'; // 'normalized' or 'pixels'
  String yoloCoords = 'letterbox'; // 'letterbox' or 'original'

  // RTM
  late String rtmModelAsset;
  int rtmInH = 256, rtmInW = 192;
  String rtmPreproc = 'rgb_255'; // default; will be auto-tuned on first crop
  List<double> rtmMean = [0, 0, 0];
  List<double> rtmStd = [255, 255, 255];
  double simccRatio = 2.0;

  // MotionBERT
  late String mbModelAsset;
  int T = 243;
  bool wrapPad = true;
  bool rootRel = false;

  // Skeleton orders (fallbacks)
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

class YDet {
  YDet(this.xyxy, this.score);
  final List<double> xyxy; // xyxy in original frame pixels
  final double score;
}

class DiskBackedPoseRunner {
  DiskBackedPoseRunner({this.yoloStride = 2});

  final _Cfg cfg = _Cfg();
  final int yoloStride;

  late final OnnxRuntime _ort;
  late OrtSession _yolo, _rtm, _mb;

  late String _yoloInName, _rtmInName, _mbInName;
  late List<String> _yoloOutNames, _rtmOutNames, _mbOutNames;

  // Reused Float32 buffers to avoid churn
  late final Float32List _yoloBuf;
  late final Float32List _rtmBuf;

  // Safe log/progress wrappers
  void _sLog(LogCb? cb, String msg) {
    try { cb?.call(msg); } catch (e, st) {
      debugPrint('onLog threw: $e\n$st');
    }
    debugPrint(msg);
  }
  void _sProgress(ProgressCb? cb, double p, String stage) {
    try { cb?.call(p, stage); } catch (e, st) {
      debugPrint('onProgress threw: $e\n$st');
    }
  }

  // ---- NEW: RTM auto-tune state ----
  String? _rtmPreprocChosen; // once decided, reused for all frames
  final List<String> _rtmTryModes = const ['rgb_255', 'bgr_255', 'rgb_ms', 'bgr_ms'];
  final double _rtmAutoThresh = 0.02; // if first-crop mean conf is below this, try other modes

  List<double>? _prevBox; // xyxy

  Future<void> initFromAssets({ProgressCb? onProgress, LogCb? onLog}) async {
    _sProgress(onProgress, 0.01, 'Loading configs');

    // Load cfgs
    final yoloCfg = jsonDecode(await rootBundle.loadString('assets/models/configs/yolo.cfg')) as Map;
    final rtmCfg  = jsonDecode(await rootBundle.loadString('assets/models/configs/rtm.cfg')) as Map;
    final mbCfg   = jsonDecode(await rootBundle.loadString('assets/models/configs/motionbert.cfg')) as Map;

    // Optional skeletons.json
    try {
      final sk = jsonDecode(await rootBundle.loadString('assets/models/configs/skeletons.json')) as Map;
      if (sk['coco17'] is List) cfg.cocoOrder = (sk['coco17'] as List).cast<String>();
      if (sk['h36m17'] is List) cfg.h36mOrder = (sk['h36m17'] as List).cast<String>();
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
    cfg.yoloUnits = (yDom['units'] ?? 'normalized') as String; // normalized|pixels
    cfg.yoloCoords= (yDom['coords'] ?? 'letterbox') as String; // letterbox|original
    final yPost = (yoloCfg['postprocess'] ?? {}) as Map;
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

    // MotionBERT
    cfg.mbModelAsset = (mbCfg['model']?['path'] ?? 'assets/models/motionbert_3d_243.onnx') as String;
    cfg.T = (mbCfg['model']?['input']?['sequence_length'] ?? 243) as int;
    cfg.rootRel = (mbCfg['model']?['output']?['root_relative'] ?? false) as bool;
    cfg.wrapPad = (mbCfg['runtime']?['wrap_pad_sequence'] ?? true) as bool;

    _sProgress(onProgress, 0.06, 'Creating ONNX sessions');
    _ort = OnnxRuntime();

    // ---- ONLY NECESSARY CHANGE #1: Prefer Core ML, then XNNPACK, then CPU.
    final providers = <OrtProvider>[
      OrtProvider.CORE_ML, // Core ML EP on Apple
      OrtProvider.XNNPACK,
      OrtProvider.CPU,
    ];
    final opts = OrtSessionOptions(providers: providers);

    _yolo = await _ort.createSessionFromAsset(cfg.yoloModelAsset, options: opts);
    _rtm  = await _ort.createSessionFromAsset(cfg.rtmModelAsset,  options: opts);
    _mb   = await _ort.createSessionFromAsset(cfg.mbModelAsset,   options: opts);

    _yoloInName   = _yolo.inputNames.first;
    _rtmInName    = _rtm.inputNames.first;
    _mbInName     = _mb.inputNames.first;
    _yoloOutNames = _yolo.outputNames.toList();
    _rtmOutNames  = _rtm.outputNames.toList();
    _mbOutNames   = _mb.outputNames.toList();

    // [LOG] Providers + model I/O names
    _sProgress(onProgress, 0.065, 'Sessions ready');
    _sLog(onLog, '[YOLO] in=${_yoloInName} outs=${_yoloOutNames.join(",")}');
    _sLog(onLog, '[RTM]  in=${_rtmInName}  outs=${_rtmOutNames.join(",")}');
    _sLog(onLog, '[MB]   in=${_mbInName}   outs=${_mbOutNames.join(",")}');
    _sLog(onLog, '[EPs]  requested=[CORE_ML, XNNPACK, CPU]'); // choice is logged here

    // Reusable buffers
    _yoloBuf = Float32List(3 * cfg.yoloInH * cfg.yoloInW);
    _rtmBuf  = Float32List(3 * cfg.rtmInH  * cfg.rtmInW);
  }

  /// Entry point (disk-backed, low memory).
  Future<String> runThreePassOnVideo(
    File videoFile, {
    ProgressCb? onProgress,
    LogCb? onLog,
  }) async {
    final docs = await getApplicationDocumentsDirectory();
    final outDir = Directory('${docs.path}/run_${_ts()}');
    await outDir.create(recursive: true);

    // Pass 0: extract frames (JPG, fps=15, pre-letterbox to 640x640)
    _sProgress(onProgress, 0.08, 'Extracting frames');
    final framesDir = await _extractFrames(
      videoFile,
      onProgress: (p, _) => _sProgress(onProgress, 0.08 + 0.04 * p, 'Extracting frames'),
      onLog: onLog,
    );

    final frames = await framesDir
        .list()
        .where((e) => e is File && e.path.toLowerCase().endsWith('.jpg'))
        .cast<File>()
        .toList();
    frames.sort((a,b)=>a.path.compareTo(b.path));
    if (frames.isEmpty) throw Exception('No frames decoded from video.');

    final first = img.decodeJpg(await frames.first.readAsBytes());
    if (first == null) throw Exception('Failed to decode first frame.');
    final inW = first.width, inH = first.height;

    final total = frames.length;
    _sProgress(onProgress, 0.12, 'Running YOLO/RTMPose');

    // Compact per-frame results; JSON written at the end.
    final yoloBboxesNorm = <List<double>>[];
    final yoloScores     = <double>[];
    final seqCocoXYC     = <List<List<double>>>[];  // [F][17][3]
    final seqH36MXYC     = <List<List<double>>>[];  // [F][17][3]
    _prevBox = null;

    for (int fi = 0; fi < total; fi++) {
      final bytes = await frames[fi].readAsBytes();
      final frame = img.decodeJpg(bytes);
      if (frame == null) continue;

      // YOLO every yoloStride frames, hold previous otherwise
      List<double> box; double score = 0.0;
      if (fi % yoloStride == 0 || _prevBox == null) {
        double ratio = 1.0; List<double> pads = [0.0,0.0,0.0,0.0];
        img.Image yoloImg = frame;
        if (frame.width != cfg.yoloInW || frame.height != cfg.yoloInH) {
          final t = _letterbox(frame, newH: cfg.yoloInH, newW: cfg.yoloInW, padColor: img.ColorRgb8(114,114,114));
          yoloImg = t.$1; ratio = t.$2; pads = t.$3;
        }

        _toCHWFloatInto(dst: _yoloBuf, image: yoloImg,
          mean: const [0,0,0], std: const [255,255,255], bgr: false);
        final inVal = await OrtValue.fromList(_yoloBuf, [1,3,cfg.yoloInH,cfg.yoloInW]);
        final res   = await _yolo.run({_yoloInName: inVal});
        await inVal.dispose();

        final out = res[_yoloOutNames.first]!;
        final shape = out.shape; // [1,N,C] or [1,C,N]
        final flat  = (await out.asFlattenedList()).cast<num>();
        _sLog(onLog, '[YOLO] raw out shape: $shape');
        for (final v in res.values) { await v.dispose(); }

        final dets = _yoloPostprocess(
          flat: flat,
          shape: shape,
          imgW: frame.width,
          imgH: frame.height,
          netW: cfg.yoloInW,
          netH: cfg.yoloInH,
          ratio: ratio,
          pads: pads,
          confTh: cfg.confTh,
          iouTh: cfg.iouTh,
          classId: cfg.personClassId,
          activation: cfg.classScoreActivation,
          outputsAreNormalized: cfg.yoloUnits.toLowerCase() == 'normalized',
          coordsSpaceIsLetterbox: cfg.yoloCoords.toLowerCase() == 'letterbox',
        );
        if (dets.isEmpty) {
          _sLog(onLog, '[YOLO] frame=$fi det=NONE (holding prev or using heuristic)');
        } else {
          final d = dets.first;
          _sLog(onLog, '[YOLO] frame=$fi score=${d.score.toStringAsFixed(3)} '
                       'xyxy=[${d.xyxy.map((v)=>v.toStringAsFixed(1)).join(", ")}]');
        }

        if (dets.isEmpty) {
          box = _prevBox ?? [inW*0.25, inH*0.1, inW*0.75, inH*0.9];
        } else {
          box = dets.first.xyxy; score = dets.first.score; _prevBox = List<double>.from(box);
        }
      } else {
        box = List<double>.from(_prevBox!);
      }

      // Save bbox normalized by original frame W/H (NOT letterbox)
      yoloBboxesNorm.add([box[0]/inW, box[1]/inH, box[2]/inW, box[3]/inH]);
      yoloScores.add(score);

      // RTMPose on crop
      final cropRes = _cropToAspect(frame, box, outH: cfg.rtmInH, outW: cfg.rtmInW, scale: 1.25);
      final crop = cropRes.$1; final rect = cropRes.$2;
      if (crop == null || rect == null) {
        if (seqCocoXYC.isNotEmpty) {
          seqCocoXYC.add(_deepCopy2D(seqCocoXYC.last));
          seqH36MXYC.add(_deepCopy2D(seqH36MXYC.last));
        } else {
          seqCocoXYC.add(_zeros173());
          seqH36MXYC.add(_zeros173());
        }
      } else {
        // ---- Auto-tune RTM preprocess (first crop only), log the choice
        List<List<double>> cocoXYC;
        if (_rtmPreprocChosen == null) {
          final firstTry = await _rtmOnceWithMode(crop, rect, cfg.rtmPreproc, cfg.simccRatio);
          double bestConf = firstTry.$2;
          String bestMode = cfg.rtmPreproc;
          List<List<double>> bestPts = firstTry.$1;

          if (bestConf < _rtmAutoThresh) {
            for (final m in _rtmTryModes) {
              if (m == cfg.rtmPreproc) continue;
              final r = await _rtmOnceWithMode(crop, rect, m, cfg.simccRatio);
              if (r.$2 > bestConf) { bestConf = r.$2; bestMode = m; bestPts = r.$1; }
            }
          }
          _rtmPreprocChosen = bestMode;
          _sLog(onLog, '[RTM] selected preprocess="$bestMode" (first-crop mean_conf=${bestConf.toStringAsFixed(4)})');
          cocoXYC = bestPts;
        } else {
          final r = await _rtmOnceWithMode(crop, rect, _rtmPreprocChosen!, cfg.simccRatio);
          cocoXYC = r.$1;
        }

        final h36mXYC = _coco17ToH36M17(cocoXYC);
        seqCocoXYC.add(cocoXYC);
        seqH36MXYC.add(h36mXYC);

        if (fi < 8) {
          final vis = img.copyResize(crop, width: (cfg.rtmInW*2), height: (cfg.rtmInH*2));
          void dot(double x, double y, int r, int g, int b) {
            final xx = ((x - rect[0]) * (cfg.rtmInW / rect[2]) * 2).round();
            final yy = ((y - rect[1]) * (cfg.rtmInH / rect[3]) * 2).round();
            img.fillCircle(vis, x: xx, y: yy, radius: 2, color: img.ColorUint8.rgb(r,g,b));
          }
          for (final p in cocoXYC) { dot(p[0], p[1], 0,255,0); }
          final tmp = await getTemporaryDirectory();
          final fpath = '${tmp.path}/crop_vis_$fi.jpg';
          await File(fpath).writeAsBytes(img.encodeJpg(vis, quality: 92));
          _sLog(onLog, '[DBG] wrote $fpath');
        }

        if ((fi % 10) == 0) {
          _sLog(onLog, '[RTM] frame=$fi mean_conf=${_meanConfFrame(cocoXYC).toStringAsFixed(4)}');
        }
      }

      final p = 0.12 + 0.78 * (fi + 1) / total;
      _sProgress(onProgress, p, 'Frames ${fi + 1}/$total');
      if ((fi & 3) == 0) { await Future<void>.delayed(const Duration(milliseconds: 1)); }
    }

    // MotionBERT input
    _sProgress(onProgress, 0.91, 'Preparing MotionBERT input');
    _sLog(onLog, '[RTM] run_mean_conf=${_meanConfSeq(seqCocoXYC).toStringAsFixed(4)} '
                 '(preproc=${_rtmPreprocChosen ?? cfg.rtmPreproc})');

    final mbSeq = _buildMbSequence(seqH36MXYC, inW, inH, T: cfg.T, wrapPad: cfg.wrapPad);

    // MotionBERT run
    _sProgress(onProgress, 0.94, 'Running MotionBERT');
    final mbFlat = <double>[];
    for (int t = 0; t < cfg.T; t++) {
      for (int j = 0; j < 17; j++) {
        mbFlat.addAll(mbSeq[t][j]); // [xn, yn, c]
      }
    }
    final mbInVal = await OrtValue.fromList(mbFlat, [1, cfg.T, 17, 3]);
    final mbRes   = await _mb.run({_mbInName: mbInVal});
    await mbInVal.dispose();
    final mbTensor = mbRes[_mbOutNames.first]!;
    final mbShape  = mbTensor.shape; // [1,T,17,3]
    final mbData   = (await mbTensor.asFlattenedList()).cast<double>();
    for (final v in mbRes.values) { await v.dispose(); }
    final coords3d = _reshape3d(mbData, mbShape[1], mbShape[2], mbShape[3]);
    if (cfg.rootRel) { for (int t = 0; t < coords3d.length; t++) { coords3d[t][0] = [0.0,0.0,0.0]; } }

    // Write JSON artifacts
    _sProgress(onProgress, 0.98, 'Writing JSONs');
    final videoBase = videoFile.uri.pathSegments.last;
    await _writeJson(outDir.path, 'yolo_out.json', {
      "video": videoBase,
      "frames": yoloBboxesNorm.length,
      "bboxes_norm": yoloBboxesNorm,
      "scores": yoloScores,
      "yolo_output_units": cfg.yoloUnits,
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

    _sProgress(onProgress, 1.0, 'Done');
    return outDir.path;
  }

  // ---------- Helpers ----------

  double _meanConfFrame(List<List<double>> f) {
    double s=0; for (final j in f) s+=j[2]; return s/ (f.isEmpty?1:f.length);
  }
  double _meanConfSeq(List<List<List<double>>> seq) {
    double s=0; int n=0; for (final f in seq) { for (final j in f) { s+=j[2]; n++; } }
    return n>0 ? s/n : 0.0;
  }

  (List<double>, List<double>, bool) _rtmPreprocTripletFor(String mode) {
    switch (mode) {
      case 'rgb_ms': return (cfg.rtmMean, cfg.rtmStd, false);
      case 'bgr_ms': return (cfg.rtmMean, cfg.rtmStd, true);
      case 'bgr_255': return ([0,0,0], [255,255,255], true);
      case 'rgb_255':
      default: return ([0,0,0], [255,255,255], false);
    }
  }

  (List<double>, List<double>, bool) _rtmPreprocTriplet() {
    return _rtmPreprocTripletFor(_rtmPreprocChosen ?? cfg.rtmPreproc);
  }

  Future<(List<List<double>>, double)> _rtmOnceWithMode(
    img.Image crop, List<double> rect, String mode, double splitRatio) async {

    final trip = _rtmPreprocTripletFor(mode);
    _toCHWFloatInto(dst: _rtmBuf, image: crop,
        mean: trip.$1, std: trip.$2, bgr: trip.$3);
    final inVal = await OrtValue.fromList(_rtmBuf, [1,3,cfg.rtmInH,cfg.rtmInW]);
    final res   = await _rtm.run({_rtmInName: inVal});
    await inVal.dispose();

    final simccX = res[_rtmOutNames[0]]!;
    final simccY = res[_rtmOutNames[1]]!;
    final xShape = simccX.shape, yShape = simccY.shape;
    final xFlat  = (await simccX.asFlattenedList()).cast<double>();
    final yFlat  = (await simccY.asFlattenedList()).cast<double>();
    for (final v in res.values) { await v.dispose(); }

    final cocoCrop = _simccDecodeFast(xFlat, xShape, yFlat, yShape, splitRatio: splitRatio);
    final rx = rect[0], ry = rect[1], rw = rect[2], rh = rect[3];
    final cocoXYC = List.generate(17, (i) {
      final x = rx + cocoCrop[i][0] * (rw / cfg.rtmInW);
      final y = ry + cocoCrop[i][1] * (rh / cfg.rtmInH);
      return [x, y, cocoCrop[i][2]];
    });
    final meanC = _meanConfFrame(cocoXYC);
    return (cocoXYC, meanC);
  }

  Future<Directory> _extractFrames(
    File video, {
    ProgressCb? onProgress,
    LogCb? onLog,
  }) async {
    final tmp = await getTemporaryDirectory();
    final dir = Directory('${tmp.path}/frames_${_ts()}');
    await dir.create(recursive: true);

    final durMs = await _probeDurationMs(video.path);

    final vf = [
      'fps=15',
      'scale=640:640:force_original_aspect_ratio=decrease',
      'pad=640:640:(ow-iw)/2:(oh-ih)/2:color=0x727272',
      'format=rgb24',
    ].join(',');

    final cmd = [
      '-hide_banner',
      '-y',
      '-i', _q(video.path),
      '-vf', _q(vf),
      '-q:v', '3',
      '${_q(dir.path)}/frame_%05d.jpg'
    ].join(' ');

    await _runFfmpegAndWait(cmd,
      onLog: onLog, onProgress: onProgress, totalMs: durMs);
    return dir;
  }

  Future<void> _runFfmpegAndWait(
    String command, {
    LogCb? onLog,
    ProgressCb? onProgress,
    int? totalMs,
  }) async {
    final completer = Completer<void>();
    await FFmpegKit.executeAsync(
      command,
      (session) async {
        final rc = await session.getReturnCode();
        if (rc != null && rc.isValueSuccess()) return completer.complete();
        if (rc != null && rc.isValueCancel())  return completer.completeError(StateError('FFmpeg cancelled'));
        final output = await session.getOutput();
        completer.completeError(StateError('FFmpeg failed: ${output ?? "unknown error"}'));
      },
      (log) {
        final m = log.getMessage();
        if (m != null) _sLog(onLog, m);
      },
      (stats) {
        if (totalMs != null && totalMs > 0) {
          final t = stats.getTime();
          if (t != null) _sProgress(onProgress, (t / totalMs).clamp(0.0, 1.0), 'FFmpeg');
        }
      },
    );
    await completer.future;
  }

  Future<int?> _probeDurationMs(String videoPath) async {
    try {
      final session = await FFprobeKit.getMediaInformation(videoPath);
      final info = await session.getMediaInformation();
      final s = info?.getDuration();
      if (s == null) return null;
      final d = double.tryParse(s);
      if (d == null) return null;
      return (d * 1000).round();
    } catch (_) {
      return null;
    }
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

  void _toCHWFloatInto({
    required Float32List dst,
    required img.Image image,
    required List<double> mean,
    required List<double> std,
    required bool bgr,
  }) {
    final h = image.height, w = image.width;
    int dx = 0, plane = h * w;
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
        dst[dx]           = r;
        dst[dx + plane]   = g;
        dst[dx + 2*plane] = b;
        dx++;
      }
    }
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
      for (int i=0;i<cls.length;i++){ final v = cls[i]; cls[i] = 1.0 / (1.0 + math.exp(-v)); }
    }

    final dets = <YDet>[];

    final padL = pads[0], padT = pads[1];

    for (int i=0;i<n;i++){
      double bestScore = -1e9; int bestK = -1;
      for (int k=0;k<(c-4);k++){
        final v = cls[i*(c-4)+k];
        if (v > bestScore){ bestScore = v; bestK = k; }
      }
      if (bestK != classId) continue;
      if (bestScore < confTh) continue;

      double cx = boxes[i*4+0];
      double cy = boxes[i*4+1];
      double bw = boxes[i*4+2];
      double bh = boxes[i*4+3];

      if (outputsAreNormalized) {
        cx *= netW; cy *= netH; bw *= netW; bh *= netH;
      }

      double x1lb = cx - bw/2, y1lb = cy - bh/2, x2lb = cx + bw/2, y2lb = cy + bh/2;

      double x1, y1, x2, y2;
      if (coordsSpaceIsLetterbox) {
        x1 = _clampD((x1lb - padL) / ratio, 0.0, imgW - 1.0);
        y1 = _clampD((y1lb - padT) / ratio, 0.0, imgH - 1.0);
        x2 = _clampD((x2lb - padL) / ratio, 0.0, imgW - 1.0);
        y2 = _clampD((y2lb - padT) / ratio, 0.0, imgH - 1.0);
      } else {
        x1 = _clampD(x1lb, 0.0, imgW - 1.0);
        y1 = _clampD(y1lb, 0.0, imgH - 1.0);
        x2 = _clampD(x2lb, 0.0, imgW - 1.0);
        y2 = _clampD(y2lb, 0.0, imgH - 1.0);
      }

      dets.add(YDet([x1,y1,x2,y2], bestScore));
    }

    dets.sort((a,b)=>b.score.compareTo(a.score));
    final keep = <YDet>[];
    for (final d in dets){
      bool sup = false;
      for (final k in keep){ if (_iou(d.xyxy, k.xyxy) > iouTh){ sup = true; break; } }
      if (!sup) keep.add(d);
    }
    return keep;
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

  List<List<double>> _coco17ToH36M17(List<List<double>> c) {
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

  List<List<List<double>>> _buildMbSequence(
    List<List<List<double>>> seq, int inW, int inH, {required int T, required bool wrapPad}) {
    final out = <List<List<double>>>[];
    final s = math.min(inW,inH)/2.0, cx=inW/2.0, cy=inH/2.0;
    for (final frame in seq) {
      out.add(List.generate(17, (j){
        final x = frame[j][0], y = frame[j][1], c = frame[j][2];
        final xn = (x - cx) / s;
        final yn = (y - cy) / s;
        return [xn, yn, c];
      }));
    }
    if (wrapPad) {
      if (out.length < T) {
        final last = out.isNotEmpty ? _deepCopy2D(out.last) : _zeros173();
        while (out.length < T) out.add(_deepCopy2D(last));
      } else if (out.length > T) {
        while (out.length > T) out.removeLast();
      }
    } else if (out.length != T) {
      throw Exception('MotionBERT expects T=$T, got ${out.length}. Enable wrapPad to auto-pad/truncate.');
    }
    return out;
  }

  List<List<List<double>>> _reshape3d(List<double> flat, int T, int J, int C){
    final out = List.generate(T, (_) => List.generate(J, (_)=>List.filled(C,0.0)));
    int idx=0;
    for (int t=0;t<T;t++){
      for (int j=0;j<J;j++){
        for (int c=0;c<C;c++){ out[t][j][c] = flat[idx++]; }
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
    final ix1 = math.max(ax1, bx1), iy1 = math.max(ay1, by1);
    final ix2 = math.min(ax2, bx2), iy2 = math.min(ay2, by2);
    final iw = math.max(0, ix2 - ix1), ih = math.max(0, iy2 - iy1);
    final inter = iw * ih;
    final aArea = math.max(0, ax2 - ax1) * math.max(0, ay2 - ay1);
    final bArea = math.max(0, bx2 - bx1) * math.max(0, by2 - by1);
    return inter / (aArea + bArea - inter + 1e-6);
  }

  List<List<double>> _deepCopy2D(List<List<double>> src) =>
      List<List<double>>.generate(src.length, (i) => List<double>.from(src[i]), growable: false);

  List<List<double>> _zeros173() =>
      List<List<double>>.generate(17, (_) => [0.0, 0.0, 0.0], growable: false);

  String _q(String s) => '"${s.replaceAll('"', '\\"')}"';
}

// ------------- Compatibility helpers -------------

/// Keep your old call-sites working.
/// This simply constructs the disk-backed runner and runs the 3-pass pipeline.
Future<String> runPipelineOnVideo(
  File videoFile, {
  ProgressCb? onProgress,
  LogCb? onLog,
  int yoloStride = 2,
}) async {
  final r = DiskBackedPoseRunner(yoloStride: yoloStride);
  await r.initFromAssets(onProgress: onProgress, onLog: onLog);
  return r.runThreePassOnVideo(videoFile, onProgress: onProgress, onLog: onLog);
}

/// NEW: Run the whole pipeline on a background isolate.
/// NOTE: Callbacks (onProgress/onLog) can’t be sent across isolates; this uses
/// debugPrint inside the isolate to log EP choices & auto-tuned preprocessing.
/// If you need UI progress, keep using `runPipelineOnVideo` on the main isolate.
Future<String> runPipelineOnVideoInIsolate(
  File videoFile, {
  int yoloStride = 2,
}) async {
  // `Isolate.run` executes the closure on a worker isolate and returns the result.
  // See: https://dart.dev/language/isolates
  return await Isolate.run(() async {
    final r = DiskBackedPoseRunner(yoloStride: yoloStride);
    await r.initFromAssets(
      // These won't cross isolates; logs will go via debugPrint inside.
      onProgress: null,
      onLog: null,
    );
    return r.runThreePassOnVideo(videoFile, onProgress: null, onLog: null);
  });
}

/// If you want to keep the object around (e.g., run multiple videos without
/// recreating sessions), use this.
Future<DiskBackedPoseRunner> createDiskBackedRunner({
  ProgressCb? onProgress,
  LogCb? onLog,
  int yoloStride = 2,
}) async {
  final r = DiskBackedPoseRunner(yoloStride: yoloStride);
  await r.initFromAssets(onProgress: onProgress, onLog: onLog);
  return r;
}
