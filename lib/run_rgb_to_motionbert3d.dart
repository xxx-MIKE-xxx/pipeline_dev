import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ffmpeg_kit_flutter_min_gpl/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_min_gpl/return_code.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class PosePipelineException implements Exception {
  const PosePipelineException(this.message);
  final String message;

  @override
  String toString() => 'PosePipelineException: $message';
}

class PoseRunResult {
  const PoseRunResult({
    required this.runDirectory,
    required this.frameCount,
    required this.outputs,
  });

  final Directory runDirectory;
  final int frameCount;
  final Map<String, File> outputs;
}

class PosePipeline {
  PosePipeline._({
    required this.yoloCfg,
    required this.rtmCfg,
    required this.motionCfg,
    required this.yoloSession,
    required this.rtmSession,
    required this.motionSession,
    required this.yoloInputName,
    required this.rtmInputName,
    required this.motionInputName,
  });

  final Map<String, dynamic> yoloCfg;
  final Map<String, dynamic> rtmCfg;
  final Map<String, dynamic> motionCfg;
  final OrtSession yoloSession;
  final OrtSession rtmSession;
  final OrtSession motionSession;
  final String yoloInputName;
  final String rtmInputName;
  final String motionInputName;

  static Future<PosePipeline> create() async {
    const cfgDir = 'assets/models/configs';
    final yoloCfg = jsonDecode(await rootBundle.loadString('$cfgDir/yolo.cfg')) as Map<String, dynamic>;
    final rtmCfg = jsonDecode(await rootBundle.loadString('$cfgDir/rtm.cfg')) as Map<String, dynamic>;
    final motionCfg = jsonDecode(await rootBundle.loadString('$cfgDir/motionbert.cfg')) as Map<String, dynamic>;

    final cacheDir = await _ensureModelCache();
    final yoloModel = await _materializeModelAsset(cfgDir, yoloCfg, cacheDir);
    final rtmModel = await _materializeModelAsset(cfgDir, rtmCfg, cacheDir);
    final motionModel = await _materializeModelAsset(cfgDir, motionCfg, cacheDir);

    final yoloSession = await OrtSession.fromFile(yoloModel.path);
    final rtmSession = await OrtSession.fromFile(rtmModel.path);
    final motionSession = await OrtSession.fromFile(motionModel.path);

    return PosePipeline._(
      yoloCfg: yoloCfg,
      rtmCfg: rtmCfg,
      motionCfg: motionCfg,
      yoloSession: yoloSession,
      rtmSession: rtmSession,
      motionSession: motionSession,
      yoloInputName: yoloSession.inputNames.first,
      rtmInputName: rtmSession.inputNames.first,
      motionInputName: motionSession.inputNames.first,
    );
  }

  Future<PoseRunResult> processVideo(File videoFile) async {
    if (!await videoFile.exists()) {
      throw PosePipelineException('Video file not found: ${videoFile.path}');
    }

    final runDir = await _prepareRunDirectory();
    final framesDir = Directory(p.join(runDir.path, 'frames'));
    await framesDir.create(recursive: true);

    await _extractFrames(videoFile, framesDir);

    final frameFiles = await framesDir
        .list()
        .whereType<File>()
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    if (frameFiles.isEmpty) {
      throw const PosePipelineException('No frames extracted from video.');
    }

    final firstImage = img.decodeImage(await frameFiles.first.readAsBytes());
    if (firstImage == null) {
      throw PosePipelineException('Unable to decode first frame: ${frameFiles.first.path}');
    }
    final frameWidth = firstImage.width.toDouble();
    final frameHeight = firstImage.height.toDouble();

    final yoloInput = yoloCfg['model']['input'] as Map<String, dynamic>;
    final yoloOutput = yoloCfg['model']['output'] as Map<String, dynamic>;
    final yoloPost = yoloCfg['postprocess'] as Map<String, dynamic>;

    final lbHeight = (yoloInput['height'] ?? yoloInput['letterbox']['target_height']) as int;
    final lbWidth = (yoloInput['width'] ?? yoloInput['letterbox']['target_width']) as int;
    final padColor = (yoloInput['letterbox']['pad_color'] as List<dynamic>).cast<int>();
    final confThreshold = (yoloPost['conf_threshold'] as num).toDouble();
    final iouThreshold = (yoloPost['iou_threshold'] as num).toDouble();
    final maxDetections = (yoloPost['max_detections'] ?? 300) as int;
    final classActivation = yoloOutput['class_score_activation'] as String? ?? 'sigmoid';
    final personClassId = (yoloOutput['classes']['person_id'] as num).toInt();
    final yoloUnits = (yoloOutput['domain']['units'] as String?) ?? 'normalized';
    final yoloCoords = (yoloOutput['domain']['coords'] as String?) ?? 'letterbox';

    final rtmOutput = rtmCfg['model']['output'] as Map<String, dynamic>;
    final rtmInput = rtmCfg['model']['input'] as Map<String, dynamic>;
    final rtmPreproc = (rtmInput['preprocess'] as Map<String, dynamic>?) ?? const {};
    final splitRatio = ((rtmOutput['simcc'] as Map<String, dynamic>)['split_ratio'] as num).toDouble();
    final keypoints = (rtmOutput['keypoints'] as num).toInt();

    final mbInputCfg = motionCfg['model']['input'] as Map<String, dynamic>;
    final sequenceLength = (mbInputCfg['sequence_length'] as num).toInt();
    final wrapPad = (motionCfg['runtime'] as Map<String, dynamic>?)?['wrap_pad_sequence'] as bool? ?? false;

    final detectionLog = <Map<String, dynamic>>[];
    final yoloBboxesNorm = <List<double>>[];
    final yoloScores = <double>[];
    final seqCoco = <List<List<double>>>[];
    final seqH36m = <List<List<double>>>[];

    List<double>? previousBBox;

    for (var index = 0; index < frameFiles.length; index++) {
      final file = frameFiles[index];
      final image = index == 0 ? firstImage : img.decodeImage(await file.readAsBytes());
      if (image == null) {
        throw PosePipelineException('Unable to decode frame: ${file.path}');
      }

      final letterbox = _letterbox(image, targetH: lbHeight, targetW: lbWidth, padColor: padColor);
      final yoloInputTensor = _imageToFloat32Nchw(letterbox.image, scale: 1 / 255.0);
      final rawDetections = await _runYolo(yoloInputTensor, [1, 3, lbHeight, lbWidth]);
      final detection = _decodeYolo(
        rawDetections,
        letterboxWidth: lbWidth,
        letterboxHeight: lbHeight,
        scale: letterbox.scale,
        padX: letterbox.padX,
        padY: letterbox.padY,
        origWidth: image.width.toDouble(),
        origHeight: image.height.toDouble(),
        confThreshold: confThreshold,
        iouThreshold: iouThreshold,
        maxDetections: maxDetections,
        classActivation: classActivation,
        personClassId: personClassId,
        outputUnits: yoloUnits,
        outputCoords: yoloCoords,
      );

      List<double> bbox;
      double score;
      if (detection != null) {
        bbox = detection.bbox;
        score = detection.score;
        previousBBox = bbox;
      } else if (previousBBox != null) {
        bbox = List<double>.from(previousBBox);
        score = 0.0;
      } else {
        bbox = [
          image.width * 0.25,
          image.height * 0.1,
          image.width * 0.75,
          image.height * 0.9,
        ];
        score = 0.0;
        previousBBox = bbox;
      }

      final bboxNorm = [
        bbox[0] / image.width,
        bbox[1] / image.height,
        bbox[2] / image.width,
        bbox[3] / image.height,
      ];
      detectionLog.add({
        't': index,
        'bbox': bboxNorm,
        'score': score,
        'yolo_output_units': yoloUnits,
        'yolo_output_coords': yoloCoords,
      });
      yoloBboxesNorm.add(bboxNorm);
      yoloScores.add(score);

      final crop = _cropToAspect(image, bbox, outH: 256, outW: 192, scale: 1.25);
      if (crop == null) {
        if (seqCoco.isNotEmpty) {
          seqCoco.add(seqCoco.last.map((joint) => List<double>.from(joint)).toList());
          seqH36m.add(seqH36m.last.map((joint) => List<double>.from(joint)).toList());
        } else {
          final zeroFrame = List<List<double>>.generate(17, (_) => [0.0, 0.0, 0.0]);
          seqCoco.add(zeroFrame);
          seqH36m.add(zeroFrame);
        }
        continue;
      }

      final cropTensor = _prepareRtmInput(
        crop.image,
        mode: rtmPreproc['mode'] as String? ?? 'rgb_255',
        mean: (rtmPreproc['mean'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList(),
        std: (rtmPreproc['std'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList(),
      );

      final simcc = await _runRtm(cropTensor, const [1, 3, 256, 192]);
      final coords = _decodeSimcc(
        simccX: simcc.simccX,
        simccY: simcc.simccY,
        splitRatio: splitRatio,
        keypoints: keypoints,
      );

      final cocoFrame = _coordsToImage(coords, crop.rect, cropWidth: 192.0, cropHeight: 256.0);
      final h36mFrame = _coco17ToH36m17(cocoFrame);
      seqCoco.add(cocoFrame);
      seqH36m.add(h36mFrame);
    }

    final mbInput = _prepareMotionInput(
      seqH36m,
      frameWidth: frameWidth,
      frameHeight: frameHeight,
      sequenceLength: sequenceLength,
      wrapPad: wrapPad,
    );
    final mbOutput = await _runMotionBert(mbInput, sequenceLength: sequenceLength);

    final outputs = await _writeOutputs(
      runDir: runDir,
      videoFile: videoFile,
      detections: detectionLog,
      yoloBboxesNorm: yoloBboxesNorm,
      yoloScores: yoloScores,
      yoloUnits: yoloUnits,
      yoloCoords: yoloCoords,
      cocoFrames: seqCoco,
      h36mFrames: seqH36m,
      mbInput: mbInput,
      mbOutput: mbOutput,
    );

    return PoseRunResult(
      runDirectory: runDir,
      frameCount: seqCoco.length,
      outputs: outputs,
    );
  }

  Future<Float32List> _runYolo(Float32List input, List<int> shape) async {
    final tensor = OrtValueTensor.createTensorWithDataList(input, shape);
    final dynamic outputs = await yoloSession.run(ortInputs: {yoloInputName: tensor});
    final dynamic first = _firstOrtValue(outputs);
    final value = first?.value;
    if (value is Float32List) {
      return Float32List.fromList(value.toList());
    }
    if (value is List) {
      return Float32List.fromList(value.cast<double>());
    }
    throw const PosePipelineException('Unexpected YOLO output type');
  }

  Future<_RtmResult> _runRtm(Float32List input, List<int> shape) async {
    final tensor = OrtValueTensor.createTensorWithDataList(input, shape);
    final dynamic outputs = await rtmSession.run(ortInputs: {rtmInputName: tensor});
    if (outputs is List && outputs.length >= 2) {
      final first = outputs[0];
      final second = outputs[1];
      final firstValue = (first as dynamic).value;
      final secondValue = (second as dynamic).value;
      final simccX = firstValue is Float32List
          ? Float32List.fromList(firstValue.toList())
          : Float32List.fromList((firstValue as List).cast<double>());
      final simccY = secondValue is Float32List
          ? Float32List.fromList(secondValue.toList())
          : Float32List.fromList((secondValue as List).cast<double>());
      return _RtmResult(simccX: simccX, simccY: simccY);
    }
    if (outputs is Map && outputs.length >= 2) {
      final values = outputs.values.toList();
      final firstValue = (values[0] as dynamic).value;
      final secondValue = (values[1] as dynamic).value;
      final simccX = firstValue is Float32List
          ? Float32List.fromList(firstValue.toList())
          : Float32List.fromList((firstValue as List).cast<double>());
      final simccY = secondValue is Float32List
          ? Float32List.fromList(secondValue.toList())
          : Float32List.fromList((secondValue as List).cast<double>());
      return _RtmResult(simccX: simccX, simccY: simccY);
    }
    throw const PosePipelineException('Unexpected RTM output type');
  }

  Future<Float32List> _runMotionBert(Float32List input, {required int sequenceLength}) async {
    final tensor = OrtValueTensor.createTensorWithDataList(input, [1, sequenceLength, 17, 3]);
    final dynamic outputs = await motionSession.run(ortInputs: {motionInputName: tensor});
    final dynamic first = _firstOrtValue(outputs);
    final value = first?.value;
    if (value is Float32List) {
      return Float32List.fromList(value.toList());
    }
    if (value is List) {
      return Float32List.fromList(value.cast<double>());
    }
    throw const PosePipelineException('Unexpected MotionBERT output type');
  }
}

dynamic _firstOrtValue(dynamic outputs) {
  if (outputs is List && outputs.isNotEmpty) {
    return outputs.first;
  }
  if (outputs is Map && outputs.isNotEmpty) {
    return outputs.values.first;
  }
  return null;
}

class _RtmResult {
  const _RtmResult({required this.simccX, required this.simccY});
  final Float32List simccX;
  final Float32List simccY;
}

class _LetterboxResult {
  const _LetterboxResult({
    required this.image,
    required this.scale,
    required this.padX,
    required this.padY,
  });

  final img.Image image;
  final double scale;
  final double padX;
  final double padY;
}

class _CropResult {
  const _CropResult({required this.image, required this.rect});
  final img.Image image;
  final _RectD rect;
}

class _RectD {
  const _RectD(this.x, this.y, this.width, this.height);

  final double x;
  final double y;
  final double width;
  final double height;
}

class _YoloDetection {
  const _YoloDetection({required this.bbox, required this.score});
  final List<double> bbox;
  final double score;
}

Future<Directory> _ensureModelCache() async {
  final docs = await getApplicationDocumentsDirectory();
  final cacheDir = Directory(p.join(docs.path, '_onnx_cache'));
  await cacheDir.create(recursive: true);
  return cacheDir;
}

Future<File> _materializeModelAsset(String cfgDir, Map<String, dynamic> cfg, Directory cacheDir) async {
  final model = cfg['model'] as Map<String, dynamic>;
  final relativePath = model['path'] as String;
  final assetPath = p.normalize(p.join(cfgDir, relativePath));
  final resolvedAsset = assetPath.startsWith('assets/') ? assetPath : 'assets/$assetPath';
  final target = File(p.join(cacheDir.path, p.basename(resolvedAsset)));
  if (!await target.exists()) {
    final data = await rootBundle.load(resolvedAsset);
    await target.writeAsBytes(data.buffer.asUint8List(), flush: true);
  }
  return target;
}

Future<Directory> _prepareRunDirectory() async {
  final docs = await getApplicationDocumentsDirectory();
  final timestamp = DateTime.now().millisecondsSinceEpoch;
  final runDir = Directory(p.join(docs.path, 'run_$timestamp'));
  await runDir.create(recursive: true);
  await Directory(p.join(runDir.path, 'debug')).create(recursive: true);
  return runDir;
}

Future<void> _extractFrames(File videoFile, Directory framesDir) async {
  final pattern = p.join(framesDir.path, 'frame_%06d.png');
  final command = '-i "${videoFile.path}" -vf fps=30 "$pattern"';
  final session = await FFmpegKit.execute(command);
  final code = await session.getReturnCode();
  if (!ReturnCode.isSuccess(code)) {
    final value = code?.getValue();
    throw PosePipelineException('FFmpeg failed with code ${value ?? 'unknown'}');
  }
}

_LetterboxResult _letterbox(
  img.Image image, {
  required int targetH,
  required int targetW,
  required List<int> padColor,
}) {
  final scale = math.min(targetH / image.height, targetW / image.width);
  final resized = img.copyResize(
    image,
    width: (image.width * scale).round(),
    height: (image.height * scale).round(),
    interpolation: img.Interpolation.linear,
  );
  final canvas = img.Image(width: targetW, height: targetH);
  final color = padColor.length >= 3 ? padColor : [114, 114, 114];
  for (var y = 0; y < targetH; y++) {
    for (var x = 0; x < targetW; x++) {
      canvas.setPixelRgba(x, y, color[0], color[1], color[2]);
    }
  }
  final dx = ((targetW - resized.width) / 2).floor();
  final dy = ((targetH - resized.height) / 2).floor();
  for (var y = 0; y < resized.height; y++) {
    for (var x = 0; x < resized.width; x++) {
      canvas.setPixel(dx + x, dy + y, resized.getPixel(x, y));
    }
  }
  return _LetterboxResult(
    image: canvas,
    scale: scale,
    padX: dx.toDouble(),
    padY: dy.toDouble(),
  );
}

Float32List _imageToFloat32Nchw(img.Image image, {required double scale}) {
  final tensor = Float32List(3 * image.height * image.width);
  var offset = 0;
  for (var c = 0; c < 3; c++) {
    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        final channelValue = c == 0
            ? _getRed(pixel)
            : c == 1
                ? _getGreen(pixel)
                : _getBlue(pixel);
        tensor[offset++] = channelValue * scale;
      }
    }
  }
  return tensor;
}

int _getRed(int color) => color & 0xFF;

int _getGreen(int color) => (color >> 8) & 0xFF;

int _getBlue(int color) => (color >> 16) & 0xFF;

_YoloDetection? _decodeYolo(
  Float32List raw, {
  required int letterboxWidth,
  required int letterboxHeight,
  required double scale,
  required double padX,
  required double padY,
  required double origWidth,
  required double origHeight,
  required double confThreshold,
  required double iouThreshold,
  required int maxDetections,
  required String classActivation,
  required int personClassId,
  required String outputUnits,
  required String outputCoords,
}) {
  const stride = 84;
  if (raw.isEmpty || raw.length % stride != 0) {
    return null;
  }

  final boxes = <List<double>>[];
  final scores = <double>[];
  final detections = raw.length ~/ stride;

  for (var i = 0; i < detections; i++) {
    final base = i * stride;
    final cx = raw[base];
    final cy = raw[base + 1];
    final w = raw[base + 2];
    final h = raw[base + 3];

    double bestScore = 0;
    var bestClass = -1;
    for (var c = 0; c < 80; c++) {
      final logit = raw[base + 4 + c];
      final prob = classActivation == 'sigmoid' ? 1.0 / (1.0 + math.exp(-logit)) : logit;
      if (prob > bestScore) {
        bestScore = prob;
        bestClass = c;
      }
    }

    if (bestClass != personClassId || bestScore < confThreshold) {
      continue;
    }

    final x1 = cx - w / 2;
    final y1 = cy - h / 2;
    final x2 = cx + w / 2;
    final y2 = cy + h / 2;
    boxes.add([x1, y1, x2, y2]);
    scores.add(bestScore);
  }

  if (boxes.isEmpty) {
    return null;
  }

  final keep = _nms(boxes, scores, iouThreshold, maxDetections: maxDetections);
  if (keep.isEmpty) {
    return null;
  }

  final box = List<double>.from(boxes[keep.first]);
  if (outputUnits == 'normalized') {
    box[0] *= letterboxWidth;
    box[2] *= letterboxWidth;
    box[1] *= letterboxHeight;
    box[3] *= letterboxHeight;
  }

  if (outputCoords != 'original') {
    box[0] = (box[0] - padX) / scale;
    box[2] = (box[2] - padX) / scale;
    box[1] = (box[1] - padY) / scale;
    box[3] = (box[3] - padY) / scale;
  }

  box[0] = box[0].clamp(0, origWidth - 1);
  box[1] = box[1].clamp(0, origHeight - 1);
  box[2] = box[2].clamp(0, origWidth - 1);
  box[3] = box[3].clamp(0, origHeight - 1);

  return _YoloDetection(bbox: box, score: scores[keep.first]);
}

List<int> _nms(List<List<double>> boxes, List<double> scores, double iouThreshold, {required int maxDetections}) {
  final indices = List<int>.generate(boxes.length, (i) => i)
    ..sort((a, b) => scores[b].compareTo(scores[a]));
  final keep = <int>[];

  while (indices.isNotEmpty && keep.length < maxDetections) {
    final current = indices.removeAt(0);
    keep.add(current);
    indices.removeWhere((idx) => _iou(boxes[current], boxes[idx]) > iouThreshold);
  }
  return keep;
}

double _iou(List<double> a, List<double> b) {
  final x1 = math.max(a[0], b[0]);
  final y1 = math.max(a[1], b[1]);
  final x2 = math.min(a[2], b[2]);
  final y2 = math.min(a[3], b[3]);
  final interW = math.max(0, x2 - x1);
  final interH = math.max(0, y2 - y1);
  final inter = interW * interH;
  final areaA = math.max(0, a[2] - a[0]) * math.max(0, a[3] - a[1]);
  final areaB = math.max(0, b[2] - b[0]) * math.max(0, b[3] - b[1]);
  final union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

_CropResult? _cropToAspect(img.Image image, List<double> bbox, {required int outH, required int outW, required double scale}) {
  final x1 = bbox[0];
  final y1 = bbox[1];
  final x2 = bbox[2];
  final y2 = bbox[3];

  final cx = (x1 + x2) / 2;
  final cy = (y1 + y2) / 2;
  var bw = (x2 - x1) * scale;
  var bh = (y2 - y1) * scale;

  final targetAr = outW / outH;
  if (bw / bh > targetAr) {
    bh = bw / targetAr;
  } else {
    bw = bh * targetAr;
  }

  final x1n = math.max(0, (cx - bw / 2).round());
  final y1n = math.max(0, (cy - bh / 2).round());
  final x2n = math.min(image.width - 1, (cx + bw / 2).round());
  final y2n = math.min(image.height - 1, (cy + bh / 2).round());
  final width = x2n - x1n;
  final height = y2n - y1n;
  if (width <= 0 || height <= 0) {
    return null;
  }

  final crop = img.copyCrop(image, x: x1n, y: y1n, width: width, height: height);
  final resized = img.copyResize(crop, width: outW, height: outH, interpolation: img.Interpolation.linear);
  return _CropResult(
    image: resized,
    rect: _RectD(x1n.toDouble(), y1n.toDouble(), width.toDouble(), height.toDouble()),
  );
}

Float32List _prepareRtmInput(img.Image crop, {required String mode, List<double>? mean, List<double>? std}) {
  final tensor = Float32List(3 * crop.height * crop.width);
  final normalizedMode = mode.toLowerCase();
  final meanVals = mean ?? const [123.675, 116.28, 103.53];
  final stdVals = std ?? const [58.395, 57.12, 57.375];
  var offset = 0;

  for (var c = 0; c < 3; c++) {
    for (var y = 0; y < crop.height; y++) {
      for (var x = 0; x < crop.width; x++) {
        final pixel = crop.getPixel(x, y);
        final r = _getRed(pixel).toDouble();
        final g = _getGreen(pixel).toDouble();
        final b = _getBlue(pixel).toDouble();
        double value;
        if (normalizedMode.startsWith('rgb')) {
          value = c == 0 ? r : c == 1 ? g : b;
        } else {
          value = c == 0 ? b : c == 1 ? g : r;
        }
        if (normalizedMode.endsWith('_255')) {
          value /= 255.0;
        } else {
          value = (value - meanVals[c]) / stdVals[c];
        }
        tensor[offset++] = value;
      }
    }
  }
  return tensor;
}

List<List<double>> _decodeSimcc({
  required Float32List simccX,
  required Float32List simccY,
  required double splitRatio,
  required int keypoints,
}) {
  final xLabel = simccX.length ~/ keypoints;
  final yLabel = simccY.length ~/ keypoints;
  final coords = <List<double>>[];

  for (var k = 0; k < keypoints; k++) {
    final px = _softmaxSegment(simccX, k * xLabel, xLabel);
    final py = _softmaxSegment(simccY, k * yLabel, yLabel);
    final xIdx = _argmax(px);
    final yIdx = _argmax(py);
    final confidence = math.sqrt(px[xIdx] * py[yIdx]);
    coords.add([xIdx / splitRatio, yIdx / splitRatio, confidence]);
  }
  return coords;
}

List<double> _softmaxSegment(Float32List data, int start, int length) {
  var maxVal = -double.infinity;
  for (var i = 0; i < length; i++) {
    final v = data[start + i];
    if (v > maxVal) {
      maxVal = v;
    }
  }
  final exps = List<double>.generate(length, (i) => math.exp(data[start + i] - maxVal));
  final sum = exps.fold<double>(0, (acc, v) => acc + v);
  return exps.map((v) => v / sum).toList(growable: false);
}

int _argmax(List<double> values) {
  var bestIndex = 0;
  var bestValue = values[0];
  for (var i = 1; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

List<List<double>> _coordsToImage(List<List<double>> coords, _RectD rect, {required double cropWidth, required double cropHeight}) {
  final scaleX = rect.width / cropWidth;
  final scaleY = rect.height / cropHeight;
  return coords
      .map((c) => [rect.x + c[0] * scaleX, rect.y + c[1] * scaleY, c[2]])
      .toList(growable: false);
}

List<List<double>> _coco17ToH36m17(List<List<double>> coco) {
  List<double> avg(List<double> a, List<double> b) => [
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2,
        (a[2] + b[2]) / 2,
      ];

  final pelvis = avg(coco[11], coco[12]);
  final neck = avg(coco[5], coco[6]);
  final spine1 = avg(pelvis, neck);
  final eyesMid = (coco[1][2] > 0 && coco[2][2] > 0) ? avg(coco[1], coco[2]) : coco[0];
  final head = eyesMid;
  final site = coco[0];

  List<double> clone(int idx) => List<double>.from(coco[idx]);

  final h36m = <List<double>>[
    [pelvis[0], pelvis[1], (coco[11][2] + coco[12][2]) / 2],
    clone(12),
    clone(14),
    clone(16),
    clone(11),
    clone(13),
    clone(15),
    [spine1[0], spine1[1], (pelvis[2] + neck[2]) / 2],
    [neck[0], neck[1], (coco[5][2] + coco[6][2]) / 2],
    [head[0], head[1], (coco[1][2] > 0 && coco[2][2] > 0) ? (coco[1][2] + coco[2][2]) / 2 : coco[0][2]],
    clone(0),
    clone(5),
    clone(7),
    clone(9),
    clone(6),
    clone(8),
    clone(10),
  ];
  return h36m;
}

Float32List _prepareMotionInput(
  List<List<List<double>>> seqH36m, {
  required double frameWidth,
  required double frameHeight,
  required int sequenceLength,
  required bool wrapPad,
}) {
  final sequence = seqH36m.map((frame) => frame.map((joint) => List<double>.from(joint)).toList()).toList();
  if (wrapPad) {
    if (sequence.length < sequenceLength) {
      final last = sequence.isNotEmpty
          ? sequence.last
          : List<List<double>>.generate(17, (_) => [0.0, 0.0, 0.0]);
      while (sequence.length < sequenceLength) {
        sequence.add(last.map((joint) => List<double>.from(joint)).toList());
      }
    } else if (sequence.length > sequenceLength) {
      sequence.removeRange(sequenceLength, sequence.length);
    }
  } else if (sequence.length != sequenceLength) {
    throw PosePipelineException('MotionBERT expects $sequenceLength frames, got ${sequence.length}');
  }

  final Float32List tensor = Float32List(sequenceLength * 17 * 3);
  final s = math.min(frameWidth, frameHeight) / 2.0;
  final cx = frameWidth / 2.0;
  final cy = frameHeight / 2.0;
  var offset = 0;

  for (var t = 0; t < sequenceLength; t++) {
    final frame = t < sequence.length ? sequence[t] : sequence.last;
    for (final joint in frame) {
      final xNorm = s == 0 ? 0.0 : (joint[0] - cx) / s;
      final yNorm = s == 0 ? 0.0 : (joint[1] - cy) / s;
      tensor[offset++] = xNorm;
      tensor[offset++] = yNorm;
      tensor[offset++] = joint[2];
    }
  }
  return tensor;
}

Future<Map<String, File>> _writeOutputs({
  required Directory runDir,
  required File videoFile,
  required List<Map<String, dynamic>> detections,
  required List<List<double>> yoloBboxesNorm,
  required List<double> yoloScores,
  required String yoloUnits,
  required String yoloCoords,
  required List<List<List<double>>> cocoFrames,
  required List<List<List<double>>> h36mFrames,
  required Float32List mbInput,
  required Float32List mbOutput,
}) async {
  final outputs = <String, File>{};
  final debugDir = Directory(p.join(runDir.path, 'debug'));
  await debugDir.create(recursive: true);

  final detectionsFile = File(p.join(debugDir.path, 'detections.jsonl'));
  final sink = detectionsFile.openWrite();
  for (final det in detections) {
    sink.writeln(jsonEncode(det));
  }
  await sink.close();
  outputs['detections.jsonl'] = detectionsFile;

  final cocoArray = _flattenFrameArray(cocoFrames);
  final h36mArray = _flattenFrameArray(h36mFrames);
  final mbInFrames = mbInput.length ~/ (17 * 3);
  final mbOutFrames = mbOutput.length ~/ (17 * 3);

  await _writeNpy(debugDir, 'coco_2d.npy', cocoArray, [cocoFrames.length, 17, 3], outputs);
  await _writeNpy(debugDir, 'h36m_2d.npy', h36mArray, [h36mFrames.length, 17, 3], outputs);
  await _writeNpy(debugDir, 'mb_input_seq.npy', mbInput, [mbInFrames, 17, 3], outputs);
  await _writeNpy(debugDir, 'mb_output_3d.npy', mbOutput, [mbOutFrames, 17, 3], outputs);

  final statsFile = File(p.join(debugDir.path, 'quick_stats.json'));
  final stats = {
    'frames': cocoFrames.length,
    'coco_conf_mean': cocoFrames.isEmpty
        ? 0.0
        : cocoFrames
                .expand((frame) => frame.map((joint) => joint[2]))
                .fold<double>(0, (acc, v) => acc + v) /
            (cocoFrames.length * 17),
    'coco_x_ptp': _ptp(cocoFrames, axis: 0),
    'coco_y_ptp': _ptp(cocoFrames, axis: 1),
    'seen_person': yoloScores.any((score) => score > 0),
  };
  await statsFile.writeAsString(jsonEncode(stats), flush: true);
  outputs['quick_stats.json'] = statsFile;

  final mbFrames = _float32ToFrames(mbOutput, mbOutFrames);
  final result = {
    'video': p.basename(videoFile.path),
    'T': mbOutFrames,
    'h36m_order': const [
      'Pelvis',
      'RHip',
      'RKnee',
      'RAnkle',
      'LHip',
      'LKnee',
      'LAnkle',
      'Spine1',
      'Neck',
      'Head',
      'Site',
      'LShoulder',
      'LElbow',
      'LWrist',
      'RShoulder',
      'RElbow',
      'RWrist',
    ],
    'coords_3d': mbFrames,
  };

  final out3dFile = File(p.join(runDir.path, 'out_3d.json'));
  await out3dFile.writeAsString(jsonEncode(result));
  outputs['out_3d.json'] = out3dFile;

  final rtmPayload = {
    'video': p.basename(videoFile.path),
    'frames': cocoFrames.length,
    'coco_order': const [
      'Nose',
      'LEye',
      'REye',
      'LEar',
      'REar',
      'LShoulder',
      'RShoulder',
      'LElbow',
      'RElbow',
      'LWrist',
      'RWrist',
      'LHip',
      'RHip',
      'LKnee',
      'RKnee',
      'LAnkle',
      'RAnkle',
    ],
    'coords_2d': cocoFrames,
  };
  final rtmFile = File(p.join(runDir.path, 'rtm_out.json'));
  await rtmFile.writeAsString(jsonEncode(rtmPayload));
  outputs['rtm_out.json'] = rtmFile;

  final yoloPayload = {
    'video': p.basename(videoFile.path),
    'frames': yoloBboxesNorm.length,
    'bboxes_norm': yoloBboxesNorm,
    'scores': yoloScores,
    'yolo_output_units': yoloUnits,
    'yolo_output_coords': yoloCoords,
    'normalized_by': const ['width', 'height'],
  };
  final yoloFile = File(p.join(runDir.path, 'yolo_out.json'));
  await yoloFile.writeAsString(jsonEncode(yoloPayload));
  outputs['yolo_out.json'] = yoloFile;

  final motionbertFile = File(p.join(runDir.path, 'motionbert_out.json'));
  await motionbertFile.writeAsString(jsonEncode(result));
  outputs['motionbert_out.json'] = motionbertFile;

  return outputs;
}

Future<void> _writeNpy(Directory dir, String name, Float32List data, List<int> shape, Map<String, File> outputs) async {
  final file = File(p.join(dir.path, name));
  final bytes = _createNpy(data, shape);
  await file.writeAsBytes(bytes, flush: true);
  outputs[name] = file;
}

Float32List _flattenFrameArray(List<List<List<double>>> frames) {
  final flat = Float32List(frames.length * 17 * 3);
  var offset = 0;
  for (final frame in frames) {
    for (final joint in frame) {
      flat[offset++] = joint[0];
      flat[offset++] = joint[1];
      flat[offset++] = joint[2];
    }
  }
  return flat;
}

List<List<List<double>>> _float32ToFrames(Float32List data, int frameCount) {
  final frames = <List<List<double>>>[];
  var offset = 0;
  for (var t = 0; t < frameCount; t++) {
    final joints = <List<double>>[];
    for (var j = 0; j < 17; j++) {
      joints.add([
        data[offset++],
        data[offset++],
        data[offset++],
      ]);
    }
    frames.add(joints);
  }
  return frames;
}

double _ptp(List<List<List<double>>> frames, {required int axis}) {
  if (frames.isEmpty) {
    return 0.0;
  }
  double minVal = double.infinity;
  double maxVal = -double.infinity;
  for (final frame in frames) {
    for (final joint in frame) {
      final value = joint[axis];
      if (value < minVal) {
        minVal = value;
      }
      if (value > maxVal) {
        maxVal = value;
      }
    }
  }
  return maxVal - minVal;
}

Uint8List _createNpy(Float32List data, List<int> shape) {
  final magic = <int>[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  final version = <int>[1, 0];
  final shapeStr = shape.length == 1 ? '(${shape.first},)' : '(${shape.join(', ')})';
  final headerStr = "{'descr': '<f4', 'fortran_order': False, 'shape': $shapeStr, }";
  final headerBytes = utf8.encode(headerStr);
  final headerPadding = 16 - ((magic.length + version.length + 2 + headerBytes.length) % 16);
  final paddedHeader = headerStr + ' ' * (headerPadding - 1) + '\n';
  final headerLen = paddedHeader.length;
  final headerLenBytes = Uint8List(2)..buffer.asByteData().setUint16(0, headerLen, Endian.little);
  final builder = BytesBuilder()
    ..add(magic)
    ..add(version)
    ..add(headerLenBytes)
    ..add(utf8.encode(paddedHeader))
    ..add(data.buffer.asUint8List());
  return builder.toBytes();
}
