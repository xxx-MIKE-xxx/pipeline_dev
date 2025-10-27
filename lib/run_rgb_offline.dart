// lib/run_rgb_offline.dart
//
// Offline wrapper that reuses Core2DEngine. Extracts frames as letterboxed
// 640x640 RGB JPGs via FFmpeg, feeds them to Core2DEngine, collects results,
// then runs MotionBERT (XNNPACK→CPU) and writes the final JSON artifacts.
//
// Files written into Documents/run_<ts>/:
// - rtm2d.jsonl (one JSON per processed frame)
// - motionbert_out.json
// - meta.json
//
// Depends on: core2d.dart, flutter_onnxruntime, ffmpeg_kit_flutter_new, ffprobe_kit,
// path_provider, image.

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/ffprobe_kit.dart';

import 'core2d.dart';

class OfflineRunner {
  OfflineRunner({this.sampleEvery = 3, this.yoloStride = 2});

  final int sampleEvery; // process every Nth decoded frame
  final int yoloStride;

  late final Core2DEngine _core;
  late String _rtmPreproc;

  Future<String> run(File video, {void Function(double p, String s)? onProgress, void Function(String)? onLog}) async {
    final docs = await getApplicationDocumentsDirectory();
    final outDir = Directory('${docs.path}/run_${_ts()}');
    await outDir.create(recursive: true);

    onProgress?.call(0.02, 'Probing');
    final (origW, origH, durMs) = await _probe(video.path);

    // Extract letterboxed 640x640 RGB JPGs (base 15 fps)
    onProgress?.call(0.06, 'Extracting frames');
    final framesDir = await _extractFrames(video, baseFps: 15);
    final frames = await framesDir
        .list()
        .where((e) => e is File && e.path.toLowerCase().endsWith('.jpg'))
        .cast<File>()
        .toList()
      ..sort((a,b)=>a.path.compareTo(b.path));
    if (frames.isEmpty) { throw StateError('No frames extracted.'); }

    // Initialize Core2D
    onProgress?.call(0.12, 'Initializing 2D');
    _core = Core2DEngine(yoloStride: yoloStride);
    await _core.init();

    // Ratio/pads used to map between spaces
    final r = math.min(640 / origW, 640 / origH);
    final nw = (origW * r).round(), nh = (origH * r).round();
    final padL = ((640 - nw) / 2).floor().toDouble();
    final padT = ((640 - nh) / 2).floor().toDouble();
    final pads = [padL, padT, (640 - nw - padL).toDouble(), (640 - nh - padT).toDouble()];

    // JSONL sink
    final jsonl = File('${outDir.path}/rtm2d.jsonl').openWrite();

    int kept = 0, dropped = 0;
    for (int i = 0; i < frames.length; i++) {
      if (i % sampleEvery != 0) { dropped++; continue; }
      final bytes = await frames[i].readAsBytes();
      final im = img.decodeJpg(bytes);
      if (im == null) continue;

      // Convert image → raw RGB8 bytes
      final w = im.width;
      final h = im.height;
      final rgb = Uint8List(w * h * 3);

      int di = 0;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final p = im.getPixel(x, y);
          // p.r/g/b are `num` in image v4 → cast to int (and keep in 0..255)
          final r = p.r.clamp(0, 255).toInt();
          final g = p.g.clamp(0, 255).toInt();
          final b = p.b.clamp(0, 255).toInt();

          rgb[di++] = r;
          rgb[di++] = g;
          rgb[di++] = b;
        }
      }

      final pkt = FramePacket(
        rgb640x640: rgb,
        origW: origW,
        origH: origH,
        ratio: r,
        pads: pads,
        frameIdx: kept,
        tsNs: ((durMs * (i/frames.length)) * 1e6).round(), // rough ts
      );

      final r2d = await _core.process(pkt);
      _rtmPreproc = _core.rtmPreprocChosen;

      final line = jsonEncode({
        "ts_ns": pkt.tsNs,
        "frame_idx": pkt.frameIdx,
        "orig_size": [origW, origH],
        "yolo": {"xyxy_norm": r2d.bboxNorm, "score": r2d.score},
        "coco17_xyc": r2d.cocoXYC,
      });
      jsonl.writeln(line);

      kept++;
      final p = 0.12 + 0.75 * (i + 1) / frames.length;
      onProgress?.call(p, '2D ${kept}/${frames.length~/sampleEvery}');
    }

    await jsonl.flush(); await jsonl.close();
    await _core.dispose();

    // MotionBERT
    onProgress?.call(0.92, 'Running MotionBERT');
    final out3d = await _runMotionBertFromJsonl(outDir);

    // Meta
    final meta = {
      "mode": "offline",
      "orig_size": [origW, origH],
      "sample_every": sampleEvery,
      "yolo_stride": yoloStride,
      "rtm_preproc": _rtmPreproc,
      "counts": {"kept": kept, "dropped": dropped},
    };
    await File('${outDir.path}/meta.json').writeAsString(const JsonEncoder.withIndent('  ').convert(meta));

    onProgress?.call(1.0, 'Done');
    return outDir.path;
  }

  // ---- helpers ----

  Future<(int w, int h, int ms)> _probe(String path) async {
    final session = await FFprobeKit.getMediaInformation(path);
    final info = await session.getMediaInformation();
    if (info == null) throw StateError('ffprobe failed');

    final streams = info.getStreams();
    final s0 = streams.isNotEmpty ? streams.first : null;

    final w = (s0?.getWidth() as int?) ?? 1920;
    final h = (s0?.getHeight() as int?) ?? 1080;

    final d = double.tryParse(info.getDuration() ?? '0') ?? 0.0;
    return (w, h, (d * 1000).round());
  }


  Future<Directory> _extractFrames(File video, {required int baseFps}) async {
    final tmp = await getTemporaryDirectory();
    final dir = Directory('${tmp.path}/frames_${_ts()}');
    await dir.create(recursive: true);

    final vf = [
      'fps=$baseFps',
      'scale=640:640:force_original_aspect_ratio=decrease',
      'pad=640:640:(ow-iw)/2:(oh-ih)/2:color=0x727272',
      'format=rgb24',
    ].join(',');

    final cmd = [
      '-hide_banner', '-y',
      '-i', _q(video.path),
      '-vf', _q(vf),
      '-q:v', '3',
      '${_q(dir.path)}/frame_%05d.jpg'
    ].join(' ');

    final c = Completer<void>();
    await FFmpegKit.executeAsync(cmd, (session) async {
      final rc = await session.getReturnCode();
      if (rc != null && rc.isValueSuccess()) return c.complete();
      final out = await session.getOutput();
      c.completeError(StateError('FFmpeg failed: ${out ?? ""}'));
    });
    await c.future;
    return dir;
  }

  Future<String> _runMotionBertFromJsonl(Directory runDir) async {
    final cfg = jsonDecode(await rootBundle.loadString('assets/models/configs/motionbert.cfg')) as Map;
    final mbPath = (cfg['model']?['path'] ?? 'assets/models/motionbert_3d_243.onnx') as String;
    final T = (cfg['model']?['input']?['sequence_length'] ?? 243) as int;
    final rootRel = (cfg['model']?['output']?['root_relative'] ?? false) as bool;
    final wrapPad = (cfg['runtime']?['wrap_pad_sequence'] ?? true) as bool;

    // Read jsonl frames
    final lines = await File('${runDir.path}/rtm2d.jsonl').readAsLines();
    final frames = lines.map((l) => jsonDecode(l) as Map).toList();

    if (frames.isEmpty) {
      await File('${runDir.path}/motionbert_out.json').writeAsString('{"frames":0,"coords_3d":[]}');
      return runDir.path;
    }

    final origW = (frames.first['orig_size'] as List)[0] as int;
    final origH = (frames.first['orig_size'] as List)[1] as int;

    // Build H36M normalized sequence
    final seqH36M = <List<List<double>>>[];
    final s = math.min(origW, origH) / 2.0, cx = origW / 2.0, cy = origH / 2.0;

    for (final f in frames) {
      final coco = (f['coco17_xyc'] as List).map((p) => (p as List).map((e)=>(e as num).toDouble()).toList()).toList();
      final h36m = Core2DEngine.coco17ToH36M17(coco);
      final frame = List.generate(17, (j) {
        final x = h36m[j][0], y = h36m[j][1], c = h36m[j][2];
        return [(x - cx) / s, (y - cy) / s, c];
      });
      seqH36M.add(frame);
    }

    // Pad/truncate
    if (wrapPad) {
      final last = seqH36M.isNotEmpty ? seqH36M.last : List.generate(17, (_)=>[0.0,0.0,0.0]);
      while (seqH36M.length < T) seqH36M.add(List.generate(17, (j)=>[last[j][0], last[j][1], last[j][2]]));
      while (seqH36M.length > T) seqH36M.removeLast();
    } else if (seqH36M.length != T) {
      throw StateError('MotionBERT expects T=$T, got ${seqH36M.length}');
    }

    // Flatten and run MB
    final mbFlat = <double>[];
    for (int t=0; t<T; t++) { for (int j=0;j<17;j++) { mbFlat.addAll(seqH36M[t][j]); } }

    final ort = OnnxRuntime();
    final opts = OrtSessionOptions(providers: [OrtProvider.XNNPACK, OrtProvider.CPU]);
    final sess = await ort.createSessionFromAsset(mbPath, options: opts);

    final inName = sess.inputNames.first;
    final outName = sess.outputNames.first;
    final inVal = await OrtValue.fromList(mbFlat, [1, T, 17, 3]);
    final res   = await sess.run({inName: inVal});
    await inVal.dispose();
    final out   = res[outName]!;
    final shape = out.shape; // [1,T,17,3]
    final data  = (await out.asFlattenedList()).cast<double>();
    for (final v in res.values) { await v.dispose(); }
    try { await (sess as dynamic).dispose(); } catch (_){}

    // Reshape and post
    final coords3d = List.generate(shape[1], (_) => List.generate(shape[2], (_)=>List.filled(shape[3], 0.0)));
    int idx = 0;
    for (int t=0;t<shape[1];t++){
      for (int j=0;j<shape[2];j++){
        for (int c=0;c<shape[3];c++){ coords3d[t][j][c] = data[idx++]; }
      }
    }
    if (rootRel) { for (int t=0;t<coords3d.length;t++) { coords3d[t][0] = [0.0,0.0,0.0]; } }

    await File('${runDir.path}/motionbert_out.json').writeAsString(const JsonEncoder.withIndent('  ').convert({
      "T": T,
      "coords_3d": coords3d,
    }));
    return runDir.path;
  }

  String _ts() {
    final n = DateTime.now();
    String two(int v)=>v.toString().padLeft(2,'0');
    return '${n.year}${two(n.month)}${two(n.day)}_${two(n.hour)}${two(n.minute)}${two(n.second)}';
  }

  String _q(String s) => '"${s.replaceAll('"', '\\"')}"';
}

// Convenience function
Future<String> runPipelineOnVideoOffline(File video,
    {int sampleEvery = 3, int yoloStride = 2,
     void Function(double,String)? onProgress, void Function(String)? onLog}) async {
  final r = OfflineRunner(sampleEvery: sampleEvery, yoloStride: yoloStride);
  return r.run(video, onProgress: onProgress, onLog: onLog);
}
