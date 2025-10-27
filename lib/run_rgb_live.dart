// lib/run_rgb_live.dart
//
// Live (during recording) pipeline wrapper.
// - Start capture via MethodChannel('live_pose').invokeMethod('startCapture', {savePath,target2dFps})
// - Frames arrive on EventChannel('live_pose/frames'): { idx, tsNs, w=640, h=640, origW, origH, ratio, pads, rgb }
// - A dedicated isolate (Live2DWorker) owns Core2DEngine and processes frames at ~target2dFps
//   (token bucket). Backpressure: ring buffer with drop-oldest.
// - Per processed frame we append JSON to rtm2d.jsonl
// - On stop: stopCapture(), drain isolate, dispose 2D, run MotionBERT, write meta.json
//
// NOTE: Requires the iOS plugin in ios/Runner/LivePosePlugin.swift.

import 'dart:async';
import 'dart:collection';
import 'dart:convert';
import 'dart:isolate';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show debugPrint;
import 'package:flutter/services.dart'
    show
        MethodChannel,
        EventChannel,
        ServicesBinding,
        RootIsolateToken,
        BackgroundIsolateBinaryMessenger,
        rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

import 'core2d.dart';

class Live2DSession {
  Live2DSession({this.bufferDepth = 5});

  // Native channels
  final MethodChannel _method = const MethodChannel('live_pose');
  final EventChannel _events = const EventChannel('live_pose/frames');

  // Backpressure
  final int bufferDepth; // when queue >= depth, drop oldest
  final _queue = ListQueue<Map<String, dynamic>>();

  // Frame subscription
  StreamSubscription? _frameSub;

  // Token bucket to target 2D fps
  late final _TokenBucket _bucket;

  // Isolate running Core2DEngine (YOLO + RTM)
  Isolate? _iso;
  SendPort? _toIso;
  final _fromIso = ReceivePort();

  // Output / state
  Directory? _runDir;
  IOSink? _jsonSink;
  IOSink? _liveLogSink;
  int _arrived = 0, _processed = 0, _dropped = 0;
  String _rtmPreproc = 'rgb_255';
  int target2dFps = 10;
  int yoloStride = 2;

  // Lightweight on-device live logs (also written to Documents/run_.../live_log.txt)
  final _logCtrl = StreamController<String>.broadcast();
  Stream<String> get logStream => _logCtrl.stream;
  void _log(String msg) {
    final line = '[${DateTime.now().toIso8601String()}] $msg';
    debugPrint(line);
    _liveLogSink?.writeln(line);
    _logCtrl.add(line);
  }

  // --------------------------------------------------------------------------
  // Public API
  // --------------------------------------------------------------------------

  Future<void> start({
    Directory? runDir,
    int target2dFps = 10,
    int yoloStride = 2,
  }) async {
    this.target2dFps = target2dFps;
    this.yoloStride = yoloStride;

    _bucket = _TokenBucket(ratePerSec: target2dFps, capacity: target2dFps);

    _runDir = runDir ?? await _createRunDir();

    // Open rtm2d.jsonl immediately (so it surely exists on STOP)
    final jsonlFile = File('${_runDir!.path}/rtm2d.jsonl');
    await jsonlFile.parent.create(recursive: true);
    _jsonSink = jsonlFile.openWrite(mode: FileMode.writeOnlyAppend);

    // Open live log file too
    _liveLogSink = File('${_runDir!.path}/live_log.txt').openWrite(mode: FileMode.writeOnlyAppend);
    _log('Opened ${jsonlFile.path} and live_log.txt');

    // Spawn 2D worker
    await _spawnWorker(this.yoloStride);

    // Listen results/logs from isolate
    _fromIso.listen((msg) {
      final m = msg as Map;
      final typ = m['type'];
      if (typ == 'ready') {
        _log('Live2D worker ready');
      } else if (typ == 'result') {
        _jsonSink?.writeln(jsonEncode(m['data']));
        _processed++;
      } else if (typ == 'meta') {
        _rtmPreproc = (m['rtm_preproc'] as String?) ?? _rtmPreproc;
      } else if (typ == 'log') {
        final line = (m['line'] as String?) ?? '';
        if (line.isNotEmpty) _log(line);
      } else if (typ == 'error') {
        _log('Worker error: ${m['message']}');
      }
    });

    // Subscribe to frames from native
    _frameSub = _events.receiveBroadcastStream().listen((event) {
      _arrived++;
      final m = Map<String, dynamic>.from(event as Map);
      // Backpressure: drop oldest
      if (_queue.length >= bufferDepth) {
        _queue.removeFirst();
        _dropped++;
      }
      _queue.addLast(m);
      _maybeDispatch();
    });

    // Start native capture (IMPORTANT: pass a FILE path, not a directory)
    await _method.invokeMethod<void>('startCapture', {
      'savePath': '${_runDir!.path}/capture.mp4',
      'target2dFps': target2dFps,
    });

    _log('Live started: runDir=${_runDir!.path}  target2dFps=$target2dFps  yoloStride=$yoloStride');
  }

  Future<String> stopAndFinalize() async {
    _log('Stopping live capture…');

    // Stop native first (this ends the frame stream)
    Map? resp;
    try {
      resp = await _method.invokeMethod<Map>('stopCapture');
    } catch (e) {
      _log('stopCapture threw: $e');
    }

    // Cancel frame subscription
    await _frameSub?.cancel();
    _frameSub = null;

    // Drain queue quickly (prefer newest)
    while (_queue.isNotEmpty && _toIso != null) {
      final m = _queue.removeLast();
      _sendFrameToWorker(m);
    }

    // Ask worker to shutdown
    if (_toIso != null) {
      _toIso!.send({'type': 'shutdown'});
    }
    await Future<void>.delayed(const Duration(milliseconds: 60));
    _iso?.kill(priority: Isolate.immediate);
    _iso = null;

    // Close sinks before we read files
    await _jsonSink?.flush();
    await _jsonSink?.close();
    _jsonSink = null;

    await _liveLogSink?.flush();
    await _liveLogSink?.close();
    _liveLogSink = null;

    // Verify paths & counts returned by native
    final runDir = (resp?['runDir'] as String?) ?? _runDir!.path;
    final counts = Map<String, Object?>.from(resp?['counts'] as Map? ?? {});
    _log('iOS counts: $counts');
    _log('runDir: $runDir');

    // Probe jsonl presence
    final jsonlPath = '$runDir/rtm2d.jsonl';
    final jsonlFile = File(jsonlPath);
    final exists = await jsonlFile.exists();
    final size = exists ? await jsonlFile.length() : 0;
    _log('rtm2d.jsonl exists=$exists size=$size');

    // List files to help debugging if something is off
    try {
      for (final e in Directory(runDir).listSync()) {
        _log('• ${e.path}');
      }
    } catch (_) {}

    // If jsonl somehow missing, create an empty one (so MotionBERT won’t crash)
    if (!exists) {
      await jsonlFile.writeAsString('');
      _log('Created empty rtm2d.jsonl as fallback');
    }

    // Run MotionBERT over saved 2D
    await _runMotionBertFromJsonl(Directory(runDir));

    // Write meta
    final meta = {
      "mode": "live",
      "target2dFps": target2dFps,
      "yolo_stride": yoloStride,
      "rtm_preproc": _rtmPreproc,
      "counts": {"arrived": _arrived, "processed": _processed, "dropped": _dropped},
    };
    await File('$runDir/meta.json').writeAsString(const JsonEncoder.withIndent('  ').convert(meta));

    _log('Live stop complete. Output: $runDir');
    return runDir;
  }

  // --------------------------------------------------------------------------
  // Internals
  // --------------------------------------------------------------------------

  Future<Directory> _createRunDir() async {
    final docs = await getApplicationDocumentsDirectory();
    String two(int v) => v.toString().padLeft(2, '0');
    final n = DateTime.now();
    final d = Directory(
        '${docs.path}/run_${n.year}${two(n.month)}${two(n.day)}_${two(n.hour)}${two(n.minute)}${two(n.second)}');
    await d.create(recursive: true);
    return d;
  }

  void _maybeDispatch() {
    if (_queue.isEmpty || _toIso == null) return;
    if (_bucket.tryTake()) {
      final m = _queue.removeLast(); // prefer newest
      _sendFrameToWorker(m);
    } else {
      // Try again shortly
      Future.delayed(const Duration(milliseconds: 15), _maybeDispatch);
    }
  }

  void _sendFrameToWorker(Map<String, dynamic> m) {
    final rgb = (m['rgb'] as Uint8List);
    final ttd = TransferableTypedData.fromList([rgb]);
    final pkt = {
      'type': 'frame',
      'idx': m['idx'] as int,
      'tsNs': m['tsNs'] as int,
      'origW': m['origW'] as int,
      'origH': m['origH'] as int,
      'ratio': (m['ratio'] as num).toDouble(),
      'pads': (m['pads'] as List).map((e) => (e as num).toDouble()).toList(),
      'rgb': ttd,
    };
    _toIso!.send(pkt);
  }

  Future<void> _spawnWorker(int yoloStride) async {
    final initPort = ReceivePort();
    final RootIsolateToken? token = ServicesBinding.instance.rootIsolateToken;

    _iso = await Isolate.spawn<_IsoConfig>(
      _isoMain,
      _IsoConfig(
        mainSendPort: _fromIso.sendPort,
        initReplyPort: initPort.sendPort,
        yoloStride: yoloStride,
        rootToken: token,
      ),
    );
    _toIso = (await initPort.first) as SendPort;
  }

  Future<void> _runMotionBertFromJsonl(Directory runDir) async {
    // Read MB config
    final cfg =
        jsonDecode(await rootBundle.loadString('assets/models/configs/motionbert.cfg')) as Map;
    final mbPath =
        (cfg['model']?['path'] ?? 'assets/models/motionbert_3d_243.onnx') as String;
    final T = (cfg['model']?['input']?['sequence_length'] ?? 243) as int;
    final rootRel = (cfg['model']?['output']?['root_relative'] ?? false) as bool;
    final wrapPad = (cfg['runtime']?['wrap_pad_sequence'] ?? true) as bool;

    final jsonlPath = '${runDir.path}/rtm2d.jsonl';
    final lines = await File(jsonlPath).readAsLines();
    if (lines.isEmpty) {
      await File('${runDir.path}/motionbert_out.json')
          .writeAsString('{"frames":0,"coords_3d":[]}');
      _log('No 2D frames; wrote empty MotionBERT output.');
      return;
    }

    final frames = lines.map((l) => jsonDecode(l) as Map).toList();
    final origW = (frames.first['orig_size'] as List)[0] as int;
    final origH = (frames.first['orig_size'] as List)[1] as int;

    // Build H36M normalized sequence
    final seqH36M = <List<List<double>>>[];
    final s = (origW < origH ? origW : origH) / 2.0;
    final cx = origW / 2.0;
    final cy = origH / 2.0;

    for (final f in frames) {
      final coco = (f['coco17_xyc'] as List)
          .map((p) => (p as List).map((e) => (e as num).toDouble()).toList())
          .toList();
      final h36m = Core2DEngine.coco17ToH36M17(coco);
      seqH36M.add(List.generate(
          17, (j) => [(h36m[j][0] - cx) / s, (h36m[j][1] - cy) / s, h36m[j][2]]));
    }

    if (wrapPad) {
      final last =
          seqH36M.isNotEmpty ? seqH36M.last : List.generate(17, (_) => [0.0, 0.0, 0.0]);
      while (seqH36M.length < T) {
        seqH36M.add(List.generate(17, (j) => [last[j][0], last[j][1], last[j][2]]));
      }
      while (seqH36M.length > T) {
        seqH36M.removeLast();
      }
    } else if (seqH36M.length != T) {
      throw StateError('MotionBERT expects T=$T, got ${seqH36M.length}');
    }

    // Flatten and run ONNX
    final flat = <double>[];
    for (int t = 0; t < T; t++) {
      for (int j = 0; j < 17; j++) {
        flat.addAll(seqH36M[t][j]); // [xn, yn, c]
      }
    }

    final ort = OnnxRuntime();
    final sess = await ort.createSessionFromAsset(
      mbPath,
      options: OrtSessionOptions(providers: [OrtProvider.XNNPACK, OrtProvider.CPU]),
    );
    final inName = sess.inputNames.first;
    final outName = sess.outputNames.first;

    final inVal = await OrtValue.fromList(flat, [1, T, 17, 3]);
    final res = await sess.run({inName: inVal});
    await inVal.dispose();

    final out = res[outName]!;
    final shape = out.shape; // [1,T,17,3]
    final data = (await out.asFlattenedList()).cast<double>();
    for (final v in res.values) {
      await v.dispose();
    }
    try {
      await (sess as dynamic).dispose();
    } catch (_) {}

    // Reshape
    final coords3d =
        List.generate(shape[1], (_) => List.generate(shape[2], (_) => List.filled(shape[3], 0.0)));
    int idx = 0;
    for (int t = 0; t < shape[1]; t++) {
      for (int j = 0; j < shape[2]; j++) {
        for (int c = 0; c < shape[3]; c++) {
          coords3d[t][j][c] = data[idx++];
        }
      }
    }
    if (rootRel) {
      for (int t = 0; t < coords3d.length; t++) {
        coords3d[t][0] = [0.0, 0.0, 0.0];
      }
    }

    await File('${runDir.path}/motionbert_out.json').writeAsString(
      const JsonEncoder.withIndent('  ').convert({
        "T": T,
        "coords_3d": coords3d,
      }),
    );
  }
}

// ---------------------------------------------------------------------------
// Isolate code
// ---------------------------------------------------------------------------

class _IsoConfig {
  final SendPort mainSendPort;
  final SendPort initReplyPort;
  final int yoloStride;
  final RootIsolateToken? rootToken;
  _IsoConfig({
    required this.mainSendPort,
    required this.initReplyPort,
    required this.yoloStride,
    required this.rootToken,
  });
}

void _isoMain(_IsoConfig cfg) async {
  final recv = ReceivePort();
  cfg.initReplyPort.send(recv.sendPort);

  void glog(String s) => cfg.mainSendPort.send({'type': 'log', 'line': s});

  // IMPORTANT: allow assets/plugin calls in this background isolate
  try {
    if (cfg.rootToken != null) {
      BackgroundIsolateBinaryMessenger.ensureInitialized(cfg.rootToken!);
    } else {
      glog('⚠️ rootIsolateToken is null; assets may fail to load');
    }
  } catch (e) {
    cfg.mainSendPort.send({'type': 'error', 'message': 'ensureInitialized failed: $e'});
    return;
  }

  final engine = Core2DEngine(yoloStride: cfg.yoloStride);
  try {
    await engine.init();
    cfg.mainSendPort.send({'type': 'ready'});
    glog('Core2DEngine initialized (yoloStride=${cfg.yoloStride})');
  } catch (e, st) {
    cfg.mainSendPort
        .send({'type': 'error', 'message': 'Core2DEngine.init failed: $e\n$st'});
    return;
  }

  await for (final msg in recv) {
    final m = msg as Map;
    final typ = m['type'];
    if (typ == 'frame') {
      final ttd = m['rgb'] as TransferableTypedData;
      final rgb = ttd.materialize().asUint8List();

      final pkt = FramePacket(
        rgb640x640: rgb,
        origW: m['origW'] as int,
        origH: m['origH'] as int,
        ratio: (m['ratio'] as num).toDouble(),
        pads: (m['pads'] as List).map((e) => (e as num).toDouble()).toList(),
        frameIdx: m['idx'] as int,
        tsNs: m['tsNs'] as int,
      );

      try {
        final r = await engine.process(pkt);

        cfg.mainSendPort.send({
          'type': 'result',
          'data': {
            "ts_ns": pkt.tsNs,
            "frame_idx": pkt.frameIdx,
            "orig_size": [pkt.origW, pkt.origH],
            "yolo": {"xyxy_norm": r.bboxNorm, "score": r.score},
            "coco17_xyc": r.cocoXYC,
          },
        });

        if ((pkt.frameIdx % 10) == 0) {
          cfg.mainSendPort.send({'type': 'meta', 'rtm_preproc': engine.rtmPreprocChosen});
          glog(
              'F${pkt.frameIdx}: YOLO score=${r.score.toStringAsFixed(3)} bboxNorm=[${r.bboxNorm.map((v)=>v.toStringAsFixed(3)).join(', ')}] meanConf=${r.meanConf.toStringAsFixed(3)}');
        }
      } catch (e) {
        cfg.mainSendPort.send({'type': 'log', 'line': 'process() failed: $e'});
      }
    } else if (typ == 'shutdown') {
      await engine.dispose();
      glog('Worker shutdown');
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Simple token bucket
// ---------------------------------------------------------------------------

class _TokenBucket {
  _TokenBucket({required this.ratePerSec, required this.capacity})
      : _tokens = capacity.toDouble(),
        _lastRefill = DateTime.now();

  final int ratePerSec;
  final int capacity;

  double _tokens;
  DateTime _lastRefill;

  bool tryTake() {
    _refill();
    if (_tokens >= 1.0) {
      _tokens -= 1.0;
      return true;
    }
    return false;
  }

  void _refill() {
    final now = DateTime.now();
    final dt = now.difference(_lastRefill).inMicroseconds / 1e6; // seconds
    if (dt > 0) {
      _tokens = (_tokens + dt * ratePerSec).clamp(0.0, capacity.toDouble());
      _lastRefill = now;
    }
  }
}
