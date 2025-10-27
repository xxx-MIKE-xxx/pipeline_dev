// lib/main.dart
//
// Minimal UI to run either:
//  - Offline pipeline (pick a video, process, write artifacts)
//  - Live pipeline (start/stop recording, process during capture, write artifacts)
//
// This app only shows status text; no preview is required (your native capture
// is headless for inference and writes the .mp4 to runDir).

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

import 'run_rgb_offline.dart';
import 'run_rgb_live.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const PoseApp());
}

class PoseApp extends StatelessWidget {
  const PoseApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Live/Offline Pose',
      theme: ThemeData(useMaterial3: true),
      home: const PoseHome(),
    );
  }
}

class PoseHome extends StatefulWidget {
  const PoseHome({super.key});
  @override
  State<PoseHome> createState() => _PoseHomeState();
}

class _PoseHomeState extends State<PoseHome> {
  String _log = '';
  String? _lastRunDir;
  bool _busy = false;

  // Live session
  Live2DSession? _live;
  bool _liveRunning = false;

  // Live log subscription (to see logs on-device even if USB debug drops)
  StreamSubscription<String>? _liveLogSub;
  final StringBuffer _liveLogBuffer = StringBuffer();
  Timer? _liveFlushTimer;

  void _append(String s) {
    setState(() {
      _log = _log.isEmpty ? s : '$_log\n$s';
    });
  }

  // ---------------- OFFLINE ----------------

  Future<void> _runOffline() async {
    if (_busy) return;
    final picker = ImagePicker();
    final x = await picker.pickVideo(source: ImageSource.gallery);
    if (x == null) return;

    setState(() {
      _busy = true;
      _log = 'Offline pipeline started…';
    });

    try {
      final outDir = await runPipelineOnVideoOffline(
        File(x.path),
        sampleEvery: 3, // process 1/3 frames → ~5 fps from 15 fps extraction
        yoloStride: 2, // YOLO every 2 processed frames (~2.5 Hz)
        onProgress: (p, s) => _append('${(p * 100).toStringAsFixed(1)}%  $s'),
        onLog: (l) => _append(l),
      );
      setState(() {
        _lastRunDir = outDir;
        _append('✅ Offline done → $outDir');
      });
    } catch (e, st) {
      _append('❌ Offline error: $e\n$st');
    } finally {
      setState(() {
        _busy = false;
      });
    }
  }

  // ---------------- LIVE ----------------

  Future<void> _startLive() async {
    if (_busy || _liveRunning) return;
    setState(() {
      _busy = true;
      _log = 'Starting live capture…';
    });

    try {
      final docs = await getApplicationDocumentsDirectory();
      final runDir =
          Directory('${docs.path}/run_live_${DateTime.now().millisecondsSinceEpoch}');
      await runDir.create(recursive: true);

      _live = Live2DSession();
      await _live!.start(runDir: runDir, target2dFps: 10, yoloStride: 2);

      // Subscribe to on-device live logs; throttle UI updates to avoid jank
      _liveLogSub = _live!.logStream.listen((line) {
        _liveLogBuffer.writeln(line);
        _liveFlushTimer ??= Timer(const Duration(milliseconds: 200), () {
          if (!mounted) return;
          setState(() {
            final chunk = _liveLogBuffer.toString();
            if (chunk.isNotEmpty) {
              _log = _log.isEmpty ? chunk : '$_log\n$chunk';
            }
            _liveLogBuffer.clear();
          });
          _liveFlushTimer = null;
        });
      });

      setState(() {
        _liveRunning = true;
        _append('🎥 Live running. Press STOP to finish.');
      });
    } catch (e, st) {
      _append('❌ Live start error: $e\n$st');
    } finally {
      setState(() {
        _busy = false;
      });
    }
  }

  Future<void> _stopLive() async {
    if (_busy || !_liveRunning || _live == null) return;
    setState(() {
      _busy = true;
      _append('Stopping live capture…');
    });

    try {
      final outDir = await _live!.stopAndFinalize();
      setState(() {
        _lastRunDir = outDir;
        _append('✅ Live done → $outDir');
      });
    } catch (e, st) {
      _append('❌ Live stop error: $e\n$st');
    } finally {
      // Tear down live log subscription and timers
      await _liveLogSub?.cancel();
      _liveLogSub = null;
      _liveFlushTimer?.cancel();
      _liveFlushTimer = null;
      _liveLogBuffer.clear();

      _liveRunning = false;
      _live = null;
      setState(() {
        _busy = false;
      });
    }
  }

  @override
  void dispose() {
    _liveLogSub?.cancel();
    _liveFlushTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final runLabel = _lastRunDir == null ? '' : '\nOutput: $_lastRunDir';

    return Scaffold(
      appBar: AppBar(title: const Text('YOLO → RTMPose → MotionBERT (Live/Offline)')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                ElevatedButton.icon(
                  onPressed: _busy ? null : _runOffline,
                  icon: const Icon(Icons.video_file),
                  label: const Text('Run OFFLINE (pick video)'),
                ),
                if (!_liveRunning)
                  ElevatedButton.icon(
                    onPressed: _busy ? null : _startLive,
                    icon: const Icon(Icons.fiber_manual_record),
                    style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                    label: const Text('START LIVE'),
                  ),
                if (_liveRunning)
                  ElevatedButton.icon(
                    onPressed: _busy ? null : _stopLive,
                    icon: const Icon(Icons.stop),
                    style:
                        ElevatedButton.styleFrom(backgroundColor: Colors.black87),
                    label: const Text('STOP LIVE'),
                  ),
              ],
            ),
            const SizedBox(height: 12),
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.black12),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: SingleChildScrollView(child: Text('$_log$runLabel')),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
