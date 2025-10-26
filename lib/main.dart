// main.dart
import 'dart:async';
import 'dart:developer' as developer;
import 'dart:io';
import 'dart:ui' as ui show PlatformDispatcher;

import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show FlutterError, debugPrint;
import 'package:image_picker/image_picker.dart';

// Legacy API (pre-11)
import 'package:share_plus/share_plus.dart'; // provides Share.shareXFiles
import 'package:cross_file/cross_file.dart';

import 'package:path_provider/path_provider.dart';

import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit_config.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/ffprobe_kit.dart';

import 'run_rgb_to_motionbert3d.dart';

// -------------------- global logging helpers --------------------

late IOSink _fileSink;

Future<void> _initFileLogger() async {
  final docs = await getApplicationDocumentsDirectory();
  final f = File('${docs.path}/pose_run.log');
  _fileSink = f.openWrite(mode: FileMode.append);
  debugPrint('📄 File logging -> ${f.path}');
}

void _logToAll(String msg, {Object? error, StackTrace? stack}) {
  final line = '[${DateTime.now().toIso8601String()}] $msg'
      '${error != null ? '\nERROR: $error' : ''}'
      '${stack != null ? '\n$stack' : ''}';
  debugPrint(line);
  developer.log(msg, error: error, stackTrace: stack);
  try {
    _fileSink.writeln(line);
    _fileSink.flush();
  } catch (_) {
    // ignore file errors; keep app running
  }
}

Future<void> _initFfmpegLogging() async {
  // FFmpegKit native logs -> our logger
  FFmpegKitConfig.enableLogCallback((log) {
    final m = log.getMessage();
    if (m != null) _logToAll('[FFmpeg] $m');
  });

  // FFmpegKit statistics -> our logger
  FFmpegKitConfig.enableStatisticsCallback((stats) {
    final t = stats.getTime();      // ms
    final s = stats.getSize();      // bytes
    final sp = stats.getSpeed();    // e.g. 2.0x (double)
    _logToAll('[FFmpeg] stats t=${t ?? -1}ms size=${s ?? -1}B speed=${sp ?? 0}');
  });
}

// -------------------- main() with robust error catching --------------------

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await _initFileLogger();
  await _initFfmpegLogging();

  // Catch framework errors
  FlutterError.onError = (details) {
    _logToAll('Flutter error', error: details.exception, stack: details.stack);
  };

  // Catch uncaught async errors on the UI/platform dispatcher
  ui.PlatformDispatcher.instance.onError = (error, stack) {
    _logToAll('PlatformDispatcher error', error: error, stack: stack);
    return true; // handled
  };

  // Catch everything else
  runZonedGuarded(() {
    runApp(const PoseApp());
  }, (error, stack) {
    _logToAll('Uncaught (zone)', error: error, stack: stack);
  });
}

// -------------------- UI --------------------

class PoseApp extends StatelessWidget {
  const PoseApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pose Estimation',
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
  String? _lastRunDir;
  bool _busy = false;
  String _log = '';

  Future<void> _pickAndRun() async {
    final picker = ImagePicker();
    final picked = await picker.pickVideo(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() {
      _busy = true;
      _log = 'Running YOLO → RTMPose → MotionBERT…\n';
    });

    // simple throttle to avoid too-frequent setState() calls
    DateTime lastUiUpdate = DateTime.now();

    String appendAndMaybePaint(String current, String line) {
      final now = DateTime.now();
      final next = current.isEmpty ? line : '$current\n$line';
      if (now.difference(lastUiUpdate).inMilliseconds > 150 && mounted) {
        lastUiUpdate = now;
        setState(() => _log = next);
      }
      return next;
    }

    try {
      String uiLog = _log;

      final outDirPath = await runPipelineOnVideo(
        File(picked.path),
        onLog: (s) {
          _logToAll(s);                          // file + console
          uiLog = appendAndMaybePaint(uiLog, s); // on-screen
        },
        onProgress: (p, stage) {
          final line = '${(p * 100).toStringAsFixed(1)}%  $stage';
          _logToAll(line);                       // file + console
          uiLog = appendAndMaybePaint(uiLog, line); // on-screen
        },
      );

      if (!mounted) return;
      setState(() {
        _lastRunDir = outDirPath;
        _log = '$uiLog\n\nDone.\nSaved results to:\n$outDirPath\n'
               'Files:\n- yolo_out.json\n- rtm_out.json\n- motionbert_out.json';
      });
    } catch (e, st) {
      _logToAll('Pipeline error', error: e, stack: st);
      if (mounted) setState(() => _log = 'Error: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _shareResults() async {
    if (_lastRunDir == null) return;
    final dir = Directory(_lastRunDir!);
    final files = [
      XFile('${dir.path}/yolo_out.json'),
      XFile('${dir.path}/rtm_out.json'),
      XFile('${dir.path}/motionbert_out.json'),
    ];
    await Share.shareXFiles(files, text: 'Pose estimation results');
  }

  @override
  Widget build(BuildContext context) {
    final canShare = _lastRunDir != null && !_busy;

    return Scaffold(
      appBar: AppBar(title: const Text('YOLO → RTMPose → MotionBERT')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            ElevatedButton.icon(
              onPressed: _busy ? null : _pickAndRun,
              icon: const Icon(Icons.upload),
              label: const Text('Pick video from gallery'),
            ),
            const SizedBox(height: 12),
            if (_lastRunDir != null)
              SelectableText('Output: $_lastRunDir', maxLines: 4),
            const SizedBox(height: 12),
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.black12),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: SingleChildScrollView(child: Text(_log)),
              ),
            ),
            if (_lastRunDir != null)
              ElevatedButton.icon(
                onPressed: canShare ? _shareResults : null,
                icon: const Icon(Icons.share),
                label: const Text('Share results'),
              ),
          ],
        ),
      ),
    );
  }
}