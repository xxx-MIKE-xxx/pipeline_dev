// lib/main.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import 'run_rgb_to_motionbert3d.dart';

void main() => runApp(const PoseApp());

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

  final _cfg = PoseRunnerConfig();

  Future<void> _pickAndRun() async {
    final picker = ImagePicker();
    final x = await picker.pickVideo(source: ImageSource.gallery);
    if (x == null) return;

    setState(() { _busy = true; _log = 'Processing…'; });
    try {
      final runner = PoseRunner(_cfg);
      final outDir = await runner.runOnVideo(File(x.path));
      setState(() {
        _lastRunDir = outDir;
        _log = 'Done.\nSaved:\n$outDir';
      });
    } catch (e, st) {
      setState(() { _log = 'Error: $e'; });
    } finally {
      setState(() { _busy = false; });
    }
  }

  Future<void> _shareResults() async {
    if (_lastRunDir == null) return;
    final dir = Directory(_lastRunDir!);
    final files = await dir.list().toList();
    final targets = <XFile>[];
    for (final f in files) {
      if (f is File && (f.path.endsWith('.json') || f.path.endsWith('.mp4'))) {
        targets.add(XFile(f.path));
      }
    }
    if (targets.isEmpty) return;
    await Share.shareXFiles(targets, subject: 'Pose estimation results');
  }

  @override
  Widget build(BuildContext context) {
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
              Row(
                children: [
                  Expanded(child: Text('Output: $_lastRunDir')),
                  const SizedBox(width: 8),
                  ElevatedButton.icon(
                    onPressed: _busy ? null : _shareResults,
                    icon: const Icon(Icons.ios_share),
                    label: const Text('Share results'),
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
                child: SingleChildScrollView(child: Text(_log)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
