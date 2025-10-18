import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

// Legacy API (pre-11)
import 'package:share_plus/share_plus.dart'; // provides Share.shareXFiles
import 'package:cross_file/cross_file.dart';

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

  Future<void> _pickAndRun() async {
    final picker = ImagePicker();
    final picked = await picker.pickVideo(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() { _busy = true; _log = 'Running YOLO → RTMPose → MotionBERT…'; });

    try {
      final outDirPath = await runPipelineOnVideo(File(picked.path));
      setState(() {
        _lastRunDir = outDirPath;
        _log = 'Done.\nSaved results to:\n$outDirPath\n\n'
               'Files:\n- yolo_out.json\n- rtm_out.json\n- motionbert_out.json';
      });
    } catch (e) {
      setState(() => _log = 'Error: $e');
    } finally {
      setState(() => _busy = false);
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
