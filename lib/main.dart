import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:share_plus/share_plus.dart';

import 'run_rgb_to_motionbert3d.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const PoseApp());
}

class PoseApp extends StatelessWidget {
  const PoseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MotionBERT Pipeline',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const PoseHomePage(),
    );
  }
}

class PoseHomePage extends StatefulWidget {
  const PoseHomePage({super.key});

  @override
  State<PoseHomePage> createState() => _PoseHomePageState();
}

class _PoseHomePageState extends State<PoseHomePage> {
  PosePipeline? _pipeline;
  PoseRunResult? _lastResult;
  bool _initializing = true;
  bool _running = false;
  String _status = 'Preparing models...';
  String? _error;

  @override
  void initState() {
    super.initState();
    _initialisePipeline();
  }

  Future<void> _initialisePipeline() async {
    try {
      final pipeline = await PosePipeline.create();
      if (!mounted) return;
      setState(() {
        _pipeline = pipeline;
        _status = 'Ready';
        _initializing = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _initializing = false;
        _error = e.toString();
        _status = 'Failed to initialise';
      });
    }
  }

  Future<void> _pickAndProcess() async {
    final pipeline = _pipeline;
    if (pipeline == null || _running) {
      return;
    }

    final picker = ImagePicker();
    final picked = await picker.pickVideo(source: ImageSource.gallery);
    if (picked == null) {
      return;
    }

    setState(() {
      _running = true;
      _status = 'Processing ${picked.name}...';
      _error = null;
    });

    try {
      final result = await pipeline.processVideo(File(picked.path));
      if (!mounted) return;
      setState(() {
        _lastResult = result;
        _status = 'Completed (${result.frameCount} frames)';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _status = 'Processing failed';
      });
    } finally {
      if (!mounted) return;
      setState(() {
        _running = false;
      });
    }
  }

  Future<void> _shareResults() async {
    final result = _lastResult;
    if (result == null) {
      return;
    }

    final files = await result.runDirectory
        .list(recursive: true)
        .whereType<File>()
        .toList();
    if (files.isEmpty) {
      return;
    }

    final xfiles = files.map((file) => XFile(file.path)).toList();
    await Share.shareXFiles(
      xfiles,
      text: 'MotionBERT pipeline outputs for ${result.frameCount} frames.',
      subject: 'MotionBERT Pose Results',
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('RGB → MotionBERT Pipeline'),
      ),
      body: _initializing
          ? const Center(child: CircularProgressIndicator())
          : Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text('Status: $_status'),
                  if (_error != null) ...[
                    const SizedBox(height: 8),
                    Text(
                      _error!,
                      style: const TextStyle(color: Colors.redAccent),
                    ),
                  ],
                  const SizedBox(height: 16),
                  ElevatedButton.icon(
                    onPressed: (_pipeline == null || _running) ? null : _pickAndProcess,
                    icon: const Icon(Icons.video_library),
                    label: const Text('Select video from gallery'),
                  ),
                  if (_running) ...[
                    const SizedBox(height: 12),
                    const LinearProgressIndicator(),
                  ],
                  const SizedBox(height: 16),
                  if (_lastResult != null)
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Text(
                            'Run folder: ${_lastResult!.runDirectory.path}',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                          const SizedBox(height: 12),
                          Text('Generated files', style: Theme.of(context).textTheme.titleMedium),
                          const SizedBox(height: 8),
                          Expanded(
                            child: Card(
                              child: ListView(
                                children: _lastResult!.outputs.entries
                                    .toList()
                                    ..sort((a, b) => a.key.compareTo(b.key))
                                    .map(
                                      (entry) => ListTile(
                                        dense: true,
                                        title: Text(entry.key),
                                        subtitle: Text(entry.value.path),
                                        leading: const Icon(Icons.insert_drive_file),
                                      ),
                                    )
                                    .toList(),
                              ),
                            ),
                          ),
                          const SizedBox(height: 12),
                          ElevatedButton.icon(
                            onPressed: _running ? null : _shareResults,
                            icon: const Icon(Icons.ios_share),
                            label: const Text('Share results'),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
    );
  }
}
