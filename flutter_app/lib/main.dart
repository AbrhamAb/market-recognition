import 'dart:io';

import 'package:flutter/material.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import 'api_client.dart';
import 'history_store.dart';
import 'predict_state.dart';

const String appName = 'wagaLine';
const Color sand = Color(0xFFF8F4EC);
const Color warmGreen = Color(0xFF4CAF50);
const Color goldenYellow = Color(0xFFF5C242);
const Color deepRed = Color(0xFFD64545);
const Color darkText = Color(0xFF1F1B16);
const Color midGray = Color(0xFF6B6B6B);

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Hive.initFlutter();
  runApp(const MarketRecApp());
}

class MarketRecApp extends StatelessWidget {
  const MarketRecApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(
            create: (_) => PredictState(ApiClient(), HistoryStore())),
      ],
      child: MaterialApp(
        title: appName,
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          useMaterial3: true,
          scaffoldBackgroundColor: sand,
          colorScheme: ColorScheme.fromSeed(
            seedColor: warmGreen,
            primary: warmGreen,
            secondary: goldenYellow,
            surface: Colors.white,
          ).copyWith(onPrimary: Colors.white, onSurface: darkText),
          appBarTheme: const AppBarTheme(
            backgroundColor: sand,
            foregroundColor: darkText,
            elevation: 0,
          ),
          filledButtonTheme: FilledButtonThemeData(
            style: FilledButton.styleFrom(
              backgroundColor: warmGreen,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
              padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 16),
            ),
          ),
          outlinedButtonTheme: OutlinedButtonThemeData(
            style: OutlinedButton.styleFrom(
              foregroundColor: darkText,
              side: const BorderSide(color: Color(0xFFD8D2C4)),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
              padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 16),
            ),
          ),
          chipTheme: ChipThemeData(
            backgroundColor: const Color(0xFFEAE3D5),
            labelStyle: const TextStyle(color: darkText),
            selectedColor: warmGreen.withAlpha((0.12 * 255).round()),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
          textTheme: const TextTheme(
            headlineMedium: TextStyle(
              color: darkText,
              fontWeight: FontWeight.w700,
            ),
            titleLarge: TextStyle(
              color: darkText,
              fontWeight: FontWeight.w700,
            ),
            titleMedium: TextStyle(
              color: darkText,
              fontWeight: FontWeight.w600,
            ),
            bodyMedium: TextStyle(color: darkText),
            bodySmall: TextStyle(color: midGray),
          ),
        ),
        home: const SplashGate(),
      ),
    );
  }
}

class SplashGate extends StatefulWidget {
  const SplashGate({super.key});

  @override
  State<SplashGate> createState() => _SplashGateState();
}

class _SplashGateState extends State<SplashGate> {
  bool _showSplash = true;

  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(milliseconds: 1200), () {
      if (mounted) {
        setState(() => _showSplash = false);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return _showSplash ? const SplashScreen() : const HomeShell();
  }
}

class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: sand,
      body: Stack(
        children: [
          const Positioned.fill(child: TextileBackground()),
          Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 88,
                  height: 88,
                  decoration: BoxDecoration(
                    color: warmGreen,
                    borderRadius: BorderRadius.circular(22),
                    boxShadow: const [
                      BoxShadow(
                        color: Color(0x442A8A2A),
                        blurRadius: 16,
                        offset: Offset(0, 10),
                      )
                    ],
                  ),
                  child: const Icon(Icons.handshake,
                      color: Colors.white, size: 44),
                ),
                const SizedBox(height: 16),
                const Text(
                  appName,
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w700,
                    color: darkText,
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  'Fair prices for everyday markets',
                  style: TextStyle(fontSize: 16, color: midGray),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 12),
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: const [
                    _FlagDot(color: warmGreen),
                    SizedBox(width: 6),
                    _FlagDot(color: goldenYellow),
                    SizedBox(width: 6),
                    _FlagDot(color: deepRed),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _index,
        children: const [
          ScanScreen(),
          HistoryScreen(),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _index,
        onTap: (value) => setState(() => _index = value),
        selectedItemColor: warmGreen,
        unselectedItemColor: midGray,
        backgroundColor: Colors.white,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.center_focus_strong),
            label: 'Scan',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.history),
            label: 'History',
          ),
        ],
      ),
    );
  }
}

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  final TextEditingController _vendorCtrl =
      TextEditingController(text: 'V0001');
  final ImagePicker _picker = ImagePicker();
  double _qty = 1.0;

  @override
  void dispose() {
    _vendorCtrl.dispose();
    super.dispose();
  }

  Future<void> _pick(ImageSource source) async {
    final xfile = await _picker.pickImage(source: source, imageQuality: 85);
    if (xfile != null && mounted) {
      context.read<PredictState>().setImage(File(xfile.path));
    }
  }

  Future<void> _runPrediction() async {
    final state = context.read<PredictState>();
    await state.predict(
      vendorId: _vendorCtrl.text.trim(),
      qty: _qty,
    );
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<PredictState>();
    final hasImage = state.image != null;

    return SafeArea(
      child: Stack(
        children: [
          SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _Header(),
                const SizedBox(height: 12),
                const PatternDivider(),
                const SizedBox(height: 16),
                _CameraCard(
                  image: state.image,
                  onCamera: () => _pick(ImageSource.camera),
                  onGallery: () => _pick(ImageSource.gallery),
                ),
                const SizedBox(height: 12),
                const Text(
                  'Take a clear photo of the item.',
                  style: TextStyle(color: midGray, fontSize: 14),
                ),
                const SizedBox(height: 16),
                _QuantitySelector(
                  qty: _qty,
                  onChanged: (value) => setState(() => _qty = value),
                ),
                const SizedBox(height: 12),
                _InputCard(
                  label: 'Vendor ID',
                  child: TextField(
                    controller: _vendorCtrl,
                    decoration: const InputDecoration(
                      hintText: 'Enter vendor ID',
                      border: InputBorder.none,
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                // Buy price input removed — app uses market prices from backend
                const SizedBox(height: 16),
                FilledButton(
                  onPressed: state.busy ? null : _runPrediction,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      if (state.busy)
                        const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      else
                        const Icon(Icons.document_scanner),
                      const SizedBox(width: 8),
                      const Text('Scan Item'),
                    ],
                  ),
                ),
                const SizedBox(height: 10),
                OutlinedButton(
                  onPressed:
                      state.busy ? null : () => _pick(ImageSource.gallery),
                  child: const Text('Upload Photo'),
                ),
                if (state.error != null) ...[
                  const SizedBox(height: 12),
                  Text(
                    state.error!,
                    style: const TextStyle(
                        color: deepRed, fontWeight: FontWeight.w600),
                  ),
                ],
                if (state.result != null) ...[
                  const SizedBox(height: 16),
                  ResultCard(
                    result: state.result!,
                    saving: state.saving,
                    onAccept: () async {
                      final messenger = ScaffoldMessenger.of(context);
                      final added = await state.acceptCurrentResult();
                      if (added && mounted) {
                        messenger.showSnackBar(
                          const SnackBar(content: Text('Saved to history')),
                        );
                      }
                    },
                    onRescan: () {
                      state.clearResult();
                      state.setImage(null);
                    },
                  ),
                ],
                SizedBox(height: MediaQuery.of(context).padding.bottom + 12),
              ],
            ),
          ),
          if (state.busy)
            Positioned.fill(
              child: Container(
                color: Colors.black.withAlpha((0.12 * 255).round()),
              ),
            ),
          if (!hasImage)
            Positioned(
              right: 16,
              bottom: 90,
              child: FloatingActionButton.extended(
                heroTag: 'camera_fab',
                backgroundColor: warmGreen,
                onPressed: state.busy ? null : () => _pick(ImageSource.camera),
                label: const Text('Camera'),
                icon: const Icon(Icons.camera_alt),
              ),
            ),
        ],
      ),
    );
  }
}

class ResultCard extends StatelessWidget {
  const ResultCard({
    super.key,
    required this.result,
    required this.saving,
    required this.onAccept,
    required this.onRescan,
  });

  final Map<String, dynamic> result;
  final bool saving;
  final Future<void> Function() onAccept;
  final VoidCallback onRescan;

  @override
  Widget build(BuildContext context) {
    final confidence = (result['confidence'] as num?)?.toDouble() ?? 0.0;
    final low = result['low_confidence'] == true || confidence < 0.4;
    final confidenceLabel = _confidenceLabel(confidence, low);
    final pricePerUnit = (result['price_per_unit'] as num?)?.toDouble();
    final total = (result['total'] as num?)?.toDouble();
    final qty = (result['qty'] as num?)?.toDouble();
    final unit = (result['unit'] ?? '').toString();
    final topK = ((result['top_k'] as List?) ?? []).whereType<Map>().toList();

    Future<void> handleAccept() async {
      if (saving) return;
      if (low) {
        final confirm = await showDialog<bool>(
          context: context,
          builder: (ctx) {
            return AlertDialog(
              title: const Text('Confirm item'),
              content: const Text(
                'Confidence is low. Please confirm the item before accepting the price.',
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(ctx).pop(false),
                  child: const Text('Cancel'),
                ),
                FilledButton(
                  onPressed: () => Navigator.of(ctx).pop(true),
                  child: const Text('Accept'),
                ),
              ],
            );
          },
        );
        if (confirm != true) return;
      }
      await onAccept();
    }

    return Stack(
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(18),
            boxShadow: const [
              BoxShadow(
                color: Color(0x22000000),
                blurRadius: 12,
                offset: Offset(0, 8),
              )
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      (result['item'] ?? 'Detected item').toString(),
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                  ),
                  _ConfidenceChip(
                      label: confidenceLabel, confidence: confidence, low: low),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                pricePerUnit != null
                    ? 'Price per unit: ${pricePerUnit.toStringAsFixed(2)} $unit'
                    : 'Price per unit: n/a',
                style: const TextStyle(
                    color: darkText, fontWeight: FontWeight.w600),
              ),
              if (qty != null)
                Text(
                    'Qty: ${qty.toStringAsFixed(qty == qty.roundToDouble() ? 0 : 2)}'),
              const SizedBox(height: 4),
              Text(
                total != null
                    ? '${total.toStringAsFixed(2)} ${unit.isEmpty ? '' : unit}'
                    : 'Total pending',
                style: const TextStyle(
                  color: warmGreen,
                  fontWeight: FontWeight.w800,
                  fontSize: 28,
                ),
              ),
              if (low) ...[
                const SizedBox(height: 6),
                const Text(
                  'Please confirm the item.',
                  style: TextStyle(color: deepRed, fontWeight: FontWeight.w600),
                ),
              ],
              if (topK.isNotEmpty) ...[
                const SizedBox(height: 12),
                const Text('Alternative suggestions:'),
                const SizedBox(height: 6),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: topK
                      .map((e) =>
                          Chip(label: Text((e['label'] ?? '').toString())))
                      .toList(),
                ),
              ],
              const SizedBox(height: 14),
              FilledButton(
                onPressed: saving ? null : handleAccept,
                child: const Text('Accept Price'),
              ),
              const SizedBox(height: 8),
              OutlinedButton(
                onPressed: saving ? null : onRescan,
                child: const Text('Rescan'),
              ),
            ],
          ),
        ),
        if (saving)
          Positioned.fill(
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white.withAlpha((0.72 * 255).round()),
                borderRadius: BorderRadius.circular(18),
              ),
              child: const Center(
                child: SizedBox(
                  width: 32,
                  height: 32,
                  child: CircularProgressIndicator(strokeWidth: 3),
                ),
              ),
            ),
          ),
      ],
    );
  }
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final Set<int> _selected = <int>{};

  void _toggleSelect(int index) {
    setState(() {
      if (_selected.contains(index)) {
        _selected.remove(index);
      } else {
        _selected.add(index);
      }
    });
  }

  void _clearSelection() {
    setState(() => _selected.clear());
  }

  Future<void> _confirmAndDelete(BuildContext context) async {
    final count = _selected.length;
    if (count == 0) return;
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Delete $count item${count > 1 ? 's' : ''}?'),
        content:
            const Text('This will permanently remove the selected records.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.of(ctx).pop(false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.of(ctx).pop(true),
              child: const Text('Delete')),
        ],
      ),
    );
    if (confirm != true) return;

    // ignore: use_build_context_synchronously
    final state = context.read<PredictState>();
    final indices = _selected.toList();
    await state.deleteHistory(indices);
    _clearSelection();
    if (mounted) {
      // ignore: use_build_context_synchronously
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Deleted $count record${count > 1 ? 's' : ''}')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final state = context.watch<PredictState>();
    final history = state.history;

    final selectionMode = _selected.isNotEmpty;

    return SafeArea(
      child: Scaffold(
        backgroundColor: sand,
        appBar: AppBar(
          title: selectionMode
              ? Text('${_selected.length} selected')
              : Row(
                  children: const [
                    Icon(Icons.handshake, color: warmGreen),
                    SizedBox(width: 8),
                    Text(appName),
                  ],
                ),
          actions: selectionMode
              ? [
                  IconButton(
                    onPressed: () => _clearSelection(),
                    icon: const Icon(Icons.close, color: darkText),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(right: 8.0),
                    child: IconButton(
                      onPressed: () => _confirmAndDelete(context),
                      icon: const Icon(Icons.delete_forever, color: deepRed),
                    ),
                  ),
                ]
              : const [
                  Padding(
                    padding: EdgeInsets.only(right: 12),
                    child: Icon(Icons.search, color: darkText),
                  ),
                ],
        ),
        body: state.loadingHistory
            ? const Center(child: CircularProgressIndicator())
            : history.isEmpty
                ? const _EmptyState()
                : ListView.separated(
                    padding: const EdgeInsets.all(16),
                    itemCount: history.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 10),
                    itemBuilder: (context, index) {
                      final record = history[index];
                      final selected = _selected.contains(index);
                      return GestureDetector(
                        onLongPress: () => _toggleSelect(index),
                        onTap: () {
                          if (_selected.isNotEmpty) _toggleSelect(index);
                        },
                        child: _HistoryTile(
                          record: record,
                          selected: selected,
                        ),
                      );
                    },
                  ),
      ),
    );
  }
}

class _Header extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        const Icon(Icons.handshake, color: warmGreen, size: 26),
        const SizedBox(width: 8),
        const Text(
          appName,
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
        ),
        const Spacer(),
        Consumer<PredictState>(builder: (context, state, _) {
          return Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('On-device', style: TextStyle(color: midGray)),
              const SizedBox(width: 6),
              Switch.adaptive(
                value: state.useOnDevice,
                onChanged: (v) => state.setUseOnDevice(v),
              ),
              IconButton(
                icon: const Icon(Icons.help_outline, color: midGray),
                onPressed: () => showDialog<void>(
                  context: context,
                  builder: (ctx) => const AppHelpDialog(),
                ),
              ),
            ],
          );
        }),
      ],
    );
  }
}

class AppHelpDialog extends StatelessWidget {
  const AppHelpDialog({super.key});

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Container(
              width: 84,
              height: 84,
              decoration: BoxDecoration(
                color: warmGreen,
                borderRadius: BorderRadius.circular(18),
              ),
              child: const Icon(Icons.handshake, color: Colors.white, size: 44),
            ),
            const SizedBox(height: 12),
            const Text(
              appName,
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            const Text(
              'Fair prices for everyday markets. \n\nTake a photo of an item to get an estimated market price, save transactions to history, and manage them from the History tab.',
              textAlign: TextAlign.center,
              style: TextStyle(color: midGray),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Close'),
        ),
      ],
    );
  }
}

class _CameraCard extends StatelessWidget {
  const _CameraCard({
    required this.image,
    required this.onCamera,
    required this.onGallery,
  });

  final File? image;
  final VoidCallback onCamera;
  final VoidCallback onGallery;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: const [
          BoxShadow(
            color: Color(0x22000000),
            blurRadius: 12,
            offset: Offset(0, 6),
          )
        ],
      ),
      child: Column(
        children: [
          ClipRRect(
            borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
            child: AspectRatio(
              aspectRatio: 4 / 3,
              child: image != null
                  ? Image.file(image!, fit: BoxFit.cover)
                  : Container(
                      color: sand,
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: const [
                          Icon(Icons.photo_camera_back_outlined,
                              size: 48, color: midGray),
                          SizedBox(height: 8),
                          Text('Camera preview',
                              style: TextStyle(color: midGray)),
                        ],
                      ),
                    ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _ActionChip(
                    label: 'Camera', icon: Icons.camera_alt, onTap: onCamera),
                _ActionChip(
                    label: 'Gallery',
                    icon: Icons.photo_outlined,
                    onTap: onGallery),
              ],
            ),
          )
        ],
      ),
    );
  }
}

class _ActionChip extends StatelessWidget {
  const _ActionChip(
      {required this.label, required this.icon, required this.onTap});

  final String label;
  final IconData icon;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return OutlinedButton.icon(
      style: OutlinedButton.styleFrom(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        side: const BorderSide(color: Color(0xFFD8D2C4)),
      ),
      onPressed: onTap,
      icon: Icon(icon, size: 18, color: darkText),
      label: Text(label),
    );
  }
}

class _QuantitySelector extends StatelessWidget {
  const _QuantitySelector({required this.qty, required this.onChanged});

  final double qty;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    return _InputCard(
      label: 'Quantity',
      child: Row(
        children: [
          _RoundButton(
            icon: Icons.remove,
            onTap: () => onChanged((qty - 0.25).clamp(0.25, 9999)),
          ),
          Expanded(
            child: Center(
              child: Text(
                qty % 1 == 0
                    ? qty.toStringAsFixed(0)
                    : qty % 0.25 == 0
                        ? qty.toStringAsFixed(2)
                        : qty.toStringAsFixed(2),
                style:
                    const TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
              ),
            ),
          ),
          _RoundButton(
            icon: Icons.add,
            onTap: () => onChanged((qty + 0.25).clamp(0.25, 9999)),
          ),
          const SizedBox(width: 8),
          const Text('kg', style: TextStyle(color: midGray)),
        ],
      ),
    );
  }
}

class _RoundButton extends StatelessWidget {
  const _RoundButton({required this.icon, required this.onTap});

  final IconData icon;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: Container(
        width: 36,
        height: 36,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: sand,
          border: Border.all(color: const Color(0xFFD8D2C4)),
        ),
        child: Icon(icon, color: darkText, size: 20),
      ),
    );
  }
}

class _InputCard extends StatelessWidget {
  const _InputCard({required this.label, required this.child});

  final String label;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontWeight: FontWeight.w700)),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: const Color(0xFFD8D2C4)),
          ),
          child: child,
        ),
      ],
    );
  }
}

class _ConfidenceChip extends StatelessWidget {
  const _ConfidenceChip(
      {required this.label, required this.confidence, required this.low});

  final String label;
  final double confidence;
  final bool low;

  @override
  Widget build(BuildContext context) {
    final color = low
        ? deepRed
        : confidence >= 0.7
            ? warmGreen
            : goldenYellow;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withAlpha((0.12 * 255).round()),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withAlpha((0.6 * 255).round())),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.circle, size: 10, color: color),
          const SizedBox(width: 6),
          Text(label,
              style: TextStyle(color: color, fontWeight: FontWeight.w700)),
        ],
      ),
    );
  }
}

class _HistoryTile extends StatelessWidget {
  const _HistoryTile({required this.record, this.selected = false});

  final PredictionRecord record;
  final bool selected;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color:
            selected ? warmGreen.withAlpha((0.06 * 255).round()) : Colors.white,
        borderRadius: BorderRadius.circular(14),
        border:
            Border.all(color: selected ? warmGreen : const Color(0xFFE4DECF)),
      ),
      child: Row(
        children: [
          Container(
            width: 52,
            height: 52,
            decoration: BoxDecoration(
              color: sand,
              borderRadius: BorderRadius.circular(12),
              image: record.imagePath != null
                  ? DecorationImage(
                      image: FileImage(File(record.imagePath!)),
                      fit: BoxFit.cover,
                    )
                  : null,
            ),
            child: record.imagePath == null
                ? const Icon(Icons.shopping_bag_outlined, color: midGray)
                : null,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(record.item,
                    style: const TextStyle(fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                Text('ETB ${record.total.toStringAsFixed(2)}',
                    style: const TextStyle(
                        color: warmGreen, fontWeight: FontWeight.w700)),
                Text(
                  _timeAgo(record.timestamp),
                  style: const TextStyle(color: midGray, fontSize: 12),
                ),
              ],
            ),
          ),
          selected
              ? Icon(Icons.check_circle, color: warmGreen)
              : const Icon(Icons.chevron_right, color: midGray),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: const [
          Icon(Icons.history, size: 48, color: midGray),
          SizedBox(height: 8),
          Text('No scans yet — start with Scan Item.',
              textAlign: TextAlign.center),
        ],
      ),
    );
  }
}

class TextileBackground extends StatelessWidget {
  const TextileBackground({super.key});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _TextilePainter(),
    );
  }
}

class _TextilePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = darkText.withAlpha((0.04 * 255).round())
      ..strokeWidth = 1;
    const double step = 24;
    for (double y = 0; y < size.height; y += step) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y - 12), paint);
    }
    for (double x = 0; x < size.width; x += step) {
      canvas.drawLine(Offset(x, 0), Offset(x - 12, size.height), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class PatternDivider extends StatelessWidget {
  const PatternDivider({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 1,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            Colors.transparent,
            darkText.withAlpha((0.08 * 255).round()),
            Colors.transparent,
          ],
        ),
      ),
    );
  }
}

class _FlagDot extends StatelessWidget {
  const _FlagDot({required this.color});

  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 12,
      height: 6,
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(4),
      ),
    );
  }
}

String _confidenceLabel(double confidence, bool low) {
  if (low) return 'Low';
  if (confidence >= 0.7) return 'High';
  if (confidence >= 0.4) return 'Medium';
  return 'Low';
}

String _timeAgo(DateTime timestamp) {
  final diff = DateTime.now().difference(timestamp);
  if (diff.inMinutes < 1) return 'Just now';
  if (diff.inHours < 1) return '${diff.inMinutes} mins ago';
  if (diff.inHours < 24) return '${diff.inHours} hours ago';
  if (diff.inDays < 7) return '${diff.inDays} days ago';
  final weeks = (diff.inDays / 7).floor();
  if (weeks < 5) return '$weeks weeks ago';
  final months = (diff.inDays / 30).floor();
  return months <= 1 ? '1 month ago' : '$months months ago';
}
