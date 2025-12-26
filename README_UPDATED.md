# Market Recognition — Informal Market Product Recognition (Updated)

This file is a completed, consolidated README covering setup, running, model handling, Flutter app builds, testing, and deployment.

## Contents
- Requirements
- Backend (FastAPI) — setup & run
- Model training & TFLite conversion
- Flutter mobile client — build & install
- Testing connectivity (USB and Wi‑Fi)
- Release & signing
- Troubleshooting checklist

---

## Requirements

- Windows 10/11 (PowerShell)
- Python 3.11 (Conda recommended)
- Java JDK 11+
- Android SDK (platform-tools, build tools)
- Flutter SDK
- Android device with USB debugging for testing

---

## Backend (FastAPI)

1. Setup environment (Conda recommended):
```powershell
conda create -n marketrec python=3.11 -y
conda activate marketrec
pip install --upgrade pip
pip install -r requirements.txt
```

2. Start backend (development):
```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` — health check
- `GET /model/info` — loaded model info
- `POST /predict` — multipart form `file=@...` + optional `vendor_id`, `qty`, `buy_price_per_unit`

Model reload:
- `POST /model/reload` to ask the server to reload model/labels from disk.

If uvicorn fails at startup, capture logs:
```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1
Get-Content uvicorn.log -Tail 200
```

---

## Model

Training example (local):
```powershell
conda activate marketrec
python model/train.py --data-dir ./dataset --output-dir ./model --epochs 10 --batch-size 32 --img-size 224 224
```
Output example:
- `model/run_*/product_classifier` (SavedModel)
- `model/run_*/product_classifier.keras` or `.h5`
- `model/run_*/labels.txt`

TFLite conversion (for on-device inference):
```python
import tensorflow as tf
m = tf.keras.models.load_model('model/run_x/product_classifier')
converter = tf.lite.TFLiteConverter.from_keras_model(m)
# optionally set optimizations
tflite = converter.convert()
open('product_classifier.tflite','wb').write(tflite)
```
Place `product_classifier.tflite` and `labels.txt` in `flutter_app/assets/models/` and update `pubspec.yaml` if needed.

---

## Flutter mobile app

Project: `flutter_app/`
Assets:
- `flutter_app/assets/models/product_classifier.tflite`
- `flutter_app/assets/models/labels.txt`

Notes:
- The `pubspec.yaml` `name:` is the Dart package name, not the launcher display name.
- Launcher display name is controlled by Android `strings.xml` / iOS `Info.plist`.

Android manifest:
- `INTERNET` permission is required and present.
- `usesCleartextTraffic=true` is set for local HTTP development.

Build (use PC LAN IP when testing over Wi‑Fi):
```powershell
cd flutter_app
flutter clean
flutter build apk --release --dart-define=BACKEND_URL=http://<PC_IP>:8000
& 'C:\Users\mommy\AppData\Local\Android\Sdk\platform-tools\adb.exe' install -r build\app\outputs\flutter-apk\app-release.apk
```

Quick dev option (no rebuild):
- Use USB + `adb reverse tcp:8000 tcp:8000` and keep app configured with `http://127.0.0.1:8000`.

---

## Testing connectivity

A. Verify backend from PC:
```powershell
curl http://127.0.0.1:8000/health
curl http://<PC_IP>:8000/health
Invoke-RestMethod -Uri http://<PC_IP>:8000/model/info
```

B. From phone (same Wi‑Fi):
- In phone browser open: `http://<PC_IP>:8000/health` and `http://<PC_IP>:8000/docs`.
- If the phone browser loads the health URL, the app can reach the backend when configured correctly.

C. Simulate upload from PC:
```powershell
curl -v -F "file=@C:\path\to\photo.jpg" -F "vendor_id=test" -F "qty=1" http://<PC_IP>:8000/predict
```

D. If phone cannot reach PC:
- Check both devices are on same SSID (not guest network).
- Verify Windows Firewall rule allows inbound TCP port 8000. Temporarily disable to debug.
- Consider using Windows Mobile Hotspot with PC as hotspot so phone connects to PC-hosted Wi‑Fi.

---

## Logs & debugging

Backend logs: terminal where `uvicorn` runs.
App/device logs:
```powershell
& 'C:\Users\mommy\AppData\Local\Android\Sdk\platform-tools\adb.exe' logcat -v time | findstr /i "E/flutter|DioException|Connection failed"
```

If you see `DioException [connection error]: SocketException: Operation not permitted, errno = 1` on Android, ensure:
- `INTERNET` permission present in `AndroidManifest.xml`.
- `usesCleartextTraffic=true` for local HTTP.
- App `BACKEND_URL` matches PC IP and uses `http://`.

---

## Release & signing

1. Generate a keystore and create `android/key.properties` (example provided).
2. Configure `android/app/build.gradle.kts` signingConfig to use that file.
3. Build signed AAB and upload to Play Console.

---

## Tests

- Flutter widget tests are under `flutter_app/test/` and do not ship with the APK.
- Run tests with:
```powershell
cd flutter_app
flutter test
```

---

## Next steps I can help with

- Overwrite the original `README.md` with this consolidated version.
- Add `deploy.md` that shows Docker and cloud VM deployment steps for the backend.
- Create a short script to archive APKs before each build.
- Generate a signed AAB if you provide keystore details (or I can show commands to create one locally).

---

If you want this file copied over `README.md`, reply "replace README" and I'll overwrite the original.
