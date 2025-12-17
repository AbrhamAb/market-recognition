# Flutter mobile client

A lightweight Flutter app that talks to the FastAPI backend (`/predict`). Pick an image, set vendor/qty/buy price, and see predictions and pricing hints.

## Requirements
- Flutter SDK (3.3+ recommended)
- Android Studio / Xcode for platform toolchains

## Setup
```bash
cd flutter_app
flutter pub get
```

## Run
- Android emulator: backend at `http://10.0.2.2:8000` (default). Start:
  ```bash
  flutter run --dart-define=BACKEND_URL=http://10.0.2.2:8000
  ```
- iOS simulator or device: point to reachable backend host/IP:
  ```bash
  flutter run --dart-define=BACKEND_URL=http://<your-host>:8000
  ```

## Notes
- Permissions: add camera/photos permissions in AndroidManifest.xml / Info.plist if you enable camera capture (image_picker handles most defaults).
- Backend must be running and reachable over the given URL.
- No auth is implemented; add API keys/tokens if exposing publicly.
- For offline/on-device inference, convert the Keras model to TFLite and integrate with `tflite_flutter` (not included here).
