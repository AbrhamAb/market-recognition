Place your application launcher icon image here as `app_icon.png`.

Recommended:
- Provide a square PNG image (preferably 1024x1024) with a transparent background.
- Put the file at: `flutter_app/assets/icons/app_icon.png`.

To generate launcher icons after adding the image, run from the `flutter_app` folder:

```bash
flutter pub get
flutter pub run flutter_launcher_icons:main
```

This will replace Android and iOS launcher icons with your provided image. For adaptive Android icons, you can provide separate foreground/background images and update `pubspec.yaml` accordingly.