# DeFooocus Android Browser (Kotlin)

This is a starter Android browser app focused on opening DeFooocus Colab in desktop-like mode and keeping sessions alive as much as Android allows.

## Included features

- Kotlin Android app scaffold (Android Studio / Gradle Kotlin DSL).
- Default start URL is the DeFooocus Colab notebook.
- Desktop-like user-agent forced for every tab.
- Multiple tabs (`ViewPager2` + `TabLayout`) like a lightweight Chrome flow.
- File upload support from web pages (`<input type=file>`).
- File download support using Android `DownloadManager`.
- Cookies + DOM storage enabled to preserve login/session data.
- Foreground service with persistent notification to improve process survival.
- Boot receiver to restart keep-alive service after reboot.

## Important Android platform limits (cannot be bypassed by normal apps)

The following requests are **not fully possible** for regular Play Store apps:

- "Never kill my app"
- "Disable force stop"
- "Disable clear data / clear cache"
- "Unrestricted battery always on every device automatically"

Android keeps user/device owner control over these behaviors. The best practical approach is:

1. Run foreground service (`START_STICKY`) ✅
2. Ask user to disable battery optimization for the app ✅
3. For enterprise-managed devices only: use Device Owner / kiosk policy via EMM/MDM ✅

## Build

Open `android-browser-app/` in Android Studio and run the `app` module.

## Next recommended steps

- Add proper tab model persistence (Room/DataStore) for tab restore.
- Add explicit in-app settings screen for battery optimization guidance.
- Add incognito mode and manual logout/clear-data button.
- Add full Chrome-like tab management UI (close/reorder/tab previews).
- Add download permission handling for Android 13+ notification behavior.
