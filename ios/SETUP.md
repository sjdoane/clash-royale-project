# Running the bot on an iPhone from the Mac

## Route A (recommended): iPhone Mirroring, no Xcode

macOS shows the iPhone in its built-in **iPhone Mirroring** app. The bot
captures that window and clicks inside it, and Mirroring turns the clicks into
touches. Other open-source Clash Royale bots use the same route
([jev-royal](https://github.com/Amrit-Nigam/jev-royal), a Merge Tactics bot).
It needs macOS 15+, iOS 18+, and the same Apple ID on both. Unlike WDA, it
needs no Xcode, no signing and no 7-day expiry.

1. Open **iPhone Mirroring** on the Mac and follow its pairing steps. The phone
   must be locked and near the Mac; approve on the phone when asked.
2. Give the terminal you run the bot from **Screen Recording** and
   **Accessibility** (System Settings › Privacy & Security). Quit and reopen
   the terminal afterwards.
3. Open Clash Royale in the Mirroring window, start a **Training Camp**
   battle, and calibrate:
   ```
   cd engine/perception
   ../../.venv/bin/python tools/calibrate_ios.py --mirror --out ../../ios/geometry.json
   ```
   Check that `ios/geometry.frame.png` is an ordinary full battle frame.
   Recalibrate if you resize the Mirroring window.
4. Play:
   ```
   .venv/bin/python play_live.py            # dry run
   .venv/bin/python play_live.py --act      # plays; confirm the opponent first
   ```
   While it plays, the bot owns the mouse: don't touch the mouse or trackpad,
   and keep the Mirroring window on screen; the bot brings it to the front
   before each tap.

## Route B: WebDriverAgent over USB (needs Xcode)

Use this if Mirroring is unavailable. Run play_live.py with `--device iphone`.

Clash Royale runs on the phone as normal. The Mac reads the screen and taps
through **WebDriverAgent** (WDA), Apple's XCUITest UI-testing server, the one
Appium uses. You build it onto your own phone with your own Apple ID, and it
reaches the Mac over the USB cable. Nothing on the phone is jailbroken or
modified, and nothing touches the game.

Only Training Camp, or a Friendly Battle against a teammate who agreed.
Never ladder or Classic/Grand Challenges. See LIVE.md.

## One-time setup (~30 min, mostly the Xcode download)

1. **Install Xcode** from the Mac App Store (free, ~10 GB). Then:
   ```
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   xcodebuild -runFirstLaunch
   ```
2. **Add your Apple ID to Xcode.** Go to Xcode › Settings › Accounts and click +.
   A free account works: it becomes a "Personal Team". Copy its **Team ID**,
   a 10-character code. You can find it under Manage Certificates, or run
   `security find-identity -v -p codesigning` after step 4 and read the code
   in parentheses.
3. **Prepare the iPhone.**
   - Plug it in with a cable, unlock it, and tap **Trust** on the prompt.
   - Turn on Developer Mode: Settings › Privacy & Security › **Developer Mode** › On.
     The phone restarts. If the option is missing, open Xcode once with the
     phone plugged in, under Window › Devices and Simulators, and it appears.
4. **Build WDA onto the phone:**
   ```
   ios/build_wda.sh <TEAM_ID>
   ```
   The first time, the app installs and then fails to launch with "not
   trusted". On the phone, go to Settings › General › **VPN & Device Management**,
   tap your Apple ID, then tap **Trust**. Run the script again.
   When it prints `ServerURLHere`, WDA is up. **Leave that terminal open.**
   A free-account build expires after 7 days; rerunning the script renews it.

Check it from a second terminal:
```
curl -s localhost:8100/status | head -c 300
```

## Calibrate once per phone (2 min)

The iPhone screen is taller than the 9:16 layout the detector was trained on.
The calibration finds the game frame inside the screenshot.

1. On the phone, start a **Training Camp** battle. Leave all towers standing
   and keep your hand on screen.
2. Run:
   ```
   cd engine/perception
   ../../.venv/bin/python tools/calibrate_ios.py --out ../../ios/geometry.json
   ```
3. Open `ios/geometry.frame.png`. It must look like an ordinary full battle:
   the arena on top and the four cards plus elixir bar at the bottom, nothing
   cut off. `ios/geometry.overlay.png` shows where the frame was found.

If it says the match is weak, try again mid-battle. Recalibrate if you change
phones or turn on Display Zoom.

## Every session

Terminal 1 (leave open):
```
ios/build_wda.sh <TEAM_ID>
```
Terminal 2: start a Training Camp battle on the phone, then:
```
.venv/bin/python play_live.py                # dry run: watches and decides, no taps
.venv/bin/python play_live.py --act          # plays; asks you to confirm the opponent first
```

## Notes

- **Speed.** Each frame is one WDA screenshot, roughly 3-8 per second
  depending on the phone, against about 10 on BlueStacks. The loop prints its
  frame period at the end of every run (`producer period`). If decisions lag,
  say so: WDA also has an MJPEG stream on port 9100 that is faster.
- **Taps** go through the same crop as the frames, so a tap lands on the
  pixel the detector saw. Each placement is one WDA request: tap the card,
  then tap the cell.
- **Do not touch the phone** while the bot plays. WDA taps fight your finger.
