#!/bin/zsh
# Build WebDriverAgent onto the plugged-in iPhone and leave it running, then
# forward it to localhost:8100. Keep this terminal open during play.
#
#   ios/build_wda.sh <TEAM_ID>          # TEAM_ID: Xcode > Settings > Accounts
#
# First run only: after the build installs, the phone refuses to launch it
# until you trust it: Settings > General > VPN & Device Management > your
# Apple ID > Trust. Then run this again.
set -euo pipefail
cd "${0:A:h}"
TEAM="${1:?usage: ios/build_wda.sh <TEAM_ID>  (Xcode > Settings > Accounts > your team)}"

[[ -d /Applications/Xcode.app ]] || { echo "Xcode not installed (App Store)"; exit 1; }
[[ "$(xcode-select -p)" == /Applications/Xcode.app/* ]] || {
  echo "run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"; exit 1; }
[[ -d WebDriverAgent ]] || git clone --depth 1 https://github.com/appium/WebDriverAgent.git

UDID="$(idevice_id -l | head -1)"
[[ -n "$UDID" ]] || { echo "no iPhone over USB: plug it in, unlock, tap Trust"; exit 1; }
echo "iPhone $UDID ($(ideviceinfo -u "$UDID" -k ProductVersion 2>/dev/null || echo '?'))"

# A free Apple ID needs a bundle id no one else has registered.
BUNDLE="com.${USER}.cr.WebDriverAgentRunner"

pkill -f "iproxy 8100" 2>/dev/null || true
iproxy 8100 8100 -u "$UDID" >/dev/null 2>&1 &
IPROXY=$!
trap 'kill $IPROXY 2>/dev/null' EXIT

echo "building and launching WDA (first build takes a few minutes; log: ios/wda-build.log)"
echo "when 'ServerURLHere' appears it is up: in another terminal, curl localhost:8100/status"
xcodebuild -project WebDriverAgent/WebDriverAgent.xcodeproj \
  -scheme WebDriverAgentRunner -destination "id=$UDID" \
  -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
  DEVELOPMENT_TEAM="$TEAM" CODE_SIGN_STYLE=Automatic \
  PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE" \
  test 2>&1 | tee wda-build.log | grep --line-buffered -E "ServerURLHere|error:|Testing failed|BUILD|not trusted|Developer Mode"
