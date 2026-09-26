#!/usr/bin/env bash
# Assembles the static site GitHub Pages publishes, into _site or the
# directory named by the first argument. The Pages workflow and the CI
# browser smoke both run this, so the smoke loads exactly the files that
# deploy: a file the site fetches but this list forgets fails the smoke
# rather than returning 404 only on the live site.
set -euo pipefail

site="${1:-_site}"
# The directory is deleted and recreated, so refuse anything that is not a
# plain relative path inside the checkout.
case "$site" in
  '' | /* | *..* | *\\*) echo "Refusing to build into '$site'." >&2; exit 1 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/.."

rm -rf -- "$site"
mkdir -p \
  "$site/data/perfect-chaos-prefix/red" \
  "$site/data/perfect-chaos-prefix/yellow" \
  "$site/data/perfect-classic"
cp -R \
  index.html styles.css favicon.svg favicon.ico \
  apple-touch-icon.png manifest.json cross-origin-isolation-worker.js \
  src assets icons "$site/"

grep -q 'rel="apple-touch-icon"' "$site/index.html"
grep -q 'rel="manifest"' "$site/index.html"
test -f "$site/apple-touch-icon.png"
# Without this the page cannot become cross-origin isolated and the
# neural opponent silently drops to a single WebAssembly thread.
test -f "$site/cross-origin-isolation-worker.js"
# The network is served from R2 rather than from here, so what has
# to ship is the manifest naming it - and a Content-Security-Policy
# that allows fetching it. A policy missing that origin blocks the
# download before any request leaves the browser: no network error
# and no failed response to find, just an opponent that never
# starts. That is exactly how it broke once.
SITE="$site" node -e '
  const fs = require("node:fs");
  const site = process.env.SITE;
  const manifest = JSON.parse(fs.readFileSync(`${site}/assets/neural/model.json`, "utf8"));
  if (!manifest.origin || !manifest.object) {
    throw new Error("assets/neural/model.json names no published object");
  }
  const page = fs.readFileSync(`${site}/index.html`, "utf8");
  const policy = page.match(/Content-Security-Policy[^>]*content="([^"]+)"/)?.[1] ?? "";
  const connect = policy.match(/connect-src ([^;]+)/)?.[1] ?? "";
  if (!connect.split(/\s+/).includes(manifest.origin)) {
    throw new Error(`connect-src does not allow ${manifest.origin}: ${connect}`);
  }
'
# Link previews need a raster image: Facebook, X and LinkedIn show no SVG.
test -f "$site/assets/social-preview.png"
grep -q 'og:image" content="[^"]*social-preview\.png"' "$site/index.html"
test -f "$site/icons/connect4-chaos-180.png"
test -f "$site/icons/connect4-chaos-192.png"
test -f "$site/icons/connect4-chaos-512.png"
test -f "$site/icons/connect4-chaos-512-maskable.png"

cp data/perfect-chaos-prefix/red/*.policy.bin "$site/data/perfect-chaos-prefix/red/"
cp data/perfect-chaos-prefix/yellow/*.policy.bin "$site/data/perfect-chaos-prefix/yellow/"
prefix_count="$(node -e \
  'const m=require("./data/perfect-chaos-prefix/manifest.json");
   process.stdout.write(String(JSON.stringify(m).match(/\.policy\.bin/g).length))')"
test "$(find "$site/data/perfect-chaos-prefix" -name '*.policy.bin' | wc -l)" = "$prefix_count"

mkdir -p "$site/data/perfect-chaos-complete"
cp data/perfect-chaos-complete/manifest.json data/perfect-chaos-complete/*.bin \
  "$site/data/perfect-chaos-complete/"
test "$(ls "$site/data/perfect-chaos-complete" | wc -l)" \
  = "$(ls data/perfect-chaos-complete | wc -l)"

cp data/perfect-classic/manifest.json "$site/data/perfect-classic/manifest.json"
policy_count="$(node -e \
  'const m=require("./data/perfect-classic/manifest.json"); process.stdout.write(String(m.policies?.length ?? 0))')"
if (( policy_count > 0 )); then
  cp data/perfect-classic/*.bin "$site/data/perfect-classic/"
  test "$(find "$site/data/perfect-classic" -name '*.bin' | wc -l)" = "$policy_count"
fi
touch "$site/.nojekyll"
if find "$site" -type l -print -quit | grep -q .; then
  echo 'Refusing to publish symlinks in the Pages artifact.' >&2
  exit 1
fi
if find "$site" -name '.*' ! -path "$site/.nojekyll" -print -quit | grep -q .; then
  echo 'Refusing to publish unexpected hidden files in the Pages artifact.' >&2
  exit 1
fi
