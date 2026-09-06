import { constants } from 'node:fs';
import fs from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, isAbsolute, relative, resolve, sep } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

const defaultRoot = fileURLToPath(new URL('..', import.meta.url));
const rootFiles = new Set([
  'index.html', 'styles.css', 'favicon.svg', 'favicon.ico',
  'apple-touch-icon.png', 'manifest.json',
]);
const publicDirectories = new Set(['src', 'assets', 'icons']);
const publicCatalogs = new Set([
  'perfect-classic', 'perfect-chaos-prefix', 'perfect-chaos-complete',
]);
const mimeTypes = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.png', 'image/png'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.webp', 'image/webp'],
  ['.ico', 'image/x-icon'],
  ['.bin', 'application/octet-stream'],
  ['.onnx', 'application/octet-stream'],
  ['.wasm', 'application/wasm'],
]);

function isPublicPath(path) {
  if (isAbsolute(path)) return false;
  const parts = path.split('/');
  if (parts.some((part) => !part || part.startsWith('.') || part.includes('\\')
      || part.includes(':') || part.endsWith('.') || part.endsWith(' '))) return false;
  if (rootFiles.has(path)) return true;
  if (!mimeTypes.has(extname(path))) return false;
  return (parts.length > 1 && publicDirectories.has(parts[0]))
    || (parts.length > 2 && parts[0] === 'data' && publicCatalogs.has(parts[1]));
}

function sendError(response, status, message) {
  if (response.destroyed || response.writableEnded) return;
  if (response.headersSent) {
    response.destroy();
    return;
  }
  response.writeHead(status, {
    'cache-control': 'no-store',
    'content-type': 'text/plain; charset=utf-8',
    'x-content-type-options': 'nosniff',
  });
  response.end(message);
}

async function serve(request, response, root) {
  let file;
  try {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.setHeader('allow', 'GET, HEAD');
      sendError(response, 405, 'Method not allowed');
      return;
    }
    // Do not let arbitrary DNS names expose a loopback development server.
    const host = new URL(`http://${request.headers.host ?? ''}`);
    if (!['localhost', '127.0.0.1', '[::1]'].includes(host.hostname)
        || host.username || host.password) {
      sendError(response, 403, 'Forbidden');
      return;
    }
    const rawPath = (request.url ?? '/').split('?', 1)[0];
    const pathname = decodeURIComponent(rawPath);
    if (!pathname.startsWith('/') || pathname.startsWith('//')
        || /[\u0000-\u001f\u007f\\]/u.test(pathname)) {
      sendError(response, 400, 'Bad request');
      return;
    }
    const publicPath = pathname === '/' ? 'index.html' : pathname.slice(1);
    if (!isPublicPath(publicPath)) {
      sendError(response, 404, 'Not found');
      return;
    }
    const candidate = resolve(root, publicPath);
    const canonical = await fs.realpath(candidate);
    const canonicalRelative = relative(root, canonical).split(sep).join('/');
    // Reapply the allowlist to the target: even an in-root symlink must not
    // expose .env, .git, tooling, private directories or unapproved file types.
    if (!isPublicPath(canonicalRelative)) {
      sendError(response, 404, 'Not found');
      return;
    }
    file = await fs.open(canonical, constants.O_RDONLY
      | (constants.O_NOFOLLOW ?? 0) | (constants.O_NONBLOCK ?? 0));
    const stat = await file.stat();
    if (!stat.isFile()) {
      sendError(response, 404, 'Not found');
      return;
    }
    response.writeHead(200, {
      'cache-control': 'no-store',
      'content-type': mimeTypes.get(extname(canonical)),
      'content-length': stat.size,
      'x-content-type-options': 'nosniff',
    });
    if (request.method === 'HEAD') response.end();
    else await pipeline(file.createReadStream({ autoClose: false }), response);
  } catch (error) {
    if (error instanceof URIError || error.code === 'ERR_INVALID_URL') {
      sendError(response, 400, 'Bad request');
    } else if (['ENOENT', 'ENOTDIR', 'ELOOP'].includes(error.code)) {
      sendError(response, 404, 'Not found');
    } else if (['EACCES', 'EPERM'].includes(error.code)) {
      sendError(response, 403, 'Forbidden');
    } else {
      sendError(response, 500, 'Internal server error');
    }
  } finally {
    // Also release the descriptor after a HEAD request, read error or disconnect.
    if (file) await file.close().catch(() => {});
  }
}

export async function createStaticServer(directory = defaultRoot) {
  const root = await fs.realpath(directory);
  return createServer((request, response) => {
    void serve(request, response, root);
  });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const value = process.env.PORT ?? '4173';
    const port = Number(value);
    if (!/^\d+$/u.test(value) || !Number.isInteger(port) || port < 0 || port > 65535) {
      throw new RangeError('PORT must be an integer from 0 through 65535');
    }
    const server = await createStaticServer();
    server.on('error', (error) => {
      console.error(`Could not start Connect 4: ${error.message}`);
      process.exitCode = 1;
    });
    server.listen(port, '127.0.0.1', () => {
      console.log(`Connect 4 is running at http://127.0.0.1:${server.address().port}`);
    });
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
