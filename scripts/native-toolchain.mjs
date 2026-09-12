/** Native GCC/Clang linker flags shared by solver scripts and their tests.
 * MinGW must not pick up an unrelated libstdc++ DLL from Git for Windows.
 * Darwin needs its dynamic system runtime; Linux does not need the workaround.
 * These entry points build for the host, using GCC-style compiler drivers.
 */
export function nativeLinkFlags(platform = process.platform) {
  if (typeof platform !== 'string' || !platform) throw new TypeError('A host platform is required.');
  return platform === 'win32' ? ['-static'] : [];
}
