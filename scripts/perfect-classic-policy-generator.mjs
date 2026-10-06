#!/usr/bin/env node
// Generates classic policy catalogs: the native generator writes each
// starting role's policy, the release gate's own replay checks it before it
// enters a manifest, and merge-manifests combines boards into one catalog.
// `verify` generates and replays complete small references. Kept apart from
// scripts/perfect-classic-policy.mjs, which the release gate fingerprints.
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { integerOption, parseArguments as parseCommand } from './cli-options.mjs';
import { isEntryPoint } from './entry-point.mjs';
import { buildNative, includedHeaders, parseJsonLines, runProcess } from './native-build.mjs';
import { replayPerfectClassicPolicy } from './perfect-classic-policy.mjs';
import {
  PERFECT_CLASSIC_ROLE_FIRST,
  PERFECT_CLASSIC_ROLE_SECOND,
  decodePerfectClassicPolicy,
} from '../src/perfect-classic-policy.js';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const SOURCE = join(ROOT, 'native', 'perfect-classic-policy.cpp');
const ROOT_VALUES = join(ROOT, 'data', 'perfect-classic-root-values.json');
const WIN = 1;
const DRAW = 0;

// The options each command reads. Anything else is refused rather than
// ignored: a misspelt bound used to leave its default in force.
const COMMAND_OPTIONS = Object.freeze({
  verify: [],
  generate: ['rows', 'columns', 'connect', 'role', 'handoff_remaining', 'table_bits',
    'verify_table_bits', 'maximum_nodes', 'maximum_states', 'maximum_verify_nodes',
    'expected_root', 'output'],
  'merge-manifests': ['input', 'output'],
});

export function parseArguments(argv) {
  const options = parseCommand(argv, COMMAND_OPTIONS, { defaultCommand: 'verify', repeatable: ['input'] });
  // No board is assumed. The default used to be standard 6x7, the longest
  // generation there is, for a catalog entry the release gate refuses: that
  // board plays from its own strategy (docs/PERFECT_PLAY.md).
  if (options.command === 'generate' && (options.rows === undefined || options.columns === undefined)) {
    throw new RangeError('generate requires --rows and --columns.');
  }
  return options;
}

const run = (command, args) => runProcess(command, args, { cwd: ROOT });

// The cached build the native tests share (tests/native-writers.test.mjs):
// it used to compile afresh into a temporary directory on every run.
async function compile() {
  const build = await buildNative(SOURCE, { name: 'perfect-classic-policy' });
  return { compiler: build.compiler, binary: build.binary, warnings: build.warnings };
}

async function hashFile(path) {
  const buffer = await readFile(path);
  return {
    bytes: buffer.length,
    sha256: createHash('sha256').update(buffer).digest('hex'),
  };
}

async function publishedRootValue(rows, columns, connect) {
  if (connect !== 4 || rows < 4 || columns < 4) return null;
  const reference = JSON.parse(await readFile(ROOT_VALUES, 'utf8'));
  return reference.boards.find((entry) => (
    entry.rows === rows && entry.columns === columns
  ))?.value ?? null;
}

function roleSelection(value) {
  if (value === undefined || value === 'both') {
    return [PERFECT_CLASSIC_ROLE_FIRST, PERFECT_CLASSIC_ROLE_SECOND];
  }
  const role = Number.parseInt(String(value), 10);
  if (role !== PERFECT_CLASSIC_ROLE_FIRST && role !== PERFECT_CLASSIC_ROLE_SECOND) {
    throw new RangeError('role must be 1, 2, or both.');
  }
  return [role];
}

async function generatePolicies(binary, options) {
  const rows = integerOption(options.rows, undefined, 'rows', 1, 7);
  const columns = integerOption(options.columns, undefined, 'columns', 1, 7);
  const connect = integerOption(options.connect, 4, 'connect', 1, Math.max(rows, columns));
  const cellCount = rows * columns;
  const handoffRemaining = integerOption(
    options.handoff_remaining,
    Math.min(24, cellCount),
    'handoff-remaining',
    0,
    cellCount,
  );
  const tableBits = integerOption(options.table_bits, 24, 'table-bits', 8, 27);
  const verifyTableBits = integerOption(
    options.verify_table_bits,
    22,
    'verify-table-bits',
    8,
    25,
  );
  const maximumNodes = integerOption(
    options.maximum_nodes,
    0,
    'maximum-nodes',
    0,
    Number.MAX_SAFE_INTEGER,
  );
  const maximumStates = integerOption(
    options.maximum_states,
    100_000_000,
    'maximum-states',
    1,
    Number.MAX_SAFE_INTEGER,
  );
  const maximumExactNodes = options.maximum_verify_nodes === undefined
    ? Infinity
    : integerOption(
      options.maximum_verify_nodes,
      0,
      'maximum-verify-nodes',
      1,
      Number.MAX_SAFE_INTEGER,
    );
  const output = resolve(options.output ?? join(
    ROOT,
    'generated',
    `perfect-classic-${rows}x${columns}-c${connect}`,
  ));
  await mkdir(output, { recursive: true });
  const expectedFirstValue = options.expected_root === undefined
    ? await publishedRootValue(rows, columns, connect)
    : integerOption(options.expected_root, 0, 'expected-root', -1, 1);
  const policies = [];

  for (const role of roleSelection(options.role)) {
    const filename = `${rows}x${columns}-c${connect}-role${role}.bin`;
    const path = join(output, filename);
    const result = await run(binary, [
      'generate',
      '--rows', String(rows),
      '--columns', String(columns),
      '--connect', String(connect),
      '--role', String(role),
      '--handoff-remaining', String(handoffRemaining),
      '--table-bits', String(tableBits),
      '--maximum-nodes', String(maximumNodes),
      '--maximum-states', String(maximumStates),
      '--output', path,
    ]);
    if (result.code !== 0) {
      throw new Error(`Classic policy generation failed for role ${role}.\n${result.stderr || result.stdout}`);
    }
    const summaries = parseJsonLines(result.stdout);
    const summary = summaries.at(-1);
    if (!summary || summary.format !== 'connect4-perfect-classic-policy-summary-v1') {
      throw new Error(`Classic policy generator returned no summary for role ${role}.`);
    }
    const bytes = await readFile(path);
    const policy = decodePerfectClassicPolicy(bytes, { rows, columns, connect, role });
    const expectedRoleValue = expectedFirstValue === null
      ? policy.rootValue
      : role === PERFECT_CLASSIC_ROLE_FIRST ? expectedFirstValue : -expectedFirstValue;
    if (policy.rootValue !== expectedRoleValue) {
      throw new Error(
        `Policy root value mismatch for role ${role}: `
        + `${policy.rootValue} instead of ${expectedRoleValue}.`,
      );
    }
    const replay = replayPerfectClassicPolicy(policy, {
      maximumExactNodes,
      exactTableBits: verifyTableBits,
    });
    const digest = await hashFile(path);
    policies.push({
      rows,
      columns,
      connect,
      role,
      handoffRemaining: policy.handoffRemaining,
      rootValue: policy.rootValue,
      entryCount: policy.entryCount,
      closureStates: policy.closureStates,
      file: `./${filename}`,
      ...digest,
      generator: summary,
      replay,
    });
  }

  const manifest = {
    format: 'connect4-perfect-classic-manifest-v1',
    generatedAt: new Date().toISOString(),
    sourceSha256: createHash('sha256').update(await readFile(SOURCE)).digest('hex'),
    // The exact search lives in native/classic-exact.hpp, so the source
    // alone does not identify the generator.
    headersSha256: await includedHeaders(SOURCE),
    policies,
  };
  await writeFile(join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return { output, manifest };
}

async function mergeManifests(inputPaths, output) {
  if (!Array.isArray(inputPaths) || inputPaths.length === 0) {
    throw new RangeError('At least one --input manifest is required.');
  }
  const outputPath = resolve(output);
  const outputDirectory = dirname(outputPath);
  await mkdir(outputDirectory, { recursive: true });
  const policies = [];
  const keys = new Set();
  for (const input of inputPaths) {
    const path = resolve(input);
    const manifest = JSON.parse(await readFile(path, 'utf8'));
    if (manifest?.format !== 'connect4-perfect-classic-manifest-v1'
        || !Array.isArray(manifest.policies)) {
      throw new Error(`Invalid perfect classic manifest: ${path}`);
    }
    for (const entry of manifest.policies) {
      const key = `${entry.rows}x${entry.columns}:c${entry.connect}:r${entry.role}`;
      if (keys.has(key)) throw new Error(`Duplicate perfect classic policy ${key}.`);
      keys.add(key);
      const source = resolve(dirname(path), entry.file);
      const filename = `${entry.rows}x${entry.columns}-c${entry.connect}-role${entry.role}.bin`;
      const target = join(outputDirectory, filename);
      await writeFile(target, await readFile(source));
      policies.push({ ...entry, file: `./${filename}` });
    }
  }
  policies.sort((first, second) => (
    first.rows - second.rows
    || first.columns - second.columns
    || first.connect - second.connect
    || first.role - second.role
  ));
  const manifest = {
    format: 'connect4-perfect-classic-manifest-v1',
    generatedAt: new Date().toISOString(),
    policies,
  };
  await writeFile(outputPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

async function verifySmall(binary, temporary) {
  const native = await run(binary, ['verify']);
  if (native.code !== 0) throw new Error(`Native policy verification failed.\n${native.stderr}`);
  const nativeRecords = parseJsonLines(native.stdout);
  if (nativeRecords.length !== 6) {
    throw new Error('Native policy verification returned the wrong case count.');
  }
  const generated = [];
  for (const [rows, columns, connect, expected] of [
    [2, 2, 2, WIN],
    [3, 3, 3, DRAW],
  ]) {
    const result = await generatePolicies(binary, {
      rows,
      columns,
      connect,
      handoff_remaining: 0,
      role: 'both',
      expected_root: expected,
      table_bits: 16,
      verify_table_bits: 14,
      maximum_nodes: 10_000_000,
      maximum_states: 1_000_000,
      output: join(temporary, `${rows}x${columns}-c${connect}`),
    });
    generated.push(...result.manifest.policies.map((entry) => entry.replay));
  }
  return { native: nativeRecords, replay: generated };
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.command === 'merge-manifests') {
    if (!options.output || options.output === true) throw new RangeError('--output is required.');
    const manifest = await mergeManifests(options.inputs, options.output);
    process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
    return;
  }
  const compiled = await compile();
  if (compiled.warnings) process.stderr.write(`${compiled.warnings}\n`);
  if (options.command === 'verify') {
    const temporary = await mkdtemp(join(tmpdir(), 'connect4-classic-policy-'));
    try {
      const verified = await verifySmall(compiled.binary, temporary);
      process.stdout.write(`${JSON.stringify({ compiler: compiled.compiler, ...verified }, null, 2)}\n`);
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
    return;
  }
  // generate: parseArguments refused every other command.
  const generated = await generatePolicies(compiled.binary, options);
  process.stdout.write(`${JSON.stringify({
    compiler: compiled.compiler,
    output: generated.output,
    manifest: generated.manifest,
  }, null, 2)}\n`);
}

if (isEntryPoint(import.meta.url)) await main();
