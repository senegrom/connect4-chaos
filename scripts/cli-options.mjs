/** Command-line parsing for the proof scripts.
 *
 * Options are spelt with underscores for the hyphens of the command line
 * (--shard-count is shard_count). Anything a command does not read is refused
 * rather than ignored: a misspelt option used to leave its default in force
 * and exit 0 - `solve --colums 5` solved seven columns, and `pack --ouput
 * new.bin` would have overwritten the committed book.
 *
 * The release gate's replay loads this module, so it is in that gate's
 * fingerprint: an edit here costs one replay.
 */

/** The `--name [value]` options in `argv`, refusing any not in `known`;
 * `owner` names the command or script in that refusal. An option without a
 * value is true; one named in `repeatable` may be given again and collects
 * into an array under its name plus "s". */
export function parseOptions(argv, known, owner, { repeatable = [] } = {}) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith('--')) throw new RangeError(`Unexpected argument: ${argument}`);
    const name = argument.slice(2).replaceAll('-', '_');
    if (!known.includes(name)) throw new RangeError(`${owner} has no option ${argument}.`);
    const value = argv[index + 1];
    const given = value !== undefined && !value.startsWith('--');
    if (repeatable.includes(name)) {
      if (!given) throw new RangeError(`${argument} requires a value.`);
      (options[`${name}s`] ??= []).push(value);
      index += 1;
    } else if (!given) options[name] = true;
    else {
      options[name] = value;
      index += 1;
    }
  }
  return options;
}

/** { command, ...options } for `argv`, whose first word names one of
 * `commands` (each mapped to the options it reads). An unknown command
 * throws `usage`, or says it is unknown. */
export function parseArguments(argv, commands, { defaultCommand, repeatable = [], usage } = {}) {
  const command = argv[0] ?? defaultCommand;
  const known = Object.hasOwn(commands, command ?? '') ? commands[command] : null;
  if (!known) throw new RangeError(usage ?? `Unknown command: ${command}`);
  return { command, ...parseOptions(argv.slice(1), known, command, { repeatable }) };
}

/** An integer option in [minimum, maximum], or `fallback` when not given. */
export function integerOption(value, fallback, label, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
  const selected = value === undefined ? fallback : Number.parseInt(String(value), 10);
  if (!Number.isInteger(selected) || selected < minimum || selected > maximum) {
    const range = maximum === Number.MAX_SAFE_INTEGER ? `of at least ${minimum}` : `from ${minimum} through ${maximum}`;
    throw new RangeError(`${label} must be an integer ${range}.`);
  }
  return selected;
}

/** Reports a script's failure on stderr and exits 1 once it unwinds. */
export function fail(error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
