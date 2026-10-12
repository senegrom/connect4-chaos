/** The check on each option's value that parseOptions does not make.
 *
 * parseOptions cannot tell a flag from an option that takes a value: it reads
 * `--output` with nothing after it as true, takes the word after `--verbose`
 * as its value, and keeps the '' a shell passes for `--output "$UNSET"`.
 * Scripts carried on with any of them: a strategy build wrote the strategy to
 * a file named true, a book pack wrote the book over the committed one before
 * failing on a bare --manifest, and the prefix generator resolved '' to the
 * working directory and set about deleting it.
 *
 * It belongs in parseOptions, but the release gate's replay loads
 * cli-options.mjs and an edit there costs a replay: move it there with the
 * next change to that file.
 */

/** Returns `options` once every option in it but the named `flags` has a
 * value that is not empty, and no flag has one. */
export function requireValues(options, flags = []) {
  for (const [name, value] of Object.entries(options)) {
    const option = `--${name.replaceAll('_', '-')}`;
    if (flags.includes(name)) {
      if (value !== true) throw new RangeError(`${option} takes no value, but was given ${JSON.stringify(value)}.`);
    } else if (value === true || value === '') {
      throw new RangeError(`${option} requires a value.`);
    }
  }
  return options;
}
