/** Live neural stages have no depth/node/elapsed fields yet. */
export function neuralSearchInfo(search) {
  if (search?.solver === 'neural-loading') {
    return search.note || 'Loading the neural opponent…';
  }
  if (search?.solver === 'neural-searching') {
    const fraction = search.fraction;
    const percent = Number.isFinite(fraction)
      ? ` · ${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%` : '';
    return `${search.note || 'Neural search in progress…'}${percent}`;
  }
  return null;
}
