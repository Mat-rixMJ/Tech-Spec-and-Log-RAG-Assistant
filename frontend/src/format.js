// Pure helpers, kept out of components so they're unit-testable without a DOM.

export function pct(p) {
  return `${Math.round(p * 100)}%`
}

export function riskClasses(level) {
  switch (level) {
    case 'high':
      return 'bg-red-50 text-red-700 ring-red-200'
    case 'medium':
      return 'bg-amber-50 text-amber-700 ring-amber-200'
    default:
      return 'bg-green-50 text-green-700 ring-green-200'
  }
}

// Risk order for sorting / summarizing (high first).
export const RISK_RANK = { high: 0, medium: 1, low: 2 }

export function summarize(customers) {
  const by = { high: 0, medium: 0, low: 0 }
  for (const c of customers) by[c.risk_level] = (by[c.risk_level] || 0) + 1
  return by
}
