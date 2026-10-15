import { useEffect, useState } from 'react'
import { api } from './api.js'
import { pct, riskClasses } from './format.js'

// Slide-over panel: shows one customer's risk + factors and drafts a
// policy-grounded retention email on demand.
export default function RetentionDrawer({ customer, docId, onClose }) {
  const [email, setEmail] = useState(null)
  const [sources, setSources] = useState([])
  const [demo, setDemo] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // Close on Escape for keyboard users.
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  async function draft() {
    setBusy(true)
    setError('')
    try {
      const res = await api.retentionEmail(customer, docId || null)
      setEmail(res.email)
      setSources(res.grounded_in)
      setDemo(!res.llm_live)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex">
      <button
        type="button"
        aria-label="Close panel"
        onClick={onClose}
        className="flex-1 bg-ink/30 backdrop-blur-sm"
      />
      <aside
        role="dialog"
        aria-label={`Retention plan for ${customer.name || customer.customer_id}`}
        className="flex w-full max-w-md flex-col overflow-y-auto border-l border-slate-200 bg-white p-6 shadow-2xl"
      >
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-ink">
              {customer.name || customer.customer_id}
            </h2>
            <p className="text-sm text-slate-500">{customer.contract_label} plan</p>
          </div>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ring-1 ${riskClasses(
              customer.risk_level,
            )}`}
          >
            {customer.risk_level} · {pct(customer.churn_probability)}
          </span>
        </div>

        <h3 className="mt-6 text-sm font-semibold text-ink">Why they're at risk</h3>
        <ul className="mt-2 space-y-1.5">
          {customer.top_factors.map((f) => (
            <li key={f.feature} className="flex items-center gap-2 text-sm">
              <span
                aria-hidden="true"
                className={`h-1.5 w-1.5 rounded-full ${
                  f.direction === 'increases risk' ? 'bg-red-500' : 'bg-green-500'
                }`}
              />
              <span className="capitalize text-slate-700">{f.feature}</span>
              <span className="text-slate-400">— {f.direction}</span>
            </li>
          ))}
        </ul>

        <h3 className="mt-6 text-sm font-semibold text-ink">AI retention email</h3>
        {!email && (
          <button
            type="button"
            onClick={draft}
            disabled={busy}
            className="mt-2 inline-flex items-center justify-center rounded-lg bg-gradient-to-r from-accent to-accent2 px-4 py-2 text-sm font-medium text-white shadow-sm transition-all hover:-translate-y-0.5 disabled:opacity-60"
          >
            {busy ? 'Drafting…' : 'Draft a grounded email'}
          </button>
        )}
        {error && (
          <p role="alert" className="mt-2 text-sm text-red-600">
            {error}
          </p>
        )}

        {email && (
          <>
            {demo && (
              <p className="mt-2 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-700">
                Demo mode: no LLM key configured, showing the retrieved policy context
                instead of a generated email.
              </p>
            )}
            <textarea
              readOnly
              aria-label="Drafted retention email"
              className="mt-2 h-56 w-full resize-none rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700"
              value={email}
            />
            <div className="mt-2 flex gap-2">
              <button
                type="button"
                onClick={() => navigator.clipboard?.writeText(email)}
                className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100"
              >
                Copy
              </button>
              <button
                type="button"
                onClick={draft}
                disabled={busy}
                className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 disabled:opacity-60"
              >
                Regenerate
              </button>
            </div>

            {sources.length > 0 && (
              <details className="mt-3 text-xs text-slate-500">
                <summary className="cursor-pointer">
                  Grounded in {sources.length} policy passage
                  {sources.length > 1 ? 's' : ''}
                </summary>
                <ul className="mt-1 space-y-1">
                  {sources.map((s, i) => (
                    <li key={i} className="rounded bg-slate-50 p-2 text-slate-600">
                      {s.text.slice(0, 200)}
                      {s.text.length > 200 ? '…' : ''}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </>
        )}

        <button
          type="button"
          onClick={onClose}
          className="mt-auto pt-6 text-sm text-slate-500 hover:text-ink"
        >
          Close
        </button>
      </aside>
    </div>
  )
}
