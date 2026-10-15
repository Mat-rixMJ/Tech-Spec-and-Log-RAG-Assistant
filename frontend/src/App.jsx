import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api, API_BASE } from './api.js'
import { pct, riskClasses, summarize } from './format.js'
import RetentionDrawer from './RetentionDrawer.jsx'

function Stat({ label, value, tone = 'ink' }) {
  const color =
    tone === 'red' ? 'text-red-600' : tone === 'amber' ? 'text-amber-600' : 'text-ink'
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      <p className="mt-0.5 text-sm text-slate-500">{label}</p>
    </div>
  )
}

export default function App() {
  const [customers, setCustomers] = useState([])
  const [summary, setSummary] = useState(null)
  const [scoring, setScoring] = useState(false)
  const [error, setError] = useState('')

  const [doc, setDoc] = useState(null) // { doc_id, name, num_chunks }
  const [docBusy, setDocBusy] = useState(false)
  const [selected, setSelected] = useState(null) // customer in the drawer

  async function onCustomerFile(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setError('')
    setScoring(true)
    try {
      const res = await api.scoreCustomers(file)
      setCustomers(res.customers)
      setSummary({ ...summarize(res.customers), count: res.count, atRisk: res.at_risk })
    } catch (err) {
      setError(err.message)
      setCustomers([])
      setSummary(null)
    } finally {
      setScoring(false)
      e.target.value = ''
    }
  }

  async function onDocFile(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setDocBusy(true)
    try {
      setDoc(await api.uploadDoc(file))
    } catch (err) {
      setError(err.message)
    } finally {
      setDocBusy(false)
      e.target.value = ''
    }
  }

  const chartData = summary
    ? [
        { level: 'High', count: summary.high || 0, fill: '#dc2626' },
        { level: 'Medium', count: summary.medium || 0, fill: '#d97706' },
        { level: 'Low', count: summary.low || 0, fill: '#16a34a' },
      ]
    : []

  return (
    <div className="min-h-screen bg-slate-50 text-ink">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-6 py-6">
          <h1 className="text-2xl font-bold">
            Cust<span className="bg-gradient-to-r from-accent to-accent2 bg-clip-text text-transparent">IQ</span>
          </h1>
          <p className="mt-1 text-slate-600">
            Upload customers to predict churn, upload your policy docs, and draft
            AI retention emails grounded in them.
          </p>
          <p className="mt-1 text-xs text-slate-400">
            The API sleeps when idle, so the first request may take ~30s to wake.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-10 px-6 py-8">
        {error && (
          <p role="alert" className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </p>
        )}

        {/* Step 1 — customers */}
        <section>
          <h2 className="text-lg font-semibold">1 · Score your customers</h2>
          <p className="mt-1 text-sm text-slate-600">
            Upload a CSV of customers.{' '}
            <a
              href={`${API_BASE}/sample-customers.csv`}
              className="font-medium text-accent hover:underline"
            >
              Download a sample CSV
            </a>{' '}
            to try it.
          </p>

          <label className="mt-4 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-white px-6 py-8 text-center transition-colors hover:border-accent/50">
            <span className="text-sm font-medium text-ink">
              {scoring ? 'Scoring…' : 'Choose a customer CSV'}
            </span>
            <span className="mt-1 text-xs text-slate-400">.csv up to 5 MB</span>
            <input type="file" accept=".csv" className="sr-only" onChange={onCustomerFile} />
          </label>
        </section>

        {summary && (
          <section className="space-y-6">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Stat label="Customers" value={summary.count} />
              <Stat label="At risk" value={summary.atRisk} tone="amber" />
              <Stat label="High risk" value={summary.high || 0} tone="red" />
              <Stat label="Low risk" value={summary.low || 0} />
            </div>

            <div
              className="h-56 rounded-xl border border-slate-200 bg-white p-4"
              aria-label="Customers by risk level"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="level" tick={{ fontSize: 12 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {chartData.map((d) => (
                      <Cell key={d.level} fill={d.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}

        {/* Step 2 — policy docs (optional, powers grounded emails) */}
        {customers.length > 0 && (
          <section>
            <h2 className="text-lg font-semibold">2 · Add your policy docs (optional)</h2>
            <p className="mt-1 text-sm text-slate-600">
              Upload a retention / pricing policy (PDF or TXT). Retention emails will
              only offer what your policy actually allows.
            </p>
            <label className="mt-4 inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium hover:border-accent/50">
              {docBusy ? 'Uploading…' : doc ? `Loaded: ${doc.name}` : 'Upload policy doc'}
              <input type="file" accept=".pdf,.txt" className="sr-only" onChange={onDocFile} />
            </label>
            {doc && (
              <span className="ml-3 text-sm text-green-700">
                {doc.num_chunks} chunks indexed — emails will be grounded.
              </span>
            )}
          </section>
        )}

        {/* Step 3 — customer table */}
        {customers.length > 0 && (
          <section>
            <h2 className="text-lg font-semibold">3 · Customers by churn risk</h2>
            <p className="mt-1 text-sm text-slate-600">
              Sorted highest-risk first. Open any customer to draft a retention email.
            </p>
            <div className="mt-4 overflow-x-auto rounded-xl border border-slate-200 bg-white">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-slate-200 text-slate-500">
                  <tr>
                    <th className="px-4 py-3">Customer</th>
                    <th className="px-4 py-3">Plan</th>
                    <th className="px-4 py-3">Risk</th>
                    <th className="px-4 py-3">Top reason</th>
                    <th className="px-4 py-3" />
                  </tr>
                </thead>
                <tbody>
                  {customers.map((c) => (
                    <tr key={c.customer_id} className="border-b border-slate-100 last:border-0">
                      <td className="px-4 py-3 font-medium">
                        {c.name || c.customer_id}
                      </td>
                      <td className="px-4 py-3 text-slate-600">{c.contract_label}</td>
                      <td className="px-4 py-3">
                        <span
                          className={`rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ring-1 ${riskClasses(
                            c.risk_level,
                          )}`}
                        >
                          {c.risk_level} · {pct(c.churn_probability)}
                        </span>
                      </td>
                      <td className="px-4 py-3 capitalize text-slate-600">
                        {c.top_factors[0]?.feature}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          type="button"
                          onClick={() => setSelected(c)}
                          className="rounded-lg border border-slate-300 px-3 py-1 text-xs font-medium hover:border-accent/50 hover:text-accent"
                        >
                          Retention plan
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>

      {selected && (
        <RetentionDrawer
          customer={selected}
          docId={doc?.doc_id}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  )
}
