// CustIQ API client. All fetch logic lives here (per the frontend skill).

const BASE =
  (import.meta.env && import.meta.env.VITE_API_BASE) || 'http://127.0.0.1:8300'

export const API_BASE = BASE

async function parse(res) {
  const text = await res.text()
  const data = text ? JSON.parse(text) : null
  if (!res.ok) {
    let msg = `HTTP ${res.status}`
    if (data && data.detail) {
      msg =
        typeof data.detail === 'string'
          ? data.detail
          : data.detail.map((d) => d.msg).join('; ')
    }
    throw new Error(msg)
  }
  return data
}

function postFile(path, file) {
  const form = new FormData()
  form.append('file', file)
  return fetch(`${BASE}${path}`, { method: 'POST', body: form }).then(parse)
}

export const api = {
  scoreCustomers: (file) => postFile('/score', file),
  uploadDoc: (file) => postFile('/docs-upload', file),
  importances: () => fetch(`${BASE}/churn/importances`).then(parse),
  ask: (doc_id, question) =>
    fetch(`${BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_id, question }),
    }).then(parse),
  retentionEmail: (customer, doc_id) =>
    fetch(`${BASE}/retention-email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ customer, doc_id }),
    }).then(parse),
}
