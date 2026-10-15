import { test } from 'node:test'
import assert from 'node:assert/strict'
import { pct, riskClasses, summarize } from '../src/format.js'

test('pct rounds a 0..1 probability to a whole percent', () => {
  assert.equal(pct(0.826), '83%')
  assert.equal(pct(0), '0%')
  assert.equal(pct(1), '100%')
})

test('riskClasses maps each level to a distinct colour set', () => {
  assert.match(riskClasses('high'), /red/)
  assert.match(riskClasses('medium'), /amber/)
  assert.match(riskClasses('low'), /green/)
  // Unknown levels fall back to the low (green) style, never undefined.
  assert.match(riskClasses('???'), /green/)
})

test('summarize counts customers per risk level', () => {
  const customers = [
    { risk_level: 'high' },
    { risk_level: 'high' },
    { risk_level: 'medium' },
    { risk_level: 'low' },
  ]
  assert.deepEqual(summarize(customers), { high: 2, medium: 1, low: 1 })
})
