import { check } from 'k6';
import http from 'k6/http';

const baseURL = (__ENV.K6_BASE_URL || 'http://application-api:8000').replace(/\/+$/, '');
const duration = __ENV.K6_TEST_DURATION || '30s';
const runID = __ENV.K6_EXPERIMENT_RUN_ID || '';
const scenarioName = __ENV.K6_SCENARIO || 'baseline';
const maxDelayMs = Number(__ENV.K6_MAX_EXPERIMENT_DELAY_MS || '2000');

if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(runID)) {
  throw new Error('K6_EXPERIMENT_RUN_ID (testid) is required and must be a neutral UUID v4.');
}
if (!['baseline', 'stress_cpu', 'recovery'].includes(scenarioName)) {
  throw new Error('K6_SCENARIO must be baseline, stress_cpu or recovery.');
}
if (!Number.isInteger(maxDelayMs) || maxDelayMs < 0) {
  throw new Error('K6_MAX_EXPERIMENT_DELAY_MS must be a non-negative integer.');
}
if (!/^https?:\/\/[^\s/?#@]+$/.test(baseURL)) {
  throw new Error('K6_BASE_URL must be an HTTP(S) origin without credentials, path or query.');
}
if (!/^[1-9]\d*(s|m|h)$/.test(duration)) {
  throw new Error('K6_TEST_DURATION must be a positive integer followed by s, m or h.');
}

const durationSeconds = Number(duration.slice(0, -1)) * { s: 1, m: 60, h: 3600 }[duration.slice(-1)];

function scenario(exec, rate, endpoint) {
  return {
    executor: 'constant-arrival-rate',
    exec,
    rate,
    timeUnit: '1s',
    duration,
    // Delay + dependency budget + scheduling margin (25/10/5 at 2000 ms).
    preAllocatedVUs: Math.ceil(rate * (maxDelayMs / 1000 + 3)),
    maxVUs: Math.ceil(rate * (maxDelayMs / 1000 + 5)),
    gracefulStop: '5s',
    tags: { endpoint },
  };
}

export const options = {
  scenarios: {
    products: scenario('products', 5, '/products'),
    orders: scenario('orders', 2, '/orders'),
    health: scenario('health', 1, '/health'),
  },
  systemTags: ['method', 'status', 'check', 'expected_response'],
  // testid cardinality grows only with the controlled number of experiments.
  // Request IDs, full URLs, names and free user input must never become tags.
  tags: { scenario: scenarioName, testid: runID },
  discardResponseBodies: true,
  summaryTrendStats: ['avg', 'p(50)', 'p(95)', 'p(99)', 'max'],
  thresholds: {
    dropped_iterations: ['count==0'],
    'http_reqs{endpoint:/products}': [`count>=${Math.floor(5 * durationSeconds * 0.98)}`],
    'http_reqs{endpoint:/orders}': [`count>=${Math.floor(2 * durationSeconds * 0.98)}`],
    'http_reqs{endpoint:/health}': [`count>=${Math.floor(durationSeconds * 0.98)}`],
    'checks{check:correlation headers preserved}': ['rate==1'],
  },
};

function request(method, endpoint, expectedStatus, body = null) {
  const requestID = crypto.randomUUID();
  const response = http.request(method, `${baseURL}${endpoint}`, body, {
    headers: {
      'Content-Type': 'application/json',
      'X-Request-ID': requestID,
      'X-Experiment-Run-ID': runID,
    },
    tags: { endpoint },
    timeout: '5s',
    redirects: 0,
  });
  check(response, {
    [`${method} ${endpoint} returns ${expectedStatus}`]: (r) => r.status === expectedStatus,
    'correlation headers preserved': (r) =>
      r.headers['X-Request-Id'] === requestID && r.headers['X-Experiment-Run-Id'] === runID,
  });
}

export function products() {
  request('GET', '/products', 200);
}

export function orders() {
  request('POST', '/orders', 201, JSON.stringify({ product_id: 1, quantity: 2 }));
}

export function health() {
  request('GET', '/health', 200);
}

export function handleSummary(data) {
  const summary = {
    experiment_run_id: runID,
    testid: runID,
    scenario: scenarioName,
    duration,
    requests_per_scheduled_second: (data.metrics.http_reqs?.values.count || 0) / durationSeconds,
    http_reqs: data.metrics.http_reqs?.values.count || 0,
    http_req_failed: data.metrics.http_req_failed?.values.rate ?? null,
    http_req_duration_ms: data.metrics.http_req_duration?.values || {},
    vus_max: data.metrics.vus_max?.values.max || 0,
    dropped_iterations: data.metrics.dropped_iterations?.values.count || 0,
    checks: data.metrics.checks?.values || {},
  };
  return { stdout: `${JSON.stringify(summary, null, 2)}\n` };
}
