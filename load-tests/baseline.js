import { check } from 'k6';
import execution from 'k6/execution';
import http from 'k6/http';

const baseURL = (__ENV.K6_BASE_URL || 'http://application-api:8000').replace(/\/+$/, '');
const duration = __ENV.K6_TEST_DURATION || '30s';
const runID = __ENV.K6_EXPERIMENT_RUN_ID || '';
const scenarioName = __ENV.K6_SCENARIO || 'baseline';
const maxDelayMs = Number(__ENV.K6_MAX_EXPERIMENT_DELAY_MS || '2000');

if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(runID)) {
  throw new Error('K6_EXPERIMENT_RUN_ID (testid) is required and must be a neutral UUID v4.');
}
const scenarioNames = ['baseline', 'stress_cpu', 'recovery', 'network_latency', 'network_timeout', 'combined'];
if (!scenarioNames.includes(scenarioName)) {
  throw new Error(`K6_SCENARIO must be one of: ${scenarioNames.join(', ')}.`);
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
// Rates are iterations per 10 s (client-error noise: per 100 s): arrival intervals
// (188.7/526.3/1111.1/2325.6 ms) are not multiples of 100 ms, so arrivals do not
// lock onto the CPU quota (CFS) period.
const timeUnitSeconds = 10;
const invalidTimeUnitSeconds = 100;

function scenario(exec, rate, endpoint, unitSeconds = timeUnitSeconds) {
  return {
    executor: 'constant-arrival-rate',
    exec,
    rate,
    timeUnit: `${unitSeconds}s`,
    duration,
    // Delay + dependency budget + scheduling margin (27/10/5/3 at 2000 ms).
    preAllocatedVUs: Math.ceil((rate * (maxDelayMs + 3000)) / (unitSeconds * 1000)),
    maxVUs: Math.ceil((rate * (maxDelayMs + 5000)) / (unitSeconds * 1000)),
    gracefulStop: '5s',
    tags: { endpoint },
  };
}

export const options = {
  scenarios: {
    products: scenario('products', 53, '/products'),
    orders: scenario('orders', 19, '/orders'),
    health: scenario('health', 9, '/health'),
    // Client-error noise (~5% of requests): 404 and 422 on /orders, no Go lookup.
    invalid: scenario('invalid', 43, 'invalid', invalidTimeUnitSeconds),
  },
  systemTags: ['method', 'status', 'check', 'expected_response'],
  // testid cardinality grows only with the controlled number of experiments.
  // Request IDs, full URLs, names and free user input must never become tags.
  tags: { scenario: scenarioName, testid: runID },
  discardResponseBodies: true,
  summaryTrendStats: ['avg', 'p(50)', 'p(95)', 'p(99)', 'max'],
  thresholds: {
    dropped_iterations: ['count==0'],
    'http_reqs{endpoint:/products}': [`count>=${Math.floor((53 * durationSeconds * 0.98) / timeUnitSeconds)}`],
    'http_reqs{endpoint:/orders}': [`count>=${Math.floor((19 * durationSeconds * 0.98) / timeUnitSeconds)}`],
    'http_reqs{endpoint:/health}': [`count>=${Math.floor((9 * durationSeconds * 0.98) / timeUnitSeconds)}`],
    'http_reqs{endpoint:invalid}': [`count>=${Math.floor((43 * durationSeconds * 0.98) / invalidTimeUnitSeconds)}`],
    // 404/422 never depend on stress or the network, so every status check must pass.
    'checks{endpoint:invalid}': ['rate==1'],
    'checks{check:correlation headers preserved}': ['rate==1'],
  },
};

// 404 and 422 are the expected outcome of the client-error flow, not failures.
const clientErrors = http.expectedStatuses(404, 422);

function request(method, path, expectedStatus, body = null, endpoint = path) {
  const requestID = crypto.randomUUID();
  const params = {
    headers: {
      'Content-Type': 'application/json',
      'X-Request-ID': requestID,
      'X-Experiment-Run-ID': runID,
    },
    tags: { endpoint },
    timeout: '5s',
    redirects: 0,
  };
  if (expectedStatus >= 400) params.responseCallback = clientErrors;
  const response = http.request(method, `${baseURL}${path}`, body, params);
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

export function invalid() {
  // Alternates across the whole test: unknown product (404), invalid quantity (422).
  if (execution.scenario.iterationInTest % 2 === 0) {
    request('POST', '/orders', 404, JSON.stringify({ product_id: 999, quantity: 2 }), 'invalid');
  } else {
    request('POST', '/orders', 422, JSON.stringify({ product_id: 1, quantity: 0 }), 'invalid');
  }
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
