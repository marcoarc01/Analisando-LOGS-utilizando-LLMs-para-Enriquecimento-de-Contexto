const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const source = readFileSync(`${__dirname}/baseline.js`, 'utf8')
  .replace(/^import .*;$/gm, '').replace(/export /g, '');
const id = '48dcf727-540a-4a3f-9c8e-eac82405ecb9';
function load(env = {}, capture = [], iteration = { value: 0 }) {
  const context = { __ENV: env, crypto: { randomUUID: () => 'request-uuid' },
    execution: { scenario: { get iterationInTest() { return iteration.value; } } },
    check: (response, checks) => Object.values(checks).forEach(fn => assert.equal(fn(response), true)),
    http: { expectedStatuses: (...statuses) => ({ statuses }), request: (method, url, body, params) => {
      capture.push({ method, url, body, params });
      const status = body?.includes('"product_id":999') ? 404
        : body?.includes('"quantity":0') ? 422 : method === 'POST' ? 201 : 200;
      return { status, headers: {
        'X-Request-Id': params.headers['X-Request-ID'],
        'X-Experiment-Run-Id': params.headers['X-Experiment-Run-ID'],
      }};
    }},
  };
  return vm.runInNewContext(source + '\n({options, products, orders, health, invalid, handleSummary});', context);
}
const RATES = [['products', 53, 10], ['orders', 19, 10], ['health', 9, 10], ['invalid', 43, 100]];
test('requires neutral testid and controlled scenario', () => {
  for (const value of ['', 'stress80', 'normal', 'anomaly', 'run-001']) {
    assert.throws(() => load({ K6_EXPERIMENT_RUN_ID: value }), /UUID/);
  }
  assert.throws(() => load({ K6_EXPERIMENT_RUN_ID: id, K6_SCENARIO: 'free-input' }), /K6_SCENARIO/);
  for (const value of ['baseline', 'stress_cpu', 'recovery', 'network_latency', 'network_timeout', 'combined']) {
    assert.equal(load({ K6_EXPERIMENT_RUN_ID: id, K6_SCENARIO: value }).options.tags.scenario, value);
  }
});
test('capacity, arrival rates, percentiles and thresholds', () => {
  for (const delay of [0, 2000, 4000]) {
    const { options } = load({ K6_EXPERIMENT_RUN_ID: id, K6_MAX_EXPERIMENT_DELAY_MS: String(delay) });
    assert.deepEqual(Object.keys(options.scenarios).sort(), RATES.map(([name]) => name).sort());
    for (const [name, rate, unit] of RATES) {
      const s = options.scenarios[name];
      assert.equal(s.rate, rate);
      assert.equal(s.timeUnit, `${unit}s`);
      assert.equal(s.duration, '30s');
      assert.notEqual((unit * 1000 / rate) % 100, 0);
      assert.ok(s.preAllocatedVUs > (rate / unit) * delay / 1000);
      assert.equal(s.preAllocatedVUs, Math.ceil((rate * (delay + 3000)) / (unit * 1000)));
      assert.equal(s.maxVUs, Math.ceil((rate * (delay + 5000)) / (unit * 1000)));
      assert.ok(s.maxVUs >= s.preAllocatedVUs);
    }
    for (const p of ['avg', 'p(50)', 'p(95)', 'p(99)', 'max']) assert.ok(options.summaryTrendStats.includes(p));
    assert.equal(options.thresholds.dropped_iterations[0], 'count==0');
  }
  const { options } = load({ K6_EXPERIMENT_RUN_ID: id });
  for (const [name, pre, max, minimum] of [['products', 27, 38, 155], ['orders', 10, 14, 55], ['health', 5, 7, 26]]) {
    assert.equal(options.scenarios[name].preAllocatedVUs, pre);
    assert.equal(options.scenarios[name].maxVUs, max);
    assert.equal(options.thresholds[`http_reqs{endpoint:/${name}}`][0], `count>=${minimum}`);
  }
  assert.equal(options.scenarios.invalid.preAllocatedVUs, 3);
  assert.equal(options.scenarios.invalid.maxVUs, 4);
  assert.equal(options.thresholds['http_reqs{endpoint:invalid}'][0], 'count>=12');
  assert.equal(options.thresholds['checks{endpoint:invalid}'][0], 'rate==1');
  const perSecond = Object.fromEntries(RATES.map(([name, rate, unit]) => [name, rate / unit]));
  const share = perSecond.invalid / Object.values(perSecond).reduce((a, b) => a + b);
  assert.ok(share > 0.045 && share < 0.055, `invalid share ${share}`);
  const compose = readFileSync(`${__dirname}/../docker-compose.yml`, 'utf8');
  assert.match(compose, /K6_PROMETHEUS_RW_TREND_STATS: avg,p\(50\),p\(95\),p\(99\),max/);
});
test('same ID in metrics and headers, no per-request or URL labels', () => {
  const calls = [];
  const script = load({ K6_EXPERIMENT_RUN_ID: id, K6_SCENARIO: 'stress_cpu' }, calls);
  script.products(); script.orders(); script.health();
  assert.equal(script.options.tags.testid, id);
  assert.equal(script.options.tags.scenario, 'stress_cpu');
  assert.ok(!script.options.systemTags.includes('scenario'));
  for (const tag of ['url', 'name', 'request_id', 'experiment_run_id']) assert.ok(!script.options.systemTags.includes(tag));
  for (const call of calls) {
    assert.equal(call.params.headers['X-Experiment-Run-ID'], id);
    assert.deepEqual(Object.keys(call.params.tags), ['endpoint']);
    assert.equal(call.params.responseCallback, undefined);
  }
});
test('invalid flow alternates 404 and 422 as expected responses', () => {
  const calls = [];
  const iteration = { value: 0 };
  const script = load({ K6_EXPERIMENT_RUN_ID: id, K6_SCENARIO: 'network_timeout' }, calls, iteration);
  for (let i = 0; i < 4; i += 1) {
    iteration.value = i;
    script.invalid();
  }
  assert.deepEqual(calls.map(c => JSON.parse(c.body)), [
    { product_id: 999, quantity: 2 }, { product_id: 1, quantity: 0 },
    { product_id: 999, quantity: 2 }, { product_id: 1, quantity: 0 },
  ]);
  for (const call of calls) {
    assert.equal(call.method, 'POST');
    assert.ok(call.url.endsWith('/orders'));
    assert.equal(call.params.tags.endpoint, 'invalid');
    assert.deepEqual([...call.params.responseCallback.statuses], [404, 422]);
    assert.equal(call.params.headers['X-Request-ID'], 'request-uuid');
    assert.equal(call.params.headers['X-Experiment-Run-ID'], id);
  }
});
