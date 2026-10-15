const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const source = readFileSync(`${__dirname}/baseline.js`, 'utf8')
  .replace(/^import .*;$/gm, '').replace(/export /g, '');
const id = '48dcf727-540a-4a3f-9c8e-eac82405ecb9';
function load(env = {}, capture = []) {
  const context = { __ENV: env, crypto: { randomUUID: () => 'request-uuid' },
    check: (response, checks) => Object.values(checks).forEach(fn => assert.equal(fn(response), true)),
    http: { request: (method, url, body, params) => {
      capture.push({ method, url, body, params });
      return { status: method === 'POST' ? 201 : 200, headers: {
        'X-Request-Id': params.headers['X-Request-ID'],
        'X-Experiment-Run-Id': params.headers['X-Experiment-Run-ID'],
      }};
    }},
  };
  return vm.runInNewContext(source + '\n({options, products, orders, health, handleSummary});', context);
}
test('requires neutral testid and controlled scenario', () => {
  for (const value of ['', 'stress80', 'normal', 'anomaly', 'run-001']) {
    assert.throws(() => load({ K6_EXPERIMENT_RUN_ID: value }), /UUID/);
  }
  assert.throws(() => load({ K6_EXPERIMENT_RUN_ID: id, K6_SCENARIO: 'free-input' }), /K6_SCENARIO/);
});
test('capacity, arrival rates, percentiles and thresholds', () => {
  for (const delay of [0, 2000, 4000]) {
    const { options } = load({ K6_EXPERIMENT_RUN_ID: id, K6_MAX_EXPERIMENT_DELAY_MS: String(delay) });
    for (const [name, rate] of [['products', 53], ['orders', 19], ['health', 9]]) {
      const s = options.scenarios[name];
      assert.equal(s.rate, rate);
      assert.equal(s.timeUnit, '10s');
      assert.equal(s.duration, '30s');
      assert.notEqual((10000 / rate) % 100, 0);
      assert.ok(s.preAllocatedVUs > (rate / 10) * delay / 1000);
      assert.ok(s.maxVUs > s.preAllocatedVUs);
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
  }
});
