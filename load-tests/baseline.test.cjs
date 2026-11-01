const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');
const source = readFileSync(`${__dirname}/baseline.js`, 'utf8')
  .replace(/^import .*;$/gm, '').replace(/export /g, '');
const id = '48dcf727-540a-4a3f-9c8e-eac82405ecb9';
function load(env = {}, capture = [], iteration = { value: 0 }, sleeps = []) {
  const context = { __ENV: { K6_SEED: '7', ...env }, crypto: { randomUUID: () => 'request-uuid' },
    execution: { scenario: { get iterationInTest() { return iteration.value; } } },
    sleep: (seconds) => sleeps.push(seconds),
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
  return vm.runInNewContext(source + '\n({options, products, orders, health, invalid, handleSummary, iterationRandom, intervals});', context);
}
const RATES = [['products', 53, 10], ['orders', 19, 10], ['health', 9, 10], ['invalid', 43, 100]];
test('requires neutral testid, controlled scenario and an integer seed', () => {
  for (const value of ['', 'stress80', 'normal', 'anomaly', 'run-001']) {
    assert.throws(() => load({ K6_EXPERIMENT_RUN_ID: value }), /UUID/);
  }
  assert.throws(() => load({ K6_EXPERIMENT_RUN_ID: id, K6_SCENARIO: 'free-input' }), /K6_SCENARIO/);
  for (const value of ['baseline', 'stress_cpu', 'recovery', 'network_latency', 'network_timeout', 'combined']) {
    assert.equal(load({ K6_EXPERIMENT_RUN_ID: id, K6_SCENARIO: value }).options.tags.scenario, value);
  }
  for (const value of ['', '-1', '1.5', 'abc', '4294967296', '12345678901']) {
    assert.throws(() => load({ K6_EXPERIMENT_RUN_ID: id, K6_SEED: value }), /K6_SEED/);
  }
  for (const value of ['0', '4294967295']) load({ K6_EXPERIMENT_RUN_ID: id, K6_SEED: value });
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
      assert.equal(s.preAllocatedVUs, Math.ceil((rate * (delay + 3000)) / (unit * 1000)) + 1);
      assert.equal(s.maxVUs, Math.ceil((rate * (delay + 5000)) / (unit * 1000)) + 1);
      assert.equal(s.gracefulStop, `${Math.ceil(5 + unit / rate)}s`);
      assert.ok(s.maxVUs >= s.preAllocatedVUs);
    }
    for (const p of ['avg', 'p(50)', 'p(95)', 'p(99)', 'max']) assert.ok(options.summaryTrendStats.includes(p));
    assert.equal(options.thresholds.dropped_iterations[0], 'count==0');
  }
  const { options } = load({ K6_EXPERIMENT_RUN_ID: id });
  for (const [name, pre, max, minimum] of [['products', 28, 39, 155], ['orders', 11, 15, 55], ['health', 6, 8, 26]]) {
    assert.equal(options.scenarios[name].preAllocatedVUs, pre);
    assert.equal(options.scenarios[name].maxVUs, max);
    assert.equal(options.thresholds[`http_reqs{endpoint:/${name}}`][0], `count>=${minimum}`);
  }
  assert.equal(options.scenarios.invalid.preAllocatedVUs, 4);
  assert.equal(options.scenarios.invalid.maxVUs, 5);
  assert.equal(options.scenarios.invalid.gracefulStop, '8s');
  assert.equal(options.thresholds['http_reqs{endpoint:invalid}'][0], 'count>=12');
  assert.equal(options.thresholds['checks{endpoint:invalid}'][0], 'rate==1');
  const perSecond = Object.fromEntries(RATES.map(([name, rate, unit]) => [name, rate / unit]));
  const share = perSecond.invalid / Object.values(perSecond).reduce((a, b) => a + b);
  assert.ok(share > 0.045 && share < 0.055, `invalid share ${share}`);
  const compose = readFileSync(`${__dirname}/../docker-compose.yml`, 'utf8');
  assert.match(compose, /K6_PROMETHEUS_RW_TREND_STATS: avg,p\(50\),p\(95\),p\(99\),max/);
  assert.match(compose, /K6_SEED: \$\{K6_SEED:-\}/);
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
test('every iteration sleeps a reproducible fraction of its interval before the request', () => {
  for (const name of ['products', 'orders', 'health', 'invalid']) {
    const runs = ['7', '7', '8'].map((seed) => {
      const sleeps = [];
      const iteration = { value: 0 };
      const script = load({ K6_EXPERIMENT_RUN_ID: id, K6_SEED: seed }, [], iteration, sleeps);
      for (let i = 0; i < 50; i += 1) {
        iteration.value = i;
        script[name]();
      }
      return { sleeps, interval: script.intervals[name] };
    });
    assert.equal(runs[0].sleeps.length, 50);
    assert.deepEqual(runs[0].sleeps, runs[1].sleeps);
    assert.notDeepEqual(runs[0].sleeps, runs[2].sleeps);
    for (const seconds of runs[0].sleeps) assert.ok(seconds >= 0 && seconds < runs[0].interval, `${name} ${seconds}`);
  }
});
test('invalid flow draws 404 or 422 from the seeded generator as expected responses', () => {
  const draw = (seed) => {
    const calls = [];
    const iteration = { value: 0 };
    const script = load({ K6_EXPERIMENT_RUN_ID: id, K6_SCENARIO: 'network_timeout', K6_SEED: seed }, calls, iteration);
    for (let i = 0; i < 400; i += 1) {
      iteration.value = i;
      script.invalid();
    }
    return calls;
  };
  const calls = draw('7');
  const notFound = calls.filter((c) => JSON.parse(c.body).product_id === 999).length;
  assert.ok(notFound > 160 && notFound < 240, `404 share ${notFound}/400`);
  assert.deepEqual(calls.map((c) => c.body), draw('7').map((c) => c.body));
  assert.notDeepEqual(calls.map((c) => c.body), draw('8').map((c) => c.body));
  for (const call of calls) {
    assert.ok([JSON.stringify({ product_id: 999, quantity: 2 }), JSON.stringify({ product_id: 1, quantity: 0 })].includes(call.body));
    assert.equal(call.method, 'POST');
    assert.ok(call.url.endsWith('/orders'));
    assert.equal(call.params.tags.endpoint, 'invalid');
    assert.deepEqual([...call.params.responseCallback.statuses], [404, 422]);
    assert.equal(call.params.headers['X-Request-ID'], 'request-uuid');
    assert.equal(call.params.headers['X-Experiment-Run-ID'], id);
  }
});
test('PRNG is reproducible and uniform over a large sample', () => {
  const script = load({ K6_EXPERIMENT_RUN_ID: id, K6_SEED: '20261101' });
  const again = load({ K6_EXPERIMENT_RUN_ID: id, K6_SEED: '20261101' });
  const n = 100000;
  const bins = new Array(20).fill(0);
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const u = script.iterationRandom('products', i)();
    assert.equal(u, again.iterationRandom('products', i)());
    assert.ok(u >= 0 && u < 1);
    bins[Math.floor(u * 20)] += 1;
    sum += u;
  }
  assert.ok(Math.abs(sum / n - 0.5) < 0.005, `mean ${sum / n}`);
  const expected = n / 20;
  const chiSquare = bins.reduce((acc, count) => acc + (count - expected) ** 2 / expected, 0);
  assert.ok(chiSquare < 43.82, `chi-square ${chiSquare} (df 19, p = 0.001 critical value 43.82)`);
  const second = script.iterationRandom('products', 1);
  second();
  assert.notEqual(second(), script.iterationRandom('products', 1)());
});
