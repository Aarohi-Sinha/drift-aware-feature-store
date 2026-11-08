import http from 'k6/http';
import { check } from 'k6';

const VUS = Number(__ENV.VUS || 100);
const MODE = __ENV.MODE || 'both';
const BASE = __ENV.BASE_URL || 'http://api:8000';
const H = { 'Content-Type': 'application/json' };

export const options = {
  scenarios: { load: { executor: 'constant-vus', vus: VUS, duration: '30s' } },
  summaryTrendStats: ['avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

export default function () {
  if (MODE === 'health') {
    const r = http.get(BASE + '/health');
    check(r, { 'health ok': (x) => x.status === 200 });
    return;
  }
  const entity = 'lt_' + Math.floor(Math.random() * 1000);
  if (MODE !== 'read') {
    const w = http.post(
      BASE + (__ENV.WRITE_PATH || '/features/write'),
      JSON.stringify({ entity_id: entity, feature_name: 'loadtest_feature', value: Math.random() * 100 }),
      { headers: H }
    );
    check(w, { 'write ok': (r) => r.status === 200 });
  }
  if (MODE !== 'write') {
    const rd = http.get(BASE + '/features/' + entity + '/loadtest_feature');
    check(rd, { 'read ok': (r) => r.status === 200 && r.json('source') === 'online_store' });
  }
}
