import http from 'k6/http';
import { check } from 'k6';

const VUS = Number(__ENV.VUS || 20);
const BASE = __ENV.BASE_URL || 'http://api:8000';
const JSON_HEADERS = { 'Content-Type': 'application/json' };

export const options = {
    scenarios: {
        load: { executor: 'constant-vus', vus: VUS, duration: '30s' }
    },
    summaryTrendStats: ['avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
    thresholds: {
        'http_req_duration{endpoint:write}': ['p(95)<2000'],
        'http_req_duration{endpoint:read}': ['p(95)<2000'],
    },
};

export default function () {
    const entity = 'lt_' + Math.floor(Math.random() * 1000);
    const w = http.post(
        BASE + '/features/write',
        JSON.stringify({ entity_id: entity, feature_name: 'loadtest_feature', value: Math.random() * 100 }),
        { headers: JSON_HEADERS, tags: { endpoint: 'write' } }
    );
    check(w, { 'write ok': (r) => r.status === 200 });

    const rd = http.get(BASE + '/features/' + entity + '/loadtest_feature', {
        tags: { endpoint: 'read' },
    });
    check(rd, { 'read served from redis': (r) => r.status === 200 && r.json('source') === 'online_store' });
}