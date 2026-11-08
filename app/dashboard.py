from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.config import RETRAIN_COMPUTE_COST_UNITS
from app.drift import compute_psi
from app.model import load_data, TRAIN_HOURS
from app.simulate import run_policy, POLICIES

router = APIRouter()
_cache = {}

def build_results():
    if _cache:
        return _cache
    df = load_data()
    last_hour = int(df["hour"].max())
    hours = list(range(TRAIN_HOURS, last_hour+1))
    train_amounts = df[df["hour"]< TRAIN_HOURS]["amount"].values

    psi = [
        round(compute_psi(train_amounts, df[df["hour"] == h]["amount"].values), 3)
        for h in hours
    ]

    policies = {}
    for name in POLICIES:
        retrains, accs, events = run_policy(name, df, last_hour)
        vals = [round(accs[h], 3) for h in hours]
        policies[name] = {
            "retrains": retrains,
            "compute": retrains * RETRAIN_COMPUTE_COST_UNITS,
            "mean_acc": round(sum(vals) / len(vals), 3),
            "worst_acc": min(vals),
            "accuracy": vals,
            "retrain_hours": [int(e.model_version.split("-h")[-1]) for e in events],
        }

    _cache.update({"hours": hours, "psi": psi, "policies": policies})
    return _cache

@router.get("/api/simulation")
def simulation():
    return build_results()

@router.get("/dashboard",response_class = HTMLResponse)
def dashboard():
    return PAGE

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Feature Store: Drift and Retraining</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 16px; color: #111; }
  h1 { font-size: 22px; }
  p.note { color: #555; font-size: 14px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 14px; }
  th, td { border-bottom: 1px solid #ddd; padding: 8px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  .chart-box { margin-bottom: 32px; }
</style>
</head>
<body>
<h1>Drift detection and retraining policies</h1>
<p class="note">Same data, four strategies. Large dots on the accuracy chart mark an hour where that policy retrained the model.</p>
<p id="status" class="note">Running simulation (first load takes a few seconds)...</p>

<table id="summary"></table>

<div class="chart-box"><canvas id="acc"></canvas></div>
<div class="chart-box"><canvas id="psi"></canvas></div>

<script>
const COLORS = {
  never: '#9ca3af',
  fixed_schedule: '#f59e0b',
  naive_threshold: '#ef4444',
  cost_aware: '#10b981'
};

fetch('/api/simulation')
  .then(function (r) { return r.json(); })
  .then(function (d) {
    document.getElementById('status').remove();

    var rows = '<tr><th>Policy</th><th>Retrains</th><th>Compute</th><th>Mean accuracy</th><th>Worst hour</th></tr>';
    Object.keys(d.policies).forEach(function (name) {
      var p = d.policies[name];
      rows += '<tr><td>' + name + '</td><td>' + p.retrains + '</td><td>' + p.compute.toFixed(1) +
              '</td><td>' + p.mean_acc.toFixed(3) + '</td><td>' + p.worst_acc.toFixed(3) + '</td></tr>';
    });
    document.getElementById('summary').innerHTML = rows;

    var accSets = Object.keys(d.policies).map(function (name) {
      var p = d.policies[name];
      return {
        label: name,
        data: p.accuracy,
        borderColor: COLORS[name],
        backgroundColor: COLORS[name],
        borderWidth: 2,
        tension: 0.2,
        pointRadius: d.hours.map(function (h) { return p.retrain_hours.indexOf(h) >= 0 ? 6 : 0; })
      };
    });

    new Chart(document.getElementById('acc'), {
      type: 'line',
      data: { labels: d.hours, datasets: accSets },
      options: {
        plugins: { title: { display: true, text: 'Model accuracy per simulated hour' } },
        scales: {
          y: { min: 0, max: 1, title: { display: true, text: 'Accuracy' } },
          x: { title: { display: true, text: 'Simulated hour' } }
        }
      }
    });

    new Chart(document.getElementById('psi'), {
      type: 'line',
      data: {
        labels: d.hours,
        datasets: [
          { label: 'PSI vs training data', data: d.psi, borderColor: '#3b82f6', backgroundColor: '#3b82f6', borderWidth: 2, pointRadius: 0 },
          { label: 'Action threshold (0.25)', data: d.hours.map(function () { return 0.25; }), borderColor: '#111', borderDash: [6, 4], borderWidth: 1, pointRadius: 0 }
        ]
      },
      options: {
        plugins: { title: { display: true, text: 'Drift score (PSI, log scale)' } },
        scales: {
          y: { type: 'logarithmic', title: { display: true, text: 'PSI' } },
          x: { title: { display: true, text: 'Simulated hour' } }
        }
      }
    });
  })
  .catch(function (e) {
    document.getElementById('status').textContent = 'Failed to load results: ' + e;
  });
</script>
</body>
</html>
"""