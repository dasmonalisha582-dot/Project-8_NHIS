"""
app.py - Flask REST API (+ tiny web form) for the job recommendation engine.

Run locally (from the project root):
    pip install -r requirements.txt
    cd app && python app.py                       # development server on http://127.0.0.1:5000
    cd app && gunicorn -b 0.0.0.0:5000 app:app    # production-style server (what Docker uses)

Configuration via environment variables (no secrets are needed or stored):
    JOBS_DATA_PATH   path to outputs/cleaned_jobs.csv.gz   (default: ../outputs/cleaned_jobs.csv.gz)
    PORT             port for `python app.py`              (default: 5000)

Endpoints
    GET  /            simple HTML form
    GET  /health      liveness + whether the model is loaded
    GET  /meta        allowed values, data date range, available countries
    POST /recommend   JSON preferences -> ranked recommendations (see README.md)
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, request

from recommender import JobRecommender, ValidationError, EMPLOYMENT_TYPES, EXPERIENCE_LEVELS, WORK_MODES

log = logging.getLogger("job-api")
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "outputs" / "cleaned_jobs.csv.gz"

_state = {"rec": None, "error": None}
_lock = threading.Lock()


def get_recommender() -> JobRecommender:
    """Build the TF-IDF index once (first request / startup), then reuse it. Thread-safe."""
    if _state["rec"] is None:
        with _lock:
            if _state["rec"] is None:
                path = Path(os.environ.get("JOBS_DATA_PATH", DEFAULT_DATA))
                log.info("Loading cleaned jobs from %s", path)
                _state["rec"] = JobRecommender.from_csv(path)
    return _state["rec"]


def create_app(recommender: JobRecommender | None = None) -> Flask:
    """Application factory (lets tests inject a small recommender)."""
    flask_app = Flask(__name__)
    flask_app.json.sort_keys = False  # Flask >= 2.3 API
    flask_app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # preferences are tiny; reject big bodies
    if recommender is not None:
        _state["rec"] = recommender

    def rec_or_error():
        try:
            return get_recommender(), None
        except FileNotFoundError as exc:
            return None, (jsonify(error="data_not_found", message=str(exc)), 503)

    @flask_app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=_state["rec"] is not None)

    @flask_app.get("/meta")
    def meta():
        rec, err = rec_or_error()
        if err:
            return err
        jobs = rec.jobs
        return jsonify(
            n_postings=int(len(jobs)),
            newest_posting=str(rec.reference_date),
            employment_types=sorted(EMPLOYMENT_TYPES), work_modes=sorted(WORK_MODES),
            experience_levels=sorted(EXPERIENCE_LEVELS),
            countries=sorted(jobs["country"].dropna().unique().tolist()),
            notes=["No company, experience-level or work-mode fields exist in the source data.",
                   "Pay is in USD as shown on the platform (assumption)."])

    @flask_app.post("/recommend")
    def recommend():
        if not request.is_json:
            return jsonify(error="unsupported_media_type",
                           message="Send JSON with header 'Content-Type: application/json'."), 415
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify(error="invalid_json", message="Request body is not valid JSON."), 400
        rec, err = rec_or_error()
        if err:
            return err
        try:
            return jsonify(rec.recommend(payload))
        except ValidationError as exc:
            return jsonify(error="validation_error", messages=exc.errors), 400
        except Exception:  # never leak stack traces / internals to clients
            log.exception("Unexpected error in /recommend")
            return jsonify(error="internal_error", message="Unexpected server error."), 500

    @flask_app.errorhandler(404)
    def not_found(_):
        return jsonify(error="not_found", message="Unknown endpoint. See GET /meta and POST /recommend."), 404

    @flask_app.errorhandler(405)
    def bad_method(_):
        return jsonify(error="method_not_allowed", message="Use POST for /recommend."), 405

    @flask_app.errorhandler(413)
    def too_large(_):
        return jsonify(error="payload_too_large", message="Request body too large."), 413

    @flask_app.get("/")
    def index():
        return INDEX_HTML

    return flask_app


INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Job recommender</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}
label{display:block;margin-top:.6rem;font-weight:600}input,select{padding:.4rem;width:100%;max-width:420px}
button{margin-top:1rem;padding:.6rem 1.2rem}.card{border:1px solid #ccc;border-radius:8px;padding:.8rem;margin:.8rem 0}
small{color:#555}.warn{background:#fff6d6;padding:.5rem;border-radius:6px;margin:.4rem 0}</style></head><body>
<h1>Job recommender (prototype)</h1>
<p><small>Matches your keywords against Upwork job TITLES from a 2024 snapshot. No company, experience or
work-mode data exists in the source, so those preferences act only as weak title signals.</small></p>
<label>Role / job title<input id="role" placeholder="e.g. video editor"></label>
<label>Skills / keywords (comma-separated)<input id="skills" placeholder="e.g. youtube, premiere pro"></label>
<label>Client country (optional)<input id="country" placeholder="e.g. United States"></label>
<label>Employment type<select id="employment_type"><option>any</option><option>hourly</option><option>fixed_price</option></select></label>
<label>Minimum hourly rate (USD, optional)<input id="min_hourly_rate" type="number" min="0"></label>
<label>Minimum fixed budget (USD, optional)<input id="min_fixed_budget" type="number" min="0"></label>
<label>Work mode<select id="work_mode"><option>any</option><option>remote</option><option>hybrid</option><option>onsite</option></select></label>
<label>Experience level<select id="experience_level"><option>any</option><option>junior</option><option>mid</option><option>senior</option></select></label>
<button onclick="go()">Recommend</button><div id="out"></div>
<script>
async function go(){
  const g=id=>document.getElementById(id).value.trim(); const body={top_k:10};
  for (const k of ['role','skills','country','employment_type','work_mode','experience_level']) if(g(k)) body[k]=g(k);
  for (const k of ['min_hourly_rate','min_fixed_budget']) if(g(k)) body[k]=Number(g(k));
  const r=await fetch('/recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json(); const o=document.getElementById('out'); o.innerHTML='';
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  if(!r.ok){o.innerHTML='<div class="warn">'+esc((d.messages||[d.message]).join(' | '))+'</div>';return;}
  (d.warnings||[]).forEach(w=>o.innerHTML+='<div class="warn">'+esc(w)+'</div>');
  d.recommendations.forEach(x=>{o.innerHTML+='<div class="card"><b>#'+x.rank+' '+esc(x.job_title)+'</b> <small>score '+x.score+
   '</small><br>'+esc(x.pay)+' &middot; '+esc(x.employment_type)+' &middot; client country: '+esc(x.location||'n/a')+
   '<br><small>'+x.why_recommended.map(esc).join('<br>')+'</small><br><a href="'+esc(x.url)+'" target="_blank" rel="noopener">Open original posting</a></div>';});
}
</script></body></html>"""

app = create_app()

# Optional warm-up so the first HTTP request is not slow (the Docker image sets PRELOAD=1).
if os.environ.get("PRELOAD") == "1":
    try:
        get_recommender()
    except FileNotFoundError as exc:  # server still starts; /recommend will answer 503 until data exists
        logging.getLogger("job-api").warning("Preload skipped: %s", exc)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
