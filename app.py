"""UPI Guard - wallet-style payment backend with a fraud engine in the payment path.

Payment lifecycle:  check (score, nothing moves) -> confirm with UPI PIN (money moves atomically)
                    BLOCK decisions cannot be confirmed by the user; they go to UNDER_REVIEW for an analyst.
The settlement rail is a sandbox: balances live in our own ledger. See README for plugging in a real PSP.

Files: app.py (this file: DB, fraud engine, API), index.html (frontend), requirements.txt.
"""
import os, re, json, secrets, random, time, sqlite3, difflib, urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from flask import Flask, request, jsonify, session, send_from_directory, g
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE, ".env")):  # optional .env (XAI_API_KEY, SECRET_KEY, DEMO_MODE, PORT ...)
    for line in open(os.path.join(HERE, ".env")):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v.strip("\"'"))



# ============================================================================
# 1. DATABASE (SQLite schema, atomic transaction helper)
# ============================================================================

DB_PATH = os.getenv("DB_PATH", os.path.join(HERE, "upi_guard_pay.db"))


def db_conn():
    c = sqlite3.connect(DB_PATH, timeout=15, isolation_level=None)  # we manage transactions explicitly
    c.row_factory = sqlite3.Row
    c.execute("pragma journal_mode=wal")
    c.execute("pragma foreign_keys=on")
    return c


@contextmanager
def db_tx():
    """BEGIN IMMEDIATE takes the write lock up-front, so two concurrent payments can never
    both read the same balance. Any exception rolls the whole unit of work back."""
    c = db_conn()
    try:
        c.execute("begin immediate")
        yield c
        c.execute("commit")
    except BaseException:
        if c.in_transaction:
            c.execute("rollback")
        raise
    finally:
        c.close()


SCHEMA = """
create table if not exists users(
  id integer primary key autoincrement,
  name text not null, phone text unique not null, email text,
  vpa text unique not null,
  pw_hash text not null, pin_hash text not null, pin_len integer not null default 4,
  role text not null default 'user',
  balance integer not null default 0 check(balance >= 0),          -- paise, never negative
  daily_limit integer not null default 10000000,                   -- Rs 1,00,000 in paise
  kyc text not null default 'verified',
  pin_fails integer not null default 0, locked_until text,
  created text not null);

create table if not exists reputation(                              -- network-level signals per UPI ID
  vpa text primary key, name text, complaints integer not null default 0,
  fan_in_base integer not null default 0, created text, external integer not null default 0);

create table if not exists beneficiaries(
  user_id integer not null, vpa text not null, name text, created text,
  primary key(user_id, vpa));

create table if not exists collect_requests(
  id integer primary key autoincrement,
  requester_id integer not null, payer_vpa text not null, display_name text,
  amount integer not null, note text, status text not null default 'PENDING', created text);

create table if not exists payments(
  id integer primary key autoincrement, ref text unique,
  user_id integer, payee_vpa text, payee_name text, display_name text,
  amount integer, mode text, note text,
  status text,            -- CHECKED | SUCCESS | CANCELLED | UNDER_REVIEW | REJECTED
  risk real, model_risk real, decision text, category text,
  features text, reasons text, contrib text, ai text,
  analyst_label text, user_ack integer default 0,
  source text default 'live',                                        -- live | history | seed
  collect_id integer, created text, updated text);
create index if not exists ix_pay_user on payments(user_id, status);
create index if not exists ix_pay_payee on payments(payee_vpa, status);

create table if not exists ledger(                                  -- append-only double-entry style trail
  id integer primary key autoincrement,
  payment_id integer, user_id integer, kind text,
  delta integer not null, balance_after integer, ts text);

create table if not exists reports(
  id integer primary key autoincrement,
  reporter_id integer, vpa text, payment_id integer, reason text, ts text,
  unique(reporter_id, vpa));

create table if not exists patterns(
  id integer primary key autoincrement, name text unique, description text,
  keywords text, status text, hits integer default 0, created text);

create table if not exists model_meta(id integer primary key autoincrement, ts text, metrics text);

create table if not exists audit(
  id integer primary key autoincrement, ts text, user_id integer, event text, detail text);
"""


def db_init():
    c = db_conn()
    c.executescript(SCHEMA)
    c.close()


# ============================================================================
# 2. FRAUD ENGINE (ML model + rules + optional Grok; pure functions, no DB access)
# ============================================================================

KEY = os.getenv("XAI_API_KEY")
GM = os.getenv("GROK_MODEL", "grok-3")

FEATS = ["amount_log", "night", "amount_ratio", "new_payee", "payee_age", "complaints", "fan_in", "collect", "qr",
         "link", "refund_word", "kyc_word", "urgency_word", "name_mismatch", "screenshare", "remote_app"]
FL = {"amount_log": "Large amount", "night": "Unusual hour (night)", "amount_ratio": "Amount far above your usual",
      "new_payee": "First payment to this payee", "payee_age": "Very new payee account",
      "complaints": "Past fraud complaints on payee", "fan_in": "Payee receives from many senders (mule pattern)",
      "collect": "Collect request (money goes OUT)", "qr": "Paying via QR", "link": "Paying via link",
      "refund_word": "'Refund/cashback' wording", "kyc_word": "KYC / account-block wording",
      "urgency_word": "Urgency pressure", "name_mismatch": "Name shown differs from account holder",
      "screenshare": "Screen sharing active", "remote_app": "Remote-access app installed"}
CATS = {"collect_refund": "Collect-request refund scam", "qr_swap": "QR code swap", "remote_access": "Remote-access app scam",
        "fake_kyc": "Fake KYC / bank alert", "mule_account": "Mule-account chain", "fake_job_investment": "Fake job / investment",
        "suspicious_other": "Other suspicious", "none": "Legitimate"}
M = {}


# ---------------------------------------------------------------- training
def synth(n, seed):
    r = np.random.default_rng(seed); X = []; y = []; C = []; U = r.uniform
    P = lambda q: float(r.random() < q)
    for c in r.choice(["legit", "collect_refund", "qr_swap", "remote_access", "fake_kyc", "mule_account"], n,
                      p=[.74, .06, .06, .06, .05, .03]):
        amt = float(np.clip(r.lognormal(6.3, 1.1), 10, 60000)); mode = r.choice(["pay", "qr", "link", "collect"], p=[.45, .38, .08, .09])
        new = P(.2)
        f = dict(night=P(.06), new_payee=new, payee_age=U(1, 60) if new else U(60, 2500), complaints=float(r.poisson(.05)),
                 fan_in=float(r.integers(1, 9)), refund_word=P(.01), kyc_word=P(.01), urgency_word=P(.03),
                 name_mismatch=float(np.clip(abs(r.normal(0, .06)), 0, 1)) if P(.9) else U(.1, .35), screenshare=P(.01), remote_app=P(.03))
        ratio = float(np.clip(r.lognormal(0, .6), .05, 30))
        if c == "collect_refund":
            mode = "collect"; amt = U(300, 20000); ratio = U(.5, 6)
            f.update(refund_word=P(.9), new_payee=P(.9), payee_age=U(1, 40), urgency_word=P(.4), complaints=float(r.poisson(2)))
        elif c == "qr_swap":
            mode = "qr"; f.update(name_mismatch=U(.45, .95) if P(.85) else U(0, .3), new_payee=P(.8), payee_age=U(1, 90), complaints=float(r.poisson(1)))
        elif c == "remote_access":
            amt = U(5000, 100000); ratio = U(2, 25)
            f.update(screenshare=P(.85), remote_app=P(.9), new_payee=P(.8), payee_age=U(1, 60), urgency_word=P(.5))
        elif c == "fake_kyc":
            mode = "link"; amt = U(1, 10) if P(.5) else U(500, 15000)
            f.update(kyc_word=P(.9), urgency_word=P(.8), new_payee=P(.9), payee_age=U(1, 30), name_mismatch=U(.3, .9) if P(.5) else U(0, .2))
        elif c == "mule_account":
            amt = U(2000, 40000); f.update(fan_in=U(15, 80), payee_age=U(1, 14), complaints=float(r.poisson(3) + 1), new_payee=P(.7), night=P(.4))
        f.update(amount_log=float(np.log1p(amt)), amount_ratio=ratio, collect=float(mode == "collect"), qr=float(mode == "qr"), link=float(mode == "link"))
        X.append([float(f[k]) for k in FEATS]); y.append(0 if c == "legit" else 1); C.append(c)
    return np.array(X), np.array(y), np.array(C)


def mets(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return dict(precision=round(tp / max(tp + fp, 1), 4), recall=round(tp / max(tp + fn, 1), 4), fpr=round(fp / max(fp + tn, 1), 4),
                accuracy=round((tp + tn) / len(y), 4), tp=tp, fp=fp, fn=fn, tn=tn)


def train(feedback=()):
    """feedback: iterable of (feature_dict, 'fraud'|'legit') labelled by analysts; weighted x15."""
    feedback = list(feedback)
    X, y, C = synth(8000, 1)
    Xa, Xt, ya, yt, Ca, Ct = train_test_split(X, y, C, test_size=.25, stratify=y, random_state=7)
    w = np.ones(len(ya))
    if feedback:
        Xa = np.vstack([Xa, [[float(f.get(k, 0)) for k in FEATS] for f, _ in feedback]])
        ya = np.concatenate([ya, [1 if l == "fraud" else 0 for _, l in feedback]]); w = np.concatenate([w, np.full(len(feedback), 15.)])
    clf = RandomForestClassifier(200, min_samples_leaf=2, class_weight="balanced", random_state=7, n_jobs=-1).fit(Xa, ya, sample_weight=w)
    p = clf.predict_proba(Xt)[:, 1]; pred = (p >= .4).astype(int)
    ai = {FEATS[i]: i for i in range(len(FEATS))}
    amt = np.expm1(Xt[:, ai["amount_log"]])
    base_pred = ((amt > 10000) | ((Xt[:, ai["new_payee"]] == 1) & (amt > 5000))).astype(int)
    m = dict(test_size=len(yt), fraud_in_test=int(yt.sum()), threshold=.4, model=mets(yt, pred), static_rule_baseline=mets(yt, base_pred),
             per_category_recall={k: round(float(pred[Ct == k].mean()), 3) for k in CATS if k in set(Ct) and k != "legit"},
             feedback_samples=len(feedback), trained_at=datetime.now().isoformat(timespec="seconds"))
    M.update(clf=clf, base=np.median(Xa[ya == 0], axis=0), metrics=m)
    return m


def score(f):
    """Risk probability plus per-feature attribution (swap each feature to the legit median, measure the drop)."""
    x = np.array([[float(f[k]) for k in FEATS]]); Z = [x[0]]
    for i in range(len(FEATS)):
        z = x[0].copy(); z[i] = M["base"][i]; Z.append(z)
    pr = M["clf"].predict_proba(np.array(Z))[:, 1]; rows = []
    for i, k in enumerate(FEATS):
        d = float(pr[0] - pr[i + 1])
        if d > .01: rows.append(dict(feature=k, label=FL[k], impact=round(d, 3), value=round(float(x[0, i]), 2)))
    return float(pr[0]), sorted(rows, key=lambda r: -r["impact"])[:6]


# ---------------------------------------------------------------- context analysis
REF = re.compile(r"refund|cashback|reversal|money back|reward|prize|lottery", re.I)
KYC = re.compile(r"kyc|account.{0,20}(block|suspend|freez|deactiv)|update.{0,15}(pan|aadhaar|aadhar)|verification fee|sim.{0,10}block", re.I)
URG = re.compile(r"urgent|immediately|today|tonight|last chance|expire|within \d+|final notice|disconnect", re.I)


def featurize(x):
    """x: amount (INR), mode, registered_name, display_name, message, known_payee, avg_amount, payee_age,
    complaints, fan_in, screenshare, remote_app, hour (optional)."""
    amt = float(x["amount"]); txt = x.get("message") or ""; mode = x.get("mode", "pay")
    reg = (x.get("registered_name") or "").strip().lower(); disp = (x.get("display_name") or reg).strip().lower()
    mism = 1 - difflib.SequenceMatcher(None, reg, disp).ratio() if reg and disp else 0
    hr = int(x["hour"]) if x.get("hour") is not None else datetime.now().hour
    return dict(amount_log=float(np.log1p(amt)), night=float(hr < 5 or hr >= 23),
                amount_ratio=min(amt / max(float(x.get("avg_amount") or 1500), 1), 30), new_payee=float(not x.get("known_payee")),
                payee_age=float(x.get("payee_age", 3)), complaints=float(x.get("complaints", 0)), fan_in=float(x.get("fan_in", 1)),
                collect=float(mode == "collect"), qr=float(mode == "qr"), link=float(mode == "link"),
                refund_word=float(bool(REF.search(txt))), kyc_word=float(bool(KYC.search(txt))), urgency_word=float(bool(URG.search(txt))),
                name_mismatch=float(mism), screenshare=float(bool(x.get("screenshare"))), remote_app=float(bool(x.get("remote_app"))))


def indicators(f, d):
    I = []; a = lambda t, s: I.append(dict(text=t, severity=s)); amt = d["amount"]
    if f["collect"]: a(f"COLLECT request: entering your PIN sends ₹{amt:,.0f} OUT of your account.", "high" if f["refund_word"] else "medium")
    if f["collect"] and f["refund_word"]: a("'Refund' wording on a debit request: real refunds never need your PIN.", "high")
    if f["name_mismatch"] > .4: a(f"Name shown ('{d.get('display_name')}') does not match account holder ('{d.get('registered_name')}').", "high")
    if f["screenshare"] or f["remote_app"]: a("Screen-sharing / remote-access app detected during payment.", "high")
    if f["kyc_word"]: a("Message uses KYC / account-block wording typical of fake bank alerts.", "high")
    if f["urgency_word"]: a("Message pressures you to act urgently.", "medium")
    if f["fan_in"] >= 15 and f["payee_age"] < 30: a(f"Payee is {f['payee_age']:.0f} days old but received from {f['fan_in']:.0f} senders: mule-account pattern.", "high")
    if f["complaints"] >= 2: a(f"Payee has {f['complaints']:.0f} fraud complaints on record.", "high")
    if f["new_payee"] and f["amount_ratio"] > 4: a("First payment to this payee and well above your usual amount.", "medium")
    if f["night"]: a("Payment at an unusual hour.", "low")
    return I


def grok(ctx):
    if not KEY: return None
    sysm = ("You are a UPI fraud analyst for India. Given a payment context, an ML risk score and rule indicators, decide whether it is a scam. "
            "Known categories: collect_refund, qr_swap, remote_access, fake_kyc, mule_account, fake_job_investment, none. If it is a scam fitting none of these, "
            "use category 'novel' with new_pattern_name (snake_case) and 3-6 lowercase keywords from the message that characterise it. "
            'Reply ONLY with JSON: {"fraud_confidence":0-1,"category":str,"new_pattern_name":str|null,"keywords":[str],'
            '"user_warning":"one short sentence to the payer starting with the key fact","explanation":"max 2 plain sentences citing concrete indicators"}')
    body = {"model": GM, "temperature": .1, "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": json.dumps(ctx)}]}
    rq = urllib.request.Request("https://api.x.ai/v1/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json", "Authorization": "Bearer " + KEY, "User-Agent": "upi-guard/2.0"})
    try:
        out = json.load(urllib.request.urlopen(rq, timeout=25))["choices"][0]["message"]["content"]
        return json.loads(re.search(r"\{.*\}", out, re.S).group())
    except Exception as e:
        return {"error": str(e)[:160]}


def assess(x, patterns=(), use_ai=True):
    """Returns the full assessment. `patterns` = approved analyst patterns [{id,name,keywords:[...]}].
    Result carries `learned_ids` (patterns that matched) and `novel` (a new pattern Grok proposes) for the caller to persist."""
    f = featurize(x); p, contrib = score(f); ind = indicators(f, x); txt = (x.get("message") or "").lower()
    learned = []
    for r in patterns:
        kws = r["keywords"]; need = 1 if len(kws) < 2 else 2
        if kws and sum(k in txt for k in kws) >= need:
            learned.append(r)
            ind.insert(0, dict(text=f"Matches analyst-approved pattern '{r['name']}' learned from earlier cases.", severity="high"))
    g = None
    if use_ai and (p >= .2 or txt.strip() or ind):
        g = grok(dict(amount_inr=x["amount"], mode=x.get("mode"), payee_vpa=x.get("payee_vpa"), registered_name=x.get("registered_name"),
                      display_name=x.get("display_name"), message=x.get("message"), screen_sharing=bool(x.get("screenshare")),
                      remote_app=bool(x.get("remote_app")), ml_risk=round(p, 3), indicators=[i["text"] for i in ind]))
    gc = float(g["fraud_confidence"]) if g and "fraud_confidence" in g else None
    risk = p if gc is None else .6 * p + .4 * gc
    if learned: risk = max(risk, .75)
    dec = "BLOCK" if risk >= .7 else "WARN" if risk >= .4 else "ALLOW"
    cat = "none"
    if dec != "ALLOW":
        if learned: cat = learned[0]["name"]
        elif f["screenshare"] or (f["remote_app"] and x["amount"] > 5000): cat = "remote_access"
        elif f["collect"] and (f["refund_word"] or f["complaints"] >= 2): cat = "collect_refund"
        elif f["qr"] and f["name_mismatch"] > .4: cat = "qr_swap"
        elif f["kyc_word"] and (f["link"] or f["urgency_word"]): cat = "fake_kyc"
        elif f["fan_in"] >= 15 and f["payee_age"] < 30: cat = "mule_account"
        elif g and g.get("category") not in (None, "none", "novel"): cat = g["category"]
        elif g and g.get("category") == "novel" and g.get("new_pattern_name"): cat = g["new_pattern_name"]
        else: cat = "suspicious_other"
    novel = None
    if g and g.get("category") == "novel" and gc and gc >= .6 and g.get("new_pattern_name"):
        novel = dict(name=g["new_pattern_name"], description=g.get("explanation", ""), keywords=[k.lower() for k in g.get("keywords", [])])
    amt = x["amount"]
    head = (f"You are about to PAY, not receive, ₹{amt:,.0f}" if f["collect"] and dec != "ALLOW" else
            "Screen sharing detected while paying" if cat == "remote_access" else
            f"QR shows '{x.get('display_name')}' but pays '{x.get('registered_name')}'" if cat == "qr_swap" else
            "This looks like a scam" if dec != "ALLOW" else "No fraud signals found")
    warn = (g or {}).get("user_warning") or (ind[0]["text"] if ind and dec != "ALLOW" else "Payment looks consistent with your normal behaviour.")
    expl = (g or {}).get("explanation") or (" ".join(i["text"] for i in ind[:3]) if ind else "Amount, payee and context are all normal.")
    return dict(risk=round(risk, 3), model_risk=round(p, 3), ai_confidence=gc, decision=dec, category=cat,
                category_label=CATS.get(cat, cat.replace("_", " ").title()), headline=head, warning=warn, explanation=expl,
                indicators=ind, contributions=contrib, ai_used=bool(g and "error" not in g),
                ai_error=(g or {}).get("error") if g else ("No XAI_API_KEY set: running ML + rules only" if use_ai else None),
                features=f, learned_ids=[r["id"] for r in learned], novel=novel)


# ============================================================================
# 3. API (auth, wallet, payment lifecycle, analyst endpoints, seed data)
# ============================================================================

DEMO = os.getenv("DEMO_MODE", "1") == "1"
PER_TXN_LIMIT = 100000 * 100  # paise
MAX_PIN_FAILS, LOCK_MIN = 3, 15


def secret_key():
    if os.getenv("SECRET_KEY"): return os.environ["SECRET_KEY"]
    p = os.path.join(HERE, ".secret_key")
    if not os.path.exists(p): open(p, "w").write(secrets.token_hex(32))
    return open(p).read().strip()


app = Flask(__name__, static_folder=None)  # never serve the project directory (db, source, .env)
app.config.update(SECRET_KEY=secret_key(), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=timedelta(hours=8), JSON_SORT_KEYS=False)


class Fail(Exception):
    def __init__(self, msg, code=400, **extra): self.msg, self.code, self.extra = msg, code, extra


@app.errorhandler(Fail)
def _fail(e): return jsonify(error=e.msg, **e.extra), e.code


@app.errorhandler(404)
def _404(e): return jsonify(error="Not found"), 404


@app.errorhandler(500)
def _500(e): return jsonify(error="Something went wrong on our side. Please try again."), 500


@app.before_request
def _guard():
    # JSON-only POSTs: a cross-site <form> cannot send application/json, which blocks CSRF without tokens
    if request.method == "POST" and request.path.startswith("/api/") and request.mimetype != "application/json":
        return jsonify(error="Content-Type must be application/json"), 415


@app.after_request
def _headers(r):
    r.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "same-origin",
                      "Cache-Control": "no-store" if request.path.startswith("/api/") else "no-cache"})
    return r


# ---------------------------------------------------------------- helpers
now = lambda: datetime.now().isoformat(timespec="seconds")
rs = lambda p: round((p or 0) / 100, 2)
js = lambda s, d=None: json.loads(s) if s else d
HITS = defaultdict(deque)


def limit(key, n, secs):
    q = HITS[key]; t = time.time()
    while q and q[0] < t - secs: q.popleft()
    if len(q) >= n: raise Fail("Too many attempts. Please wait a moment and try again.", 429)
    q.append(t)


def audit(c, uid, event, detail=""):
    c.execute("insert into audit(ts,user_id,event,detail) values(?,?,?,?)", (now(), uid, event, str(detail)[:300]))


def login_required(role=None):
    def deco(fn):
        def w(*a, **k):
            uid = session.get("uid")
            if not uid: raise Fail("Please sign in to continue.", 401)
            c = db_conn(); u = c.execute("select * from users where id=?", (uid,)).fetchone(); c.close()
            if not u: session.clear(); raise Fail("Please sign in to continue.", 401)
            if role and u["role"] != role: raise Fail("You do not have access to this area.", 403)
            g.user = u; return fn(*a, **k)
        w.__name__ = fn.__name__; return w
    return deco


def body(): return request.get_json(silent=True) or {}


def user_out(u, c):
    day = datetime.now().strftime("%Y-%m-%d")
    used = c.execute("select coalesce(sum(amount),0) s from payments where user_id=? and status='SUCCESS' and source='live' and substr(updated,1,10)=?",
                     (u["id"], day)).fetchone()["s"]
    pend = c.execute("select count(*) n from collect_requests where payer_vpa=? and status='PENDING'", (u["vpa"],)).fetchone()["n"]
    return dict(id=u["id"], name=u["name"], phone=u["phone"], vpa=u["vpa"], role=u["role"], balance=rs(u["balance"]), pin_len=u["pin_len"],
                daily_limit=rs(u["daily_limit"]), daily_used=rs(used), kyc=u["kyc"], pending_requests=pend, demo=DEMO)


def profile(c, vpa):
    """What the network knows about a UPI ID: verified name, account age, complaints, distinct senders."""
    u = c.execute("select id,name,created from users where vpa=?", (vpa,)).fetchone()
    r = c.execute("select * from reputation where vpa=?", (vpa,)).fetchone()
    if not u and not r: return None
    src = u or r; age = max((datetime.now() - datetime.fromisoformat(src["created"])).total_seconds() / 86400, 0)
    senders = c.execute("select count(distinct user_id) n from payments where payee_vpa=? and status='SUCCESS' and source='live'", (vpa,)).fetchone()["n"]
    return dict(name=src["name"], age_days=age, complaints=r["complaints"] if r else 0,
                fan_in=(r["fan_in_base"] if r else 0) + senders, internal=bool(u), user_id=u["id"] if u else None)


def pay_out(r, viewer):
    out = r["user_id"] == viewer
    return dict(id=r["id"], ref=r["ref"], direction="out" if out else "in", amount=rs(r["amount"]), mode=r["mode"], note=r["note"],
                status=r["status"], decision=r["decision"], risk=r["risk"], category=r["category"],
                category_label=CATS.get(r["category"], (r["category"] or "").replace("_", " ").title()),
                payee_vpa=r["payee_vpa"], payee_name=r["payee_name"], display_name=r["display_name"],
                counterparty_name=r["payee_name"] if out else r["sender_name"], counterparty_vpa=r["payee_vpa"] if out else r["sender_vpa"],
                created=r["created"], updated=r["updated"])


PAY_Q = "select p.*, s.name sender_name, s.vpa sender_vpa from payments p left join users s on s.id=p.user_id"


# ---------------------------------------------------------------- pages
@app.get("/")
def home(): return send_from_directory(HERE, "index.html")


# ---------------------------------------------------------------- auth
@app.post("/api/auth/register")
def register():
    d = body(); name = (d.get("name") or "").strip(); phone = re.sub(r"\D", "", d.get("phone") or "")[-10:]
    pw, pin = d.get("password") or "", str(d.get("pin") or "")
    if not 2 <= len(name) <= 60: raise Fail("Enter your full name.")
    if not re.fullmatch(r"[6-9]\d{9}", phone): raise Fail("Enter a valid 10-digit Indian mobile number.")
    if len(pw) < 8: raise Fail("Password must be at least 8 characters.")
    if not re.fullmatch(r"\d{4}|\d{6}", pin): raise Fail("UPI PIN must be 4 or 6 digits.")
    limit("reg:" + request.remote_addr, 10, 600)
    vpa = f"{phone}@upiguard"
    with db_tx() as c:
        if c.execute("select 1 from users where phone=?", (phone,)).fetchone(): raise Fail("This mobile number is already registered. Sign in instead.", 409)
        cur = c.execute("insert into users(name,phone,email,vpa,pw_hash,pin_hash,pin_len,created) values(?,?,?,?,?,?,?,?)",
                        (name, phone, (d.get("email") or "").strip() or None, vpa, generate_password_hash(pw), generate_password_hash(pin), len(pin), now()))
        audit(c, cur.lastrowid, "register", vpa); uid = cur.lastrowid
    session.clear(); session["uid"] = uid; session.permanent = True
    c = db_conn(); u = c.execute("select * from users where id=?", (uid,)).fetchone(); out = user_out(u, c); c.close()
    return jsonify(user=out), 201


@app.post("/api/auth/login")
def login():
    d = body(); phone = re.sub(r"\D", "", d.get("phone") or "")[-10:]
    limit(f"login:{request.remote_addr}:{phone}", 8, 300)
    c = db_conn(); u = c.execute("select * from users where phone=?", (phone,)).fetchone()
    if not u or not check_password_hash(u["pw_hash"], d.get("password") or ""):
        c.close(); raise Fail("Mobile number or password is incorrect.", 401)
    audit(c, u["id"], "login", request.remote_addr); out = user_out(u, c); c.close()
    session.clear(); session["uid"] = u["id"]; session.permanent = True
    return jsonify(user=out)


@app.post("/api/auth/logout")
def logout(): session.clear(); return jsonify(ok=True)


@app.get("/api/me")
@login_required()
def me():
    c = db_conn(); out = user_out(g.user, c); c.close(); return jsonify(user=out)


# ---------------------------------------------------------------- wallet
@app.post("/api/wallet/topup")
@login_required()
def topup():
    """Sandbox for 'add money from linked bank account'. A real rail would redirect to the PSP's collect/checkout flow."""
    try: p = round(float(body().get("amount")) * 100)
    except Exception: raise Fail("Enter an amount.")
    if not 100 <= p <= 5000000: raise Fail("You can add between ₹1 and ₹50,000 at a time.")
    with db_tx() as c:
        u = c.execute("select balance from users where id=?", (g.user["id"],)).fetchone(); nb = u["balance"] + p
        c.execute("update users set balance=? where id=?", (nb, g.user["id"]))
        c.execute("insert into ledger(user_id,kind,delta,balance_after,ts) values(?,?,?,?,?)", (g.user["id"], "topup", p, nb, now()))
        audit(c, g.user["id"], "topup", rs(p))
    return jsonify(balance=rs(nb))


@app.get("/api/vpa/resolve")
@login_required()
def resolve():
    limit(f"resolve:{g.user['id']}", 40, 60)
    vpa = (request.args.get("vpa") or "").strip().lower(); c = db_conn(); pr = profile(c, vpa); c.close()
    if not pr: raise Fail("This UPI ID does not exist. Check it and try again.", 404)
    return jsonify(vpa=vpa, name=pr["name"], verified=True)


@app.get("/api/beneficiaries")
@login_required()
def beneficiaries():
    c = db_conn(); r = [dict(x) for x in c.execute("select vpa,name from beneficiaries where user_id=? order by created desc limit 8", (g.user["id"],))]
    c.close(); return jsonify(r)


# ---------------------------------------------------------------- payments
@app.post("/api/payments/check")
@login_required()
def pay_check():
    d = body(); u = g.user; limit(f"check:{u['id']}", 30, 60)
    mode = d.get("mode", "pay")
    if mode not in ("pay", "qr", "link", "collect"): raise Fail("Unknown payment type.")
    coll = None
    c = db_conn()
    try:
        if d.get("collect_id"):
            coll = c.execute("select cr.*, r.vpa rvpa from collect_requests cr join users r on r.id=cr.requester_id where cr.id=?", (d["collect_id"],)).fetchone()
            if not coll or coll["payer_vpa"] != u["vpa"] or coll["status"] != "PENDING": raise Fail("This request is no longer available.", 404)
            mode, vpa, paise, note, disp = "collect", coll["rvpa"], coll["amount"], coll["note"], coll["display_name"]
        else:
            vpa = (d.get("payee_vpa") or "").strip().lower(); note = (d.get("message") or "").strip()[:300]; disp = (d.get("display_name") or "").strip()[:80]
            try: paise = round(float(d.get("amount")) * 100)
            except Exception: raise Fail("Enter a valid amount.")
        if paise <= 0: raise Fail("Enter an amount greater than ₹0.")
        if paise > PER_TXN_LIMIT: raise Fail("UPI allows up to ₹1,00,000 per payment.")
        if vpa == u["vpa"]: raise Fail("You cannot pay your own UPI ID.")
        pr = profile(c, vpa)
        if not pr: raise Fail("This UPI ID does not exist. Check it and try again.", 404)
        known = c.execute("select count(*) n from payments where user_id=? and payee_vpa=? and status='SUCCESS'", (u["id"], vpa)).fetchone()["n"]
        avg = c.execute("select avg(amount)/100.0 a from payments where user_id=? and status='SUCCESS'", (u["id"],)).fetchone()["a"] or 1500
        pats = [dict(id=r["id"], name=r["name"], keywords=json.loads(r["keywords"])) for r in c.execute("select * from patterns where status='approved'")]
        ctx = dict(amount=paise / 100, mode=mode, payee_vpa=vpa, registered_name=pr["name"], display_name=disp or pr["name"], message=note,
                   known_payee=known > 0, avg_amount=avg, payee_age=pr["age_days"], complaints=pr["complaints"], fan_in=pr["fan_in"],
                   screenshare=bool(d.get("screenshare")), remote_app=bool(d.get("remote_app")))
        res = assess(ctx, pats)
    finally:
        c.close()
    with db_tx() as c:
        for i in res["learned_ids"]: c.execute("update patterns set hits=hits+1 where id=?", (i,))
        n = res["novel"]
        if n and not c.execute("select 1 from patterns where name=?", (n["name"],)).fetchone():
            c.execute("insert into patterns(name,description,keywords,status,created) values(?,?,?,?,?)", (n["name"], n["description"], json.dumps(n["keywords"]), "pending", now()))
        else: res["novel"] = None
        ref = "".join(random.choices("0123456789", k=12)); t = now()
        cur = c.execute("""insert into payments(ref,user_id,payee_vpa,payee_name,display_name,amount,mode,note,status,risk,model_risk,decision,category,
                           features,reasons,contrib,ai,collect_id,created,updated) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (ref, u["id"], vpa, pr["name"], disp or None, paise, mode, note, "CHECKED", res["risk"], res["model_risk"], res["decision"], res["category"],
                         json.dumps(res["features"]), json.dumps(res["indicators"]), json.dumps(res["contributions"]),
                         json.dumps(dict(warning=res["warning"], explanation=res["explanation"], ai_used=res["ai_used"])), coll["id"] if coll else None, t, t))
        audit(c, u["id"], "payment_checked", f"#{cur.lastrowid} {res['decision']} {res['category']}")
    for k in ("features", "learned_ids"): res.pop(k)
    res.update(payment_id=cur.lastrowid, ref=ref, payee_name=pr["name"], payee_vpa=vpa, amount=paise / 100, new_pattern=(res.pop("novel") or {}).get("name"))
    return jsonify(res)


def settle(c, pid, expect_user=None):
    """Move money. Runs inside db_tx(): all-or-nothing. Debit, credit and ledger rows commit together."""
    p = c.execute("select * from payments where id=?", (pid,)).fetchone()
    if not p or (expect_user and p["user_id"] != expect_user): raise Fail("Payment not found.", 404)
    if p["status"] not in ("CHECKED", "UNDER_REVIEW"): raise Fail("This payment has already been completed or cancelled.", 409)
    u = c.execute("select * from users where id=?", (p["user_id"],)).fetchone(); amt = p["amount"]
    used = c.execute("select coalesce(sum(amount),0) s from payments where user_id=? and status='SUCCESS' and source='live' and substr(updated,1,10)=?",
                     (u["id"], datetime.now().strftime("%Y-%m-%d"))).fetchone()["s"]
    if used + amt > u["daily_limit"]: raise Fail(f"Daily limit of ₹{rs(u['daily_limit']):,.0f} would be exceeded.", 400)
    if u["balance"] < amt: raise Fail(f"Insufficient balance. You have ₹{rs(u['balance']):,.2f}.", 400)
    t = now(); nb = u["balance"] - amt
    c.execute("update users set balance=? where id=?", (nb, u["id"]))
    c.execute("insert into ledger(payment_id,user_id,kind,delta,balance_after,ts) values(?,?,?,?,?,?)", (pid, u["id"], "debit", -amt, nb, t))
    rc = c.execute("select id,balance from users where vpa=?", (p["payee_vpa"],)).fetchone()
    if rc:
        rb = rc["balance"] + amt; c.execute("update users set balance=? where id=?", (rb, rc["id"]))
        c.execute("insert into ledger(payment_id,user_id,kind,delta,balance_after,ts) values(?,?,?,?,?,?)", (pid, rc["id"], "credit", amt, rb, t))
    else:  # payee is at another bank: hand off to the (sandbox) external rail
        c.execute("insert into ledger(payment_id,user_id,kind,delta,balance_after,ts) values(?,?,?,?,?,?)", (pid, None, "external_credit", amt, None, t))
    c.execute("update payments set status='SUCCESS', updated=? where id=?", (t, pid))
    c.execute("insert into beneficiaries(user_id,vpa,name,created) values(?,?,?,?) on conflict(user_id,vpa) do update set created=excluded.created",
              (u["id"], p["payee_vpa"], p["payee_name"], t))
    if p["collect_id"]: c.execute("update collect_requests set status='PAID' where id=?", (p["collect_id"],))
    return nb


@app.post("/api/payments/<int:pid>/confirm")
@login_required()
def confirm(pid):
    d = body(); u = g.user; pin = str(d.get("pin") or "")
    c = db_conn(); p = c.execute("select * from payments where id=? and user_id=?", (pid, u["id"])).fetchone()
    if not p: c.close(); raise Fail("Payment not found.", 404)
    if p["status"] != "CHECKED": c.close(); raise Fail("This payment has already been completed or cancelled.", 409)
    if p["decision"] == "BLOCK":
        c.close(); raise Fail("This payment was blocked as high risk. You can send it for review or cancel it.", 403, hold=True)
    if p["decision"] == "WARN" and not d.get("acknowledge"): c.close(); raise Fail("Please confirm that you understand the warning.", 400)
    lock = u["locked_until"]
    if lock and datetime.fromisoformat(lock) > datetime.now():
        mins = int((datetime.fromisoformat(lock) - datetime.now()).total_seconds() // 60) + 1
        c.close(); raise Fail(f"UPI PIN locked after too many wrong attempts. Try again in {mins} min.", 423)
    if not check_password_hash(u["pin_hash"], pin):
        f = u["pin_fails"] + 1; lk = (datetime.now() + timedelta(minutes=LOCK_MIN)).isoformat(timespec="seconds") if f >= MAX_PIN_FAILS else None
        c.execute("update users set pin_fails=?, locked_until=? where id=?", (0 if lk else f, lk, u["id"])); audit(c, u["id"], "pin_failed", f"payment {pid}"); c.close()
        raise Fail(f"Locked for {LOCK_MIN} minutes after {MAX_PIN_FAILS} wrong attempts." if lk else f"Incorrect UPI PIN. {MAX_PIN_FAILS - f} attempt(s) left.", 422)
    c.execute("update users set pin_fails=0, locked_until=null where id=?", (u["id"],)); c.close()
    with db_tx() as c:
        c.execute("update payments set user_ack=? where id=?", (1 if d.get("acknowledge") else 0, pid))
        nb = settle(c, pid, u["id"]); audit(c, u["id"], "payment_success", f"#{pid} {rs(p['amount'])}")
        r = c.execute(PAY_Q + " where p.id=?", (pid,)).fetchone()
    return jsonify(payment=pay_out(r, u["id"]), balance=rs(nb))


@app.post("/api/payments/<int:pid>/cancel")
@login_required()
def cancel(pid):
    with db_tx() as c:
        n = c.execute("update payments set status='CANCELLED', updated=? where id=? and user_id=? and status in('CHECKED','UNDER_REVIEW')", (now(), pid, g.user["id"])).rowcount
        if not n: raise Fail("This payment cannot be cancelled.", 409)
        audit(c, g.user["id"], "payment_cancelled", pid)
    return jsonify(ok=True)


@app.post("/api/payments/<int:pid>/review")
@login_required()
def send_review(pid):
    """A blocked payment is parked for a human analyst instead of being sent."""
    with db_tx() as c:
        n = c.execute("update payments set status='UNDER_REVIEW', updated=? where id=? and user_id=? and status='CHECKED' and decision='BLOCK'", (now(), pid, g.user["id"])).rowcount
        if not n: raise Fail("Only blocked payments can be sent for review.", 409)
        audit(c, g.user["id"], "sent_for_review", pid)
    return jsonify(ok=True)


@app.post("/api/payments/<int:pid>/report")
@login_required()
def report(pid):
    with db_tx() as c:
        p = c.execute("select * from payments where id=? and user_id=?", (pid, g.user["id"])).fetchone()
        if not p: raise Fail("Payment not found.", 404)
        if c.execute("select 1 from reports where reporter_id=? and vpa=?", (g.user["id"], p["payee_vpa"])).fetchone(): return jsonify(ok=True, duplicate=True)
        c.execute("insert into reports(reporter_id,vpa,payment_id,reason,ts) values(?,?,?,?,?)", (g.user["id"], p["payee_vpa"], pid, (body().get("reason") or "scam")[:200], now()))
        c.execute("insert into reputation(vpa,name,complaints,created) values(?,?,1,?) on conflict(vpa) do update set complaints=complaints+1", (p["payee_vpa"], p["payee_name"], now()))
        audit(c, g.user["id"], "reported_vpa", p["payee_vpa"])
    return jsonify(ok=True)


@app.get("/api/payments")
@login_required()
def history():
    c = db_conn(); u = g.user
    rows = c.execute(PAY_Q + " where (p.user_id=? and p.status!='CHECKED') or (p.payee_vpa=? and p.status='SUCCESS') order by p.updated desc limit 60", (u["id"], u["vpa"])).fetchall()
    out = [pay_out(r, u["id"]) for r in rows]; c.close(); return jsonify(out)


@app.get("/api/payments/<int:pid>")
@login_required()
def one(pid):
    c = db_conn(); u = g.user
    r = c.execute(PAY_Q + " where p.id=? and (p.user_id=? or p.payee_vpa=?)", (pid, u["id"], u["vpa"])).fetchone(); c.close()
    if not r: raise Fail("Payment not found.", 404)
    return jsonify(pay_out(r, u["id"]))


# ---------------------------------------------------------------- collect requests (the vector behind "refund" scams)
@app.post("/api/collect")
@login_required()
def collect_create():
    d = body(); limit(f"collect:{g.user['id']}", 6, 60)
    try: p = round(float(d.get("amount")) * 100)
    except Exception: raise Fail("Enter a valid amount.")
    if not 100 <= p <= PER_TXN_LIMIT: raise Fail("Amount must be between ₹1 and ₹1,00,000.")
    vpa = (d.get("payer_vpa") or "").strip().lower()
    with db_tx() as c:
        if not c.execute("select 1 from users where vpa=?", (vpa,)).fetchone(): raise Fail("No payer found with this UPI ID.", 404)
        if vpa == g.user["vpa"]: raise Fail("You cannot request money from yourself.")
        cur = c.execute("insert into collect_requests(requester_id,payer_vpa,display_name,amount,note,created) values(?,?,?,?,?,?)",
                        (g.user["id"], vpa, (d.get("display_name") or g.user["name"])[:80], p, (d.get("note") or "")[:300], now()))
        audit(c, g.user["id"], "collect_created", f"#{cur.lastrowid} to {vpa}")
    return jsonify(id=cur.lastrowid), 201


@app.get("/api/collect/incoming")
@login_required()
def collect_in():
    c = db_conn()
    r = [dict(id=x["id"], amount=rs(x["amount"]), note=x["note"], display_name=x["display_name"], vpa=x["vpa"], name=x["name"], created=x["created"])
         for x in c.execute("select cr.*, r.vpa vpa, r.name name from collect_requests cr join users r on r.id=cr.requester_id where cr.payer_vpa=? and cr.status='PENDING' order by cr.id desc", (g.user["vpa"],))]
    c.close(); return jsonify(r)


@app.post("/api/collect/<int:i>/decline")
@login_required()
def collect_decline(i):
    with db_tx() as c: c.execute("update collect_requests set status='DECLINED' where id=? and payer_vpa=? and status='PENDING'", (i, g.user["vpa"]))
    return jsonify(ok=True)


# ---------------------------------------------------------------- demo helpers (disabled when DEMO_MODE=0)
def demo_only():
    if not DEMO: raise Fail("Not found", 404)


@app.post("/api/demo/scam-collect")
@login_required()
def demo_scam():
    """Plays the attacker: a 'refund' collect request lands in the signed-in user's inbox."""
    demo_only()
    with db_tx() as c:
        a = c.execute("select id from users where vpa='refund.help4u@ybl'").fetchone()
        c.execute("insert into collect_requests(requester_id,payer_vpa,display_name,amount,note,created) values(?,?,?,?,?,?)",
                  (a["id"], g.user["vpa"], "Amazon Refund Desk", 500000, "Your order refund of ₹5000 is ready. Approve the request and enter UPI PIN to receive the money.", now()))
    return jsonify(ok=True)


@app.post("/api/demo/reset")
@login_required()
def demo_reset():
    demo_only()
    with db_tx() as c:
        cur = c.execute("select balance from users where id=?", (g.user["id"],)).fetchone()["balance"]; tgt = 2500000
        c.execute("update users set balance=?, pin_fails=0, locked_until=null where id=?", (tgt, g.user["id"]))
        c.execute("insert into ledger(user_id,kind,delta,balance_after,ts) values(?,?,?,?,?)", (g.user["id"], "demo_reset", tgt - cur, tgt, now()))
        c.execute("update collect_requests set status='DECLINED' where payer_vpa=? and status='PENDING'", (g.user["vpa"],))
    return jsonify(balance=rs(tgt))


# ---------------------------------------------------------------- analyst console
@app.get("/api/admin/stats")
@login_required("analyst")
def stats():
    c = db_conn(); w = "where source!='history'"; q = lambda s, *a: c.execute(s, a).fetchall()
    t = q(f"select count(*) n,sum(decision!='ALLOW') fl,sum(decision='BLOCK') bl,sum(case when decision!='ALLOW' then amount else 0 end) val from payments {w}")[0]
    prot = q(f"select coalesce(sum(amount),0) s from payments {w} and decision!='ALLOW' and status in('CANCELLED','REJECTED','UNDER_REVIEW')")[0]["s"]
    cats = [dict(category=r["category"], label=CATS.get(r["category"], r["category"].replace("_", " ").title()), n=r["n"]) for r in
            q(f"select category,count(*) n from payments {w} and decision!='ALLOW' group by category order by n desc")]
    tr = [dict(day=r["d"], total=r["n"], flagged=r["fl"]) for r in q(f"select substr(created,1,10) d,count(*) n,sum(decision!='ALLOW') fl from payments {w} group by d order by d desc limit 7")][::-1]
    out = dict(total=t["n"], flagged=t["fl"] or 0, blocked=t["bl"] or 0, value_at_risk=rs(t["val"]), value_protected=rs(prot), categories=cats, trend=tr,
               pending_patterns=q("select count(*) n from patterns where status='pending'")[0]["n"],
               under_review=q("select count(*) n from payments where status='UNDER_REVIEW'")[0]["n"],
               users=q("select count(*) n from users")[0]["n"], settled=rs(q("select coalesce(sum(amount),0) s from payments where status='SUCCESS' and source='live'")[0]["s"]),
               ai_enabled=bool(KEY), ai_model=GM)
    c.close(); return jsonify(out)


@app.get("/api/admin/transactions")
@login_required("analyst")
def admin_txns():
    f = request.args.get("view", "flagged")
    cond = {"flagged": " and p.decision!='ALLOW'", "review": " and p.status='UNDER_REVIEW'"}.get(f, "")
    c = db_conn(); rows = []
    for r in c.execute(PAY_Q + f" where p.source!='history'{cond} order by p.created desc limit 150").fetchall():
        o = pay_out(r, -1); o.update(sender=r["sender_name"] or "Unknown", analyst_label=r["analyst_label"], ai=js(r["ai"], {}), reasons=js(r["reasons"], []))
        rows.append(o)
    c.close(); return jsonify(rows)


@app.post("/api/admin/payments/<int:pid>/resolve")
@login_required("analyst")
def admin_resolve(pid):
    act = body().get("action")
    if act not in ("release", "reject"): raise Fail("Unknown action.")
    with db_tx() as c:
        p = c.execute("select * from payments where id=? and status='UNDER_REVIEW'", (pid,)).fetchone()
        if not p: raise Fail("This payment is not waiting for review.", 409)
        if act == "release": settle(c, pid)
        else: c.execute("update payments set status='REJECTED', updated=? where id=?", (now(), pid))
        audit(c, g.user["id"], f"review_{act}", pid)
    return jsonify(ok=True)


@app.post("/api/admin/payments/<int:pid>/label")
@login_required("analyst")
def admin_label(pid):
    l = body().get("label")
    if l not in ("fraud", "legit"): raise Fail("Label must be fraud or legit.")
    with db_tx() as c: c.execute("update payments set analyst_label=? where id=?", (l, pid))
    return jsonify(ok=True)


@app.get("/api/admin/metrics")
@login_required("analyst")
def metrics(): return jsonify(M["metrics"])


def retrain():
    c = db_conn(); fb = [(json.loads(r["features"]), r["analyst_label"]) for r in c.execute("select features,analyst_label from payments where analyst_label is not null")]
    m = train(fb); c.execute("insert into model_meta(ts,metrics) values(?,?)", (m["trained_at"], json.dumps(m))); c.close(); return m


@app.post("/api/admin/retrain")
@login_required("analyst")
def admin_retrain(): return jsonify(retrain())


@app.get("/api/admin/patterns")
@login_required("analyst")
def patterns():
    c = db_conn(); r = [dict(x, keywords=json.loads(x["keywords"])) for x in c.execute("select * from patterns order by id desc")]; c.close(); return jsonify(r)


@app.post("/api/admin/patterns/<int:i>")
@login_required("analyst")
def pattern_set(i):
    s = body().get("status")
    if s not in ("approved", "rejected"): raise Fail("Unknown status.")
    with db_tx() as c: c.execute("update patterns set status=? where id=?", (s, i))
    return jsonify(ok=True)


@app.get("/api/admin/users")
@login_required("analyst")
def admin_users():
    c = db_conn()
    r = [dict(id=x["id"], name=x["name"], phone=x["phone"], vpa=x["vpa"], role=x["role"], kyc=x["kyc"], balance=rs(x["balance"]), created=x["created"][:10])
         for x in c.execute("select * from users where role!='analyst' order by id desc limit 100")]
    c.close(); return jsonify(r)


# ---------------------------------------------------------------- seed data
def seed():
    c = db_conn()
    if c.execute("select 1 from users").fetchone(): c.close(); return
    nw = datetime.now(); iso = lambda dt: dt.isoformat(timespec="seconds")

    def mk(name, phone, vpa, pw, pin, age, bal=0, role="user"):
        cur = c.execute("insert into users(name,phone,vpa,pw_hash,pin_hash,pin_len,role,balance,created) values(?,?,?,?,?,?,?,?,?)",
                        (name, phone, vpa, generate_password_hash(pw), generate_password_hash(pin), len(pin), role, bal * 100, iso(nw - timedelta(days=age))))
        if bal: c.execute("insert into ledger(user_id,kind,delta,balance_after,ts) values(?,?,?,?,?)", (cur.lastrowid, "opening", bal * 100, bal * 100, iso(nw)))
        return cur.lastrowid
    aarav = mk("Aarav Sharma", "9876500001", "aarav@upiguard", "demo@1234", "1234", 200, 25000)
    mk("Priya Nair", "9876500002", "priya@upiguard", "demo@1234", "4321", 120, 5000)
    mk("Fraud Analyst", "9000000001", "analyst@upiguard", "analyst@1234", "000000", 400, 0, "analyst")
    for nm, vpa, age, cmp_, fan in [("Ramesh Kirana Store", "ramesh.kirana@oksbi", 900, 0, 6), ("Chai Point", "chai.point@icici", 700, 0, 40),
                                    ("Refund Desk", "refund.help4u@ybl", 3, 7, 4), ("Rajesh Kumar", "quickcash.9921@axl", 5, 4, 46),
                                    ("KYC Update Svc", "bsnl-kyc.update@upi", 1, 12, 9), ("Easy Earn Tasks", "easyearn.task77@ibl", 4, 9, 52)]:
        mk(nm, "9" + "".join(random.choices("0123456789", k=9)), vpa, secrets.token_hex(8), "1234", age)
        c.execute("insert into reputation(vpa,name,complaints,fan_in_base,created) values(?,?,?,?,?)", (vpa, nm, cmp_, fan, iso(nw - timedelta(days=age))))
    c.execute("insert into reputation(vpa,name,complaints,fan_in_base,created,external) values(?,?,?,?,?,1)", ("elec-bill.pay@okaxis", "Bill Desk Services", 0, 1, iso(nw - timedelta(days=3))))
    for a in (320, 450, 2100, 600, 380):  # earlier payments by the demo user, so 'usual amount' and 'known payee' are real
        t = iso(nw - timedelta(days=random.randint(1, 20)))
        c.execute("insert into payments(ref,user_id,payee_vpa,payee_name,amount,mode,status,decision,category,source,created,updated,features,reasons,contrib,ai,risk) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  ("".join(random.choices("0123456789", k=12)), aarav, "ramesh.kirana@oksbi", "Ramesh Kirana Store", a * 100, "pay", "SUCCESS", "ALLOW", "none", "history", t, t, "{}", "[]", "[]", "{}", .02))
    X, y, C = synth(160, 99); random.seed(5)
    for i, row in enumerate(X):  # synthetic traffic so the analyst console has something to show on first run
        f = dict(zip(FEATS, map(float, row))); p, _ = score(f); dec = "BLOCK" if p >= .7 else "WARN" if p >= .4 else "ALLOW"
        cat = "none" if dec == "ALLOW" else (C[i] if C[i] != "legit" else "suspicious_other"); amt = float(np.expm1(f["amount_log"]))
        t = iso(nw - timedelta(days=random.randint(0, 6), hours=random.randint(0, 23), minutes=random.randint(0, 59)))
        st = "CANCELLED" if dec != "ALLOW" and random.random() < .6 else "SUCCESS"
        c.execute("insert into payments(ref,payee_vpa,payee_name,amount,mode,status,decision,category,risk,model_risk,features,reasons,contrib,ai,source,created,updated) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  ("".join(random.choices("0123456789", k=12)), f"user{i}@ok{random.choice(['sbi', 'hdfc', 'axis'])}", "Unknown", round(amt * 100),
                   "collect" if f["collect"] else "qr" if f["qr"] else "link" if f["link"] else "pay", st, dec, cat, p, p, json.dumps(f),
                   json.dumps(indicators(f, dict(amount=amt, display_name="", registered_name=""))), "[]", "{}", "seed", t, t))
    c.close()


db_init(); retrain(); seed()
if __name__ == "__main__":
    print("AI:", f"Grok ({GM})" if KEY else "OFF (set XAI_API_KEY)", "| demo mode:", DEMO)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
