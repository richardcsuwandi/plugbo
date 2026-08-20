// plugbo local viewer. Vanilla JS, no build step, no external deps.

const state = {
  runs: [],
  current: null, // detail of the single run being viewed
  activeTab: "overview",
  selected: new Set(), // run names checked for compare
  mode: "empty", // "empty" | "single" | "compare" | "experiment"
  sidebarMode: "runs", // "runs" | "experiments"
  compareGroups: [],
  compareGroupsLoading: true,
  compareGroupsError: null,
  currentGroup: null, // name of selected comparison group
  groupDetail: null,
  pendingExperiments: false,
  filters: {
    q: "",
  },
};

const CAT_COLORS = ["--cat-1", "--cat-2", "--cat-3", "--cat-4", "--cat-5", "--cat-6", "--cat-7", "--cat-8"];
const BACKEND_COLORS = {
  vanilla: "--cat-4",
  cake: "--cat-1",
  turbo: "--cat-6",
  pibo: "--cat-2",
  llambo: "--cat-8",
  "cake-turbo": "--cat-1",
  "cake-turbo-pibo": "--cat-2",
  "sara-lenz": "--cat-3",
  "sara-lenz-cake": "--cat-7",
  "sara-cake": "--cat-7",
  "sara-only": "--cat-5",
};

// -- theme -------------------------------------------------------------

function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.dataset.theme = saved;
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme;
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
    if (state.mode === "single" && state.current) renderRun(state.current);
    if (state.mode === "compare") renderCompare();
    if (state.mode === "experiment" && state.groupDetail) renderExperimentCompare(state.groupDetail);
  });
}

// -- helpers -------------------------------------------------------------

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtNum(n, digits = 4) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (typeof n !== "number") return String(n);
  if (n === 0) return "0";
  if (Math.abs(n) >= 1e5 || Math.abs(n) < 1e-3) return n.toExponential(2);
  let s = n.toFixed(digits);
  if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
  return s;
}

function hashString(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function preferredColorVar(name) {
  const key = String(name || "");
  if (BACKEND_COLORS[key]) return BACKEND_COLORS[key];
  const leaf = key.split("/").filter(Boolean).pop();
  return BACKEND_COLORS[leaf] || null;
}

function colorMapFor(names) {
  const unique = [];
  const seen = new Set();
  for (const name of names) {
    if (!name || seen.has(name)) continue;
    seen.add(name);
    unique.push(name);
  }
  const assigned = new Map();
  const used = new Set();
  for (const name of unique) {
    const pref = preferredColorVar(name);
    if (pref && !used.has(pref)) {
      used.add(pref);
      assigned.set(name, pref);
    }
  }
  for (const name of unique) {
    if (assigned.has(name)) continue;
    const start = hashString(name) % CAT_COLORS.length;
    let chosen = CAT_COLORS[start];
    for (let i = 0; i < CAT_COLORS.length; i++) {
      const c = CAT_COLORS[(start + i) % CAT_COLORS.length];
      if (!used.has(c)) {
        chosen = c;
        break;
      }
    }
    used.add(chosen);
    assigned.set(name, chosen);
  }
  return (name) => `var(${assigned.get(name) || CAT_COLORS[hashString(name || "") % CAT_COLORS.length]})`;
}

function colorForRun(name) {
  return colorMapFor([name])(name);
}

function colorForCondition(label) {
  return colorForRun(label);
}

function hideAllMainViews() {
  document.getElementById("empty-main").style.display = "none";
  document.getElementById("run-view").style.display = "none";
  document.getElementById("compare-view").style.display = "none";
  document.getElementById("experiment-compare-view").style.display = "none";
}

function showEmptyMain() {
  hideAllMainViews();
  document.getElementById("empty-main").style.display = "block";
  state.mode = "empty";
}

function relativeTime(epochSeconds) {
  if (!epochSeconds) return null;
  const diff = Date.now() / 1000 - epochSeconds;
  if (diff < 0) return "just now";
  const units = [
    ["y", 31536000],
    ["mo", 2592000],
    ["d", 86400],
    ["h", 3600],
    ["m", 60],
  ];
  for (const [label, secs] of units) {
    if (diff >= secs) return `${Math.floor(diff / secs)}${label} ago`;
  }
  return "just now";
}

function isoToEpoch(iso) {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : t / 1000;
}

function fmtAbsolute(epochSeconds) {
  if (!epochSeconds) return null;
  const d = new Date(epochSeconds * 1000);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function fmtTokens(n) {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "k";
  return String(n);
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  seconds = Math.round(seconds);
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function tokenize(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/([a-z])(\d)/g, "$1 $2")
    .replace(/(\d)([a-z])/g, "$1 $2")
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

function queryMatches(query, parts) {
  const q = String(query || "").trim();
  if (!q) return true;
  const raw = parts.filter(Boolean).join(" ").toLowerCase();
  if (raw.includes(q.toLowerCase())) return true;
  const hayTokens = tokenize(raw);
  const hayJoined = hayTokens.join(" ");
  return tokenize(q).every((tok) => hayJoined.includes(tok) || hayTokens.some((t) => t.startsWith(tok)));
}

function statusSearchTerms(status) {
  if (status === "running") return ["running", "live"];
  if (status === "failed") return ["failed", "fail", "error"];
  if (status === "completed") return ["completed", "done", "finished"];
  return status ? [status] : [];
}

function extraNicknames(obj) {
  const extra = [];
  if (obj.benchmark === "bolt_lora") extra.push("hpo", "lora", "finetune");
  if ((obj.backend || "").includes("sara") || (obj.backends || []).some((b) => String(b).includes("sara"))) {
    extra.push("agent", "llm");
  }
  if (obj.disclosure === "revealed") extra.push("noblind");
  return extra;
}

function runSearchParts(run) {
  return [
    run.name,
    run.name.replace(/[/_\-]+/g, " "),
    run.heading,
    run.benchmark,
    run.benchmark_label,
    run.backend,
    run.backend_label,
    run.disclosure,
    run.disclosure_label,
    run.condition,
    run.model,
    run.run_kind,
    run.surrogate,
    run.seed != null ? `seed ${run.seed}` : "",
    ...statusSearchTerms(run.status),
    ...extraNicknames(run),
  ];
}

function groupSearchParts(g) {
  return [
    g.name,
    g.name.replace(/[/_\-]+/g, " "),
    g.title,
    g.heading,
    g.caption,
    g.benchmark,
    g.benchmark_label,
    g.seed != null ? `seed ${g.seed}` : "",
    ...(g.backends || []),
    ...(g.disclosures || []),
    ...(g.statuses || []).flatMap(statusSearchTerms),
    ...extraNicknames(g),
  ];
}

function runMatches(run) {
  return queryMatches(state.filters.q, runSearchParts(run));
}

function groupMatches(g) {
  return queryMatches(state.filters.q, groupSearchParts(g));
}

function filtersActive() {
  return Boolean(state.filters.q && state.filters.q.trim());
}

function dateBucket(iso) {
  const epoch = isoToEpoch(iso);
  if (!epoch) return { key: "unknown", label: "", order: 1e12 };
  const day = new Date(epoch * 1000);
  day.setHours(0, 0, 0, 0);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diffDays = Math.round((today.getTime() - day.getTime()) / 86400000);
  const key = String(day.getTime());
  if (diffDays === 0) return { key, label: "Today", order: 0 };
  if (diffDays === 1) return { key, label: "Yesterday", order: 1 };
  const label = day.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
  return { key, label, order: diffDays };
}

function appendDateSections(list, items, isoOf, renderItem) {
  const buckets = new Map();
  for (const item of items) {
    const b = dateBucket(isoOf(item));
    if (!buckets.has(b.key)) buckets.set(b.key, { label: b.label, order: b.order, items: [] });
    buckets.get(b.key).items.push(item);
  }
  const ordered = [...buckets.values()].sort((a, b) => a.order - b.order);
  for (const bucket of ordered) {
    bucket.items.sort((a, b) => (isoToEpoch(isoOf(b)) || 0) - (isoToEpoch(isoOf(a)) || 0));
    if (bucket.label) {
      const header = document.createElement("li");
      header.className = "date-header";
      header.textContent = bucket.label;
      list.appendChild(header);
    }
    for (const item of bucket.items) list.appendChild(renderItem(item));
  }
}

function writeFiltersToUrl() {
  const url = new URL(location.href);
  if (state.filters.q) url.searchParams.set("q", state.filters.q);
  else url.searchParams.delete("q");
  ["status", "bench", "backend", "disclosure", "when", "from", "to", "view"].forEach((k) => url.searchParams.delete(k));
  history.replaceState(null, "", url.pathname + url.search + url.hash);
}

function readFiltersFromUrl() {
  const p = new URLSearchParams(location.search);
  state.filters.q = p.get("q") || "";
  const search = document.getElementById("search-input");
  if (search) search.value = state.filters.q;
}

function renderFilterBar() {
  const visible =
    state.sidebarMode === "experiments"
      ? state.compareGroups.filter(groupMatches).length
      : state.runs.filter(runMatches).length;
  const total = state.sidebarMode === "experiments" ? state.compareGroups.length : state.runs.length;
  document.getElementById("filter-count").textContent = total ? `${visible}/${total}` : "";
  document.getElementById("filter-clear").hidden = !filtersActive();
}

function applyFilters() {
  renderFilterBar();
  renderSidebar();
  renderExperimentList();
  writeFiltersToUrl();
}

const LENZ_COMMAND_RE = /\blenz\s+([a-z-]+)/;
const CHANGE_COMMANDS = new Set([
  "create", "submit", "observe", "set-bounds", "set-acqf", "set-objectives",
  "set-constraints", "set-surrogate", "evolve-kernels",
]);

function extractLenzCommand(cmd) {
  const m = LENZ_COMMAND_RE.exec(cmd || "");
  return m ? m[1] : null;
}

function truncate(s, n) {
  if (!s) return "";
  s = String(s);
  return s.length > n ? s.slice(0, n) + `… (${s.length - n} more chars)` : s;
}

// -- nice axis ticks (Heckbert's "nice numbers" algorithm) -------------------------------------------------------------

function niceNum(range, round) {
  if (range === 0) return 1;
  const exponent = Math.floor(Math.log10(range));
  const fraction = range / Math.pow(10, exponent);
  let niceFraction;
  if (round) {
    if (fraction < 1.5) niceFraction = 1;
    else if (fraction < 3) niceFraction = 2;
    else if (fraction < 7) niceFraction = 5;
    else niceFraction = 10;
  } else {
    if (fraction <= 1) niceFraction = 1;
    else if (fraction <= 2) niceFraction = 2;
    else if (fraction <= 5) niceFraction = 5;
    else niceFraction = 10;
  }
  return niceFraction * Math.pow(10, exponent);
}

function niceTicks(min, max, count = 5) {
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const range = niceNum(max - min, false);
  const step = niceNum(range / (count - 1), true);
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = niceMin; v <= niceMax + step * 1e-9; v += step) ticks.push(Math.round(v / step) * step);
  return { ticks, min: niceMin, max: niceMax };
}

// -- sidebar -------------------------------------------------------------

async function loadCompareGroups() {
  state.compareGroupsLoading = true;
  state.compareGroupsError = null;
  renderGroupPicker();
  try {
    const res = await fetch("/api/compare-groups");
    if (!res.ok) {
      throw new Error(res.status === 404 ? "endpoint missing (restart sara-viz)" : `HTTP ${res.status}`);
    }
    state.compareGroups = await res.json();
  } catch (err) {
    state.compareGroups = [];
    state.compareGroupsError = err.message || String(err);
  } finally {
    state.compareGroupsLoading = false;
    renderGroupPicker();
    if (state.sidebarMode === "experiments" || state.pendingExperiments) {
      state.pendingExperiments = false;
      openExperimentsView();
    }
  }
}

function openExperimentsView() {
  if (state.compareGroupsLoading) {
    state.pendingExperiments = true;
    showExperimentLoading();
    return;
  }
  if (state.compareGroupsError) {
    showExperimentError(state.compareGroupsError);
    return;
  }
  if (!state.compareGroups.length) {
    showEmptyMain();
    return;
  }
  const visible = state.compareGroups.filter(groupMatches);
  const target =
    state.currentGroup && visible.some((g) => g.name === state.currentGroup)
      ? state.currentGroup
      : visible[0]
        ? visible[0].name
        : null;
  if (!target) {
    showEmptyMain();
    document.getElementById("empty-main").textContent = "No experiments match.";
    return;
  }
  selectCompareGroup(target);
}

function showExperimentLoading() {
  hideAllMainViews();
  document.getElementById("experiment-compare-view").style.display = "block";
  document.getElementById("experiment-compare-view").innerHTML =
    '<div class="empty-note" style="padding-top:8px">Loading…</div>';
  state.mode = "experiment";
}

function showExperimentError(message) {
  hideAllMainViews();
  document.getElementById("experiment-compare-view").style.display = "block";
  document.getElementById("experiment-compare-view").innerHTML = `
    <div class="empty-note" style="padding-top:8px">
      Could not load groups (${escapeHtml(message)}).
    </div>`;
  state.mode = "experiment";
}

function renderGroupPicker() {
  renderFilterBar();
  renderExperimentList();
}

function renderExperimentList() {
  const list = document.getElementById("experiment-list");
  const empty = document.getElementById("group-empty");
  if (!list) return;
  list.innerHTML = "";
  if (state.compareGroupsLoading) {
    empty.style.display = "none";
    list.innerHTML = '<li class="empty-note sidebar-empty">Loading…</li>';
    return;
  }
  if (state.compareGroupsError) {
    empty.style.display = "none";
    list.innerHTML = `<li class="empty-note sidebar-empty">${escapeHtml(state.compareGroupsError)}</li>`;
    return;
  }
  if (!state.compareGroups.length) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";
  const visible = state.compareGroups.filter(groupMatches);
  if (!visible.length) {
    list.innerHTML = '<li class="empty-note sidebar-empty">No experiments match.</li>';
    return;
  }
  const colorOf = colorMapFor(visible.map((g) => g.name));
  appendDateSections(list, visible, (g) => g.started_at, (g) => makeExperimentItem(g, colorOf(g.name)));
}

function makeExperimentItem(g, color) {
  const li = document.createElement("li");
  li.className = "run-item" + (g.name === state.currentGroup ? " active" : "");
  li.dataset.name = g.name;
  const pending = g.n_scored < g.n_conditions ? `${g.n_scored}/${g.n_conditions} ready` : `${g.n_conditions} conditions`;
  const rel = relativeTime(isoToEpoch(g.started_at));
  const meta = [pending, rel].filter(Boolean);
  li.innerHTML = `
      <span class="run-dot" style="background:${color || colorForRun(g.name)}"></span>
      <div class="run-body">
        <div class="run-name" title="${escapeHtml(g.name)}">${escapeHtml(g.heading || g.name)}</div>
        <div class="run-meta">${meta.map((bit) => `<span>${escapeHtml(bit)}</span>`).join('<span class="sep">·</span>')}</div>
      </div>`;
  li.addEventListener("click", () => selectCompareGroup(g.name));
  return li;
}

function initSidebarNav() {
  document.querySelectorAll(".sidebar-nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.sidebar;
      if (mode === state.sidebarMode) return;
      setSidebarMode(mode);
    });
  });
}

function applySidebarModeUI(mode) {
  state.sidebarMode = mode;
  document.querySelectorAll(".sidebar-nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.sidebar === mode);
  });
  document.getElementById("runs-sidebar").classList.toggle("is-hidden", mode !== "runs");
  document.getElementById("experiments-sidebar").classList.toggle("is-hidden", mode !== "experiments");
}

function splitGroupRef(name) {
  const sep = "::";
  const raw = String(name || "");
  const i = raw.lastIndexOf(sep);
  if (i === -1) return { rel: raw, seed: null };
  const rhs = raw.slice(i + sep.length);
  if (!/^-?\d+$/.test(rhs)) return { rel: raw, seed: null };
  return { rel: raw.slice(0, i), seed: Number(rhs) };
}

function runNameForCurrentExperiment() {
  const fromDetail = (state.groupDetail?.conditions || []).map((c) => c.run_name).find(Boolean);
  if (fromDetail) return fromDetail;
  if (!state.currentGroup) return null;
  const { rel, seed } = splitGroupRef(state.currentGroup);
  const match = state.runs.find(
    (r) => r.group === rel && (seed == null || r.seed === seed || r.seed == null)
  );
  return match ? match.name : null;
}

function setSidebarMode(mode) {
  if (mode === state.sidebarMode) return;
  applySidebarModeUI(mode);
  renderFilterBar();
  if (mode === "experiments") {
    openExperimentsView();
    return;
  }
  loadRuns().then(() => {
    const runName = runNameForCurrentExperiment();
    if (runName && state.runs.some((r) => r.name === runName)) {
      selectRun(runName);
      return;
    }
    if (state.mode === "experiment") showEmptyMain();
  });
}

async function selectCompareGroup(name) {
  if (!name) return;
  state.currentGroup = name;
  document.querySelectorAll("#experiment-list .run-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.name === name);
  });
  const url = new URL(location.href);
  url.searchParams.delete("compare");
  url.searchParams.set("group", name);
  url.hash = "";
  history.replaceState(null, "", url.pathname + url.search);

  const res = await fetch(`/api/compare-groups/${encodeURIComponent(name)}`);
  if (!res.ok) {
    alert(`Could not load comparison group "${name}".`);
    return;
  }
  state.groupDetail = await res.json();
  state.mode = "experiment";
  hideAllMainViews();
  document.getElementById("experiment-compare-view").style.display = "block";
  renderExperimentCompare(state.groupDetail);
}

async function loadRuns() {
  try {
    const res = await fetch("/api/runs");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const payload = await res.json();
    state.runs = Array.isArray(payload) ? payload : [];
  } catch (err) {
    console.error(err);
    if (!Array.isArray(state.runs)) state.runs = [];
  }
  renderFilterBar();
  renderSidebar();
}

function statusBadge(run) {
  if (run.run_kind === "lenz") return "";
  const status = run.status || "completed";
  const cls = `status-${status}`;
  const pulse = status === "running" ? '<span class="status-pulse"></span> ' : "";
  return `<span class="badge ${cls}">${pulse}${escapeHtml(status)}</span>`;
}

function runTitle(run) {
  const backend = run.backend_label || run.condition || "";
  const bench = run.benchmark_label || "";
  const seedBit = run.seed != null ? ` · seed ${run.seed}` : "";
  if (backend && bench) return `${backend} · ${bench}${seedBit}`;
  if (run.heading) return run.heading;
  if (backend) return backend;
  if (bench) return bench;
  const parts = String(run.name || "")
    .split("/")
    .filter((p) => p && !p.startsWith("sandbox_"));
  return parts.slice(-2).join(" / ") || run.name;
}

function makeRunItem(run, color) {
  const li = document.createElement("li");
  li.className = "run-item";
  li.dataset.name = run.name;
  const rel = relativeTime(isoToEpoch(run.started_at));
  const meta = [run.disclosure_label, `${run.n_observed}/${run.n_trials}`, rel].filter(Boolean);
  const badges = [
    statusBadge(run),
    run.surrogate === "cake" ? '<span class="badge cake">CAKE</span>' : "",
    run.is_moo ? '<span class="badge">MOO</span>' : "",
    run.run_kind === "lenz" ? '<span class="badge">lenz-only</span>' : "",
  ].filter(Boolean);
  li.innerHTML = `
      <input type="checkbox" class="run-check" title="Select for compare" ${state.selected.has(run.name) ? "checked" : ""} />
      <span class="run-dot" style="background:${color || colorForRun(run.name)}"></span>
      <div class="run-body">
        <div class="run-name" title="${escapeHtml(run.name)}">${escapeHtml(runTitle(run))}</div>
        <div class="run-meta">${meta.map((bit) => `<span>${escapeHtml(bit)}</span>`).join('<span class="sep">·</span>')}${badges.length ? badges.join("") : ""}</div>
      </div>
      <button class="run-delete" title="Delete this run">🗑</button>`;
  li.querySelector(".run-check").addEventListener("click", (e) => {
    e.stopPropagation();
    toggleSelect(run.name, e.target.checked);
  });
  li.querySelector(".run-delete").addEventListener("click", (e) => {
    e.stopPropagation();
    deleteRun(run.name);
  });
  li.addEventListener("click", () => selectRun(run.name));
  return li;
}

function renderSidebar() {
  const list = document.getElementById("run-list");
  const empty = document.getElementById("empty-state");
  list.innerHTML = "";
  if (state.runs.length === 0) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";

  const visible = state.runs.filter(runMatches);
  if (!visible.length) {
    list.innerHTML = '<li class="empty-note sidebar-empty">No runs match.</li>';
    return;
  }
  const colorOf = colorMapFor(visible.map((run) => run.name));
  appendDateSections(list, visible, (run) => run.started_at, (run) => makeRunItem(run, colorOf(run.name)));
}

function toggleSelect(name, checked) {
  if (checked) state.selected.add(name);
  else state.selected.delete(name);
  updateCompareBar();
}

function updateCompareBar() {
  const bar = document.getElementById("compare-bar");
  const count = state.selected.size;
  bar.classList.toggle("visible", count > 0);
  document.getElementById("compare-count").textContent = count ? `${count} selected` : "0 selected";
  document.getElementById("compare-btn").disabled = count < 2;
  document.getElementById("compare-btn").style.opacity = count < 2 ? 0.5 : 1;
}

async function deleteRun(name) {
  const ok = confirm(`Delete run "${name}"?\n\nThis permanently removes its state.json/trace.jsonl from disk. This cannot be undone.`);
  if (!ok) return;
  const res = await fetch(`/api/runs/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) {
    alert(`Failed to delete "${name}".`);
    return;
  }
  state.selected.delete(name);
  if (state.current && state.current.name === name) {
    state.current = null;
    showEmptyMain();
    history.replaceState(null, "", location.pathname);
  }
  await loadRuns();
  updateCompareBar();
}

function initSearchAndCompare() {
  document.getElementById("search-input").addEventListener("input", (e) => {
    state.filters.q = e.target.value;
    applyFilters();
  });
  document.getElementById("filter-clear").addEventListener("click", () => {
    state.filters.q = "";
    document.getElementById("search-input").value = "";
    applyFilters();
  });
  document.getElementById("compare-clear").addEventListener("click", () => {
    state.selected.clear();
    renderSidebar();
    updateCompareBar();
  });
  document.getElementById("compare-btn").addEventListener("click", () => {
    if (state.selected.size < 2) return;
    const q = [...state.selected].map(encodeURIComponent).join(",");
    const url = new URL(location.href);
    url.searchParams.delete("group");
    url.searchParams.set("compare", q);
    url.hash = "";
    history.replaceState(null, "", url.pathname + url.search);
    renderCompare();
  });
}

// -- single run: fetch + shell -------------------------------------------------------------

async function selectRun(name) {
  applySidebarModeUI("runs");
  document.querySelectorAll("#run-list .run-item").forEach((el) => el.classList.toggle("active", el.dataset.name === name));
  const url = new URL(location.href);
  url.searchParams.delete("group");
  url.searchParams.delete("compare");
  history.replaceState(null, "", `${url.pathname}${url.search}#${encodeURIComponent(name)}|${state.activeTab}`);
  const res = await fetch(`/api/runs/${encodeURIComponent(name)}`);
  const detail = await res.json();
  state.current = detail;
  state.mode = "single";
  hideAllMainViews();
  document.getElementById("run-view").style.display = "block";
  renderRun(detail);
}

function runInfoRows(detail) {
  const meta = detail.meta;
  const shelf = detail.state.shelf;
  const startEpoch = isoToEpoch(detail.started_at);
  const rows = [];
  const isSara = detail.run_kind === "sara";
  rows.push(["Kind", isSara ? "Sara + lenz" : "lenz-only (no Sara agent)"]);
  if (isSara && meta) {
    rows.push(["Provider", meta.provider || "-"]);
    rows.push(["Model", meta.model || "-", "mono"]);
    if (meta.base_url) rows.push(["Base URL", meta.base_url, "mono"]);
    rows.push(["Budget", meta.budget ?? "-"]);
    if (meta.eval_cmd) rows.push(["Eval command", meta.eval_cmd, "mono"]);
  } else if (!isSara && meta && meta.policy) {
    rows.push(["Policy", meta.policy, "mono"]);
    if (meta.budget != null) rows.push(["Budget", meta.budget]);
  }
  rows.push(["Surrogate", shelf.surrogate === "cake" ? "CAKE (adaptive)" : "fixed Matérn"]);
  rows.push(["Acquisition", shelf.acqf]);
  if (detail.regret && detail.regret.f_opt != null) {
    rows.push(["True f_opt", fmtNum(detail.regret.f_opt, 6), "mono"]);
  }
  if (detail.regret && detail.regret.benchmark) {
    rows.push(["True benchmark", detail.regret.benchmark, "mono"]);
  }
  rows.push(["Started", startEpoch ? fmtAbsolute(startEpoch) : "-"]);
  rows.push(["Duration", fmtDuration(detail.duration_seconds)]);
  return rows;
}

function renderRun(detail) {
  const shelf = detail.state.shelf;
  const meta = detail.meta;
  const isCake = shelf.surrogate === "cake";

  document.getElementById("run-title").innerHTML =
    `<span class="run-dot" style="background:${colorForRun(detail.name)};display:inline-block;margin-top:0"></span> ${escapeHtml(detail.name)}`;
  document.getElementById("run-subtitle").textContent = shelf.objectives.map((o) => `${o.metric} (${o.minimize ? "min" : "max"})`).join(", ");

  const startEpoch = isoToEpoch(detail.started_at);
  const subrowParts = [];
  if (meta && meta.model) subrowParts.push(`<strong>${escapeHtml(meta.model)}</strong>`);
  if (startEpoch) subrowParts.push(`started ${relativeTime(startEpoch)}`);
  if (detail.duration_seconds !== null && detail.duration_seconds !== undefined) subrowParts.push(`ran ${fmtDuration(detail.duration_seconds)}`);
  const nObs = (detail.state.trials || []).filter((t) => t.status === "observed").length;
  subrowParts.push(`${nObs}/${(detail.state.trials || []).length} evaluations`);
  document.getElementById("run-subrow").innerHTML = subrowParts.map((p) => `<span>${p}</span>`).join('<span class="sep">·</span>');

  const badges = [];
  badges.push(`<span class="kind-badge ${detail.run_kind}">${detail.run_kind === "sara" ? "Sara + lenz" : "lenz-only"}</span>`);
  if (meta && meta.status) badges.push(`<span class="badge status-${meta.status}">${meta.status === "running" ? '<span class="status-pulse"></span> ' : ""}${escapeHtml(meta.status)}</span>`);
  const headerBadges = document.getElementById("header-badges");
  headerBadges.innerHTML = badges.join(" ") + `<button class="run-delete" id="header-delete" title="Delete this run" style="opacity:0.55">🗑</button>`;
  document.getElementById("header-delete").addEventListener("click", () => deleteRun(detail.name));

  document.getElementById("kernels-tab-btn").style.display = isCake ? "inline-block" : "none";
  if (!isCake && state.activeTab === "kernels") state.activeTab = "overview";

  const hasToolUse = detail.run_kind === "sara" && detail.tool_use && detail.tool_use.length > 0;
  document.getElementById("tool-use-tab-btn").style.display = hasToolUse ? "inline-block" : "none";
  if (!hasToolUse && state.activeTab === "toolUse") state.activeTab = "overview";

  const panels = document.getElementById("panels");
  panels.innerHTML = `
    <div class="tab-panel" id="panel-overview"></div>
    <div class="tab-panel" id="panel-config"></div>
    <div class="tab-panel" id="panel-trials"></div>
    <div class="tab-panel" id="panel-trace"></div>
    <div class="tab-panel" id="panel-toolUse"></div>
    <div class="tab-panel" id="panel-kernels"></div>
  `;

  renderOverview(detail);
  renderConfig(detail);
  renderTrials(detail);
  renderTrace(detail);
  if (hasToolUse) renderToolUse(detail);
  if (isCake) renderKernels(detail);

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === state.activeTab);
    btn.onclick = () => {
      state.activeTab = btn.dataset.tab;
      history.replaceState(null, "", `#${encodeURIComponent(detail.name)}|${state.activeTab}`);
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
    };
  });
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
  const activePanel = document.getElementById(`panel-${state.activeTab}`) || document.getElementById("panel-overview");
  activePanel.classList.add("active");

  // Rendered unconditionally, even while the overview panel is hidden behind another
  // active tab: the chart's ResizeObserver (see renderChart) picks up the transition
  // to visible whenever the user switches to Overview later and redraws at that point.
  if (detail.convergence && detail.convergence.points.length >= 2) {
    renderChart(document.getElementById("chart-wrap"), [{ name: detail.name, convergence: detail.convergence }]);
  }
  if (detail.regret && detail.regret.points && detail.regret.points.length >= 2) {
    const wrap = document.getElementById("overview-regret-wrap");
    if (wrap) {
      renderChart(wrap, [
        {
          name: "true regret",
          convergence: { metric: "regret", minimize: true, points: detail.regret.points },
        },
      ]);
    }
  }
}

// -- run info + config -------------------------------------------------------------

function renderOverview(detail) {
  const panel = document.getElementById("panel-overview");
  const shelf = detail.state.shelf;
  const trials = detail.state.trials || [];
  const observed = trials.filter((t) => t.status === "observed");
  const inFlight = trials.filter((t) => t.status === "in_flight");

  let statCards = `
    <div class="stat"><div class="value">${observed.length}</div><div class="label">Observed</div></div>
    <div class="stat"><div class="value">${inFlight.length}</div><div class="label">In-flight</div></div>
    <div class="stat"><div class="value">${escapeHtml(shelf.acqf)}</div><div class="label">Acquisition</div></div>
    <div class="stat"><div class="value">${fmtDuration(detail.duration_seconds)}</div><div class="label">Duration</div></div>
  `;
  if (detail.convergence && detail.convergence.points.length) {
    const pts = detail.convergence.points;
    const best = pts[pts.length - 1].best;
    statCards += `<div class="stat"><div class="value">${fmtNum(best)}</div><div class="label">Best ${escapeHtml(detail.convergence.metric)}</div></div>`;
  }
  if (detail.regret) {
    statCards += `<div class="stat"><div class="value">${fmtNum(detail.regret.best, 6)}</div><div class="label">Best regret</div></div>`;
    statCards += `<div class="stat"><div class="value">${fmtNum(detail.regret.eval1, 4)}</div><div class="label">Regret @ 1</div></div>`;
  }
  const usage = detail.meta && detail.meta.usage;
  if (usage) {
    const total = (usage.input_tokens || 0) + (usage.output_tokens || 0);
    statCards += `<div class="stat"><div class="value">${fmtTokens(total)}</div><div class="label">Tokens (${fmtTokens(usage.input_tokens)} in / ${fmtTokens(usage.output_tokens)} out)</div></div>`;
  }

  const infoRows = runInfoRows(detail)
    .map(([label, value, cls]) => `<div class="info-item"><div class="info-label">${escapeHtml(label)}</div><div class="info-value ${cls || ""}">${escapeHtml(String(value))}</div></div>`)
    .join("");

  let chartHtml = "";
  if (detail.convergence && detail.convergence.points.length >= 2) {
    chartHtml = `
      <div class="card">
        <h3>Convergence</h3>
        <div id="chart-wrap"></div>
        <div class="legend" id="chart-legend"></div>
      </div>`;
  } else if (shelf.objectives.length > 1) {
    chartHtml = `<div class="card"><h3>Convergence</h3><div class="empty-note">Multi-objective study: convergence chart not shown (see Trials for the full Pareto history).</div></div>`;
  } else {
    chartHtml = `<div class="card"><h3>Convergence</h3><div class="empty-note">Not enough observed evaluations yet.</div></div>`;
  }
  if (detail.regret && detail.regret.points && detail.regret.points.length >= 2) {
    chartHtml += `
      <div class="card">
        <h3>True regret</h3>
        <div class="chart-note">Best-so-far simple regret vs the hidden optimum.</div>
        <div id="overview-regret-wrap"></div>
      </div>`;
  }

  panel.innerHTML = `
    <div class="card">
      <h3>Run</h3>
      <div class="info-grid" style="margin-bottom:16px">${infoRows}</div>
      <div class="stat-row">${statCards}</div>
    </div>
    ${chartHtml}
  `;
}

function spaceTable(space) {
  const rows = Object.entries(space)
    .map(([name, spec]) => {
      let range;
      if (spec.kind === "range") {
        range = `[${fmtNum(spec.lower, 4)}, ${fmtNum(spec.upper, 4)}]${spec.log_scale ? " (log)" : ""}${spec.step ? ` step ${spec.step}` : ""}`;
      } else {
        range = (spec.values || []).map((v) => escapeHtml(String(v))).join(", ");
      }
      return `<tr><td class="mono">${escapeHtml(name)}</td><td>${escapeHtml(spec.kind)}</td><td class="mono">${range}</td></tr>`;
    })
    .join("");
  return `<table><thead><tr><th>parameter</th><th>kind</th><th>domain</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderConfig(detail) {
  const panel = document.getElementById("panel-config");
  const shelf = detail.state.shelf;
  const meta = detail.meta;

  const objectivesRows = shelf.objectives.map((o) => `<tr><td>${escapeHtml(o.metric)}</td><td>${o.minimize ? "minimize" : "maximize"}</td></tr>`).join("");
  const constraintsRows = (shelf.constraints || []).length
    ? shelf.constraints.map((c) => `<tr><td>${escapeHtml(c.metric)}</td><td>${c.lower ?? "—"}</td><td>${c.upper ?? "—"}</td></tr>`).join("")
    : `<tr><td colspan="3" class="empty-note">None</td></tr>`;

  let kernelSection = "";
  if (shelf.surrogate === "cake") {
    const cake = (detail.state.plugins && detail.state.plugins.cake) || shelf;
    const kl = cake.kernel_llm || {};
    kernelSection = `
      <div class="subhead">CAKE kernel evolution</div>
      <table class="kv-table">
        <tr><td>LLM provider</td><td class="mono">${escapeHtml(kl.provider || "—")}</td></tr>
        <tr><td>LLM model</td><td class="mono">${escapeHtml(kl.model || "—")}</td></tr>
        <tr><td>Population size</td><td>${cake.kernel_population_size}</td></tr>
        <tr><td>Init after</td><td>${cake.kernel_init_after} observations</td></tr>
        <tr><td>Evolve every</td><td>${cake.kernel_evolve_every} observations</td></tr>
        <tr><td>Freeze fraction</td><td>${cake.kernel_freeze_fraction}</td></tr>
      </table>`;
  }

  let metaSection = "";
  if (meta) {
    metaSection = `
      <div class="subhead">Agent run</div>
      <table class="kv-table">
        <tr><td>Provider</td><td class="mono">${escapeHtml(meta.provider || "—")}</td></tr>
        <tr><td>Model</td><td class="mono">${escapeHtml(meta.model || "—")}</td></tr>
        <tr><td>Base URL</td><td class="mono">${escapeHtml(meta.base_url || "—")}</td></tr>
        <tr><td>Context file</td><td class="mono">${escapeHtml(meta.context_path || "—")}</td></tr>
        <tr><td>Eval command</td><td class="mono">${escapeHtml(meta.eval_cmd || "—")}</td></tr>
        <tr><td>Budget</td><td>${meta.budget ?? "—"}</td></tr>
      </table>`;
  }

  panel.innerHTML = `
    ${metaSection ? `<div class="card">${metaSection}</div>` : ""}
    <div class="card">
      <div class="subhead">Search space</div>
      ${spaceTable(detail.state.space)}
      <div class="subhead">Objectives</div>
      <table><thead><tr><th>metric</th><th>direction</th></tr></thead><tbody>${objectivesRows}</tbody></table>
      <div class="subhead">Constraints</div>
      <table><thead><tr><th>metric</th><th>lower</th><th>upper</th></tr></thead><tbody>${constraintsRows}</tbody></table>
      <div class="subhead">Acquisition</div>
      <table class="kv-table">
        <tr><td>Function</td><td class="mono">${escapeHtml(shelf.acqf)}</td></tr>
        <tr><td>Params</td><td class="mono">${escapeHtml(JSON.stringify(shelf.acqf_params || {}))}</td></tr>
        <tr><td>Active bounds</td><td class="mono">${Object.keys(shelf.bounds || {}).length ? escapeHtml(JSON.stringify(shelf.bounds)) : "full domain"}</td></tr>
      </table>
      ${kernelSection}
    </div>
  `;
}

// -- convergence chart (single series or overlaid comparison) -------------------------------------------------------------

function renderChart(container, series, colorOf) {
  // series: [{name, convergence: {metric, minimize, points}}]
  colorOf = colorOf || (series.length > 1 ? colorMapFor(series.map((s) => s.name)) : () => "var(--series-1)");
  function draw() {
    const W = container.clientWidth || 600;
    const H = 280;
    const margin = { top: 10, right: 20, bottom: 28, left: 56 };
    const plotW = Math.max(10, W - margin.left - margin.right);
    const plotH = H - margin.top - margin.bottom;

    const allPoints = series.flatMap((s) => s.convergence.points);
    if (!allPoints.length) {
      container.innerHTML = `<div class="empty-note">No data to plot.</div>`;
      return;
    }
    const maxI = Math.max(...series.map((s) => s.convergence.points[s.convergence.points.length - 1].i));
    const minI = 1;
    const allY = allPoints.flatMap((p) => [p.value, p.best]);
    const { ticks: yTicks, min: yMin, max: yMax } = niceTicks(Math.min(...allY), Math.max(...allY), 5);

    const xScale = (i) => margin.left + ((i - minI) / (maxI - minI || 1)) * plotW;
    const yScale = (v) => margin.top + plotH - ((v - yMin) / (yMax - yMin || 1)) * plotH;

    let gridlines = "";
    let yLabels = "";
    for (const v of yTicks) {
      const y = yScale(v);
      gridlines += `<line class="gridline" x1="${margin.left}" x2="${W - margin.right}" y1="${y}" y2="${y}"></line>`;
      yLabels += `<text class="axis-label" x="${margin.left - 8}" y="${y + 3}" text-anchor="end">${fmtNum(v, 2)}</text>`;
    }

    const nTicksX = Math.min(6, maxI);
    let xLabels = "";
    for (let k = 0; k < nTicksX; k++) {
      const i = Math.round(minI + (k / Math.max(1, nTicksX - 1)) * (maxI - minI));
      const x = xScale(i);
      const anchor = k === 0 ? "start" : k === nTicksX - 1 ? "end" : "middle";
      xLabels += `<text class="axis-label" x="${x}" y="${H - 6}" text-anchor="${anchor}">${i}</text>`;
    }

    let seriesSvg = "";
    for (const s of series) {
      const pts = s.convergence.points;
      const color = series.length > 1 ? colorOf(s.name) : "var(--series-1)";
      const path = pts.map((p, idx) => `${idx === 0 ? "M" : "L"}${xScale(p.i)},${yScale(p.best)}`).join(" ");
      const dots = series.length > 1 ? "" : pts.map((p) => `<circle class="raw-dot" cx="${xScale(p.i)}" cy="${yScale(p.value)}" r="3"></circle>`).join("");
      seriesSvg += `${dots}<path fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="${path}"></path>`;
    }

    container.innerHTML = `
      <svg class="chart" viewBox="0 0 ${W} ${H}">
        ${gridlines}
        <line class="baseline" x1="${margin.left}" x2="${W - margin.right}" y1="${margin.top + plotH}" y2="${margin.top + plotH}"></line>
        ${yLabels}
        ${xLabels}
        ${seriesSvg}
        <g id="hover-layer"></g>
      </svg>
    `;

    const svg = container.querySelector("svg");
    const hoverLayer = container.querySelector("#hover-layer");
    const tooltip = document.getElementById("tooltip");

    svg.addEventListener("mousemove", (evt) => {
      const rect = svg.getBoundingClientRect();
      const mx = ((evt.clientX - rect.left) / rect.width) * W;
      const targetI = Math.round(minI + ((mx - margin.left) / plotW) * (maxI - minI));
      let dotsHtml = `<line class="crosshair" x1="${xScale(targetI)}" x2="${xScale(targetI)}" y1="${margin.top}" y2="${margin.top + plotH}"></line>`;
      let tipRows = "";
      for (const s of series) {
        const pts = s.convergence.points;
        let nearest = pts[0];
        let best = Infinity;
        for (const p of pts) {
          const d = Math.abs(p.i - targetI);
          if (d < best) {
            best = d;
            nearest = p;
          }
        }
        const color = series.length > 1 ? colorOf(s.name) : "var(--series-1)";
        dotsHtml += `<circle class="hover-dot" cx="${xScale(nearest.i)}" cy="${yScale(nearest.best)}" r="4.5" style="fill:${color}"></circle>`;
        tipRows += `<div><span style="color:${color}">●</span> ${series.length > 1 ? escapeHtml(s.name) + ": " : ""}${fmtNum(nearest.best)} <span style="color:var(--text-muted)">(eval ${nearest.i})</span></div>`;
      }
      hoverLayer.innerHTML = dotsHtml;
      tooltip.style.display = "block";
      tooltip.style.left = evt.pageX + 14 + "px";
      tooltip.style.top = evt.pageY - 10 + "px";
      tooltip.innerHTML = tipRows;
    });
    svg.addEventListener("mouseleave", () => {
      hoverLayer.innerHTML = "";
      tooltip.style.display = "none";
    });
  }

  draw();
  if (!container._resizeObserver) {
    let raf = null;
    const ro = new ResizeObserver(() => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(draw);
    });
    ro.observe(container);
    container._resizeObserver = ro;
  }

  const legend = document.getElementById("chart-legend");
  if (legend) {
    if (series.length > 1) {
      legend.innerHTML = series
        .map((s) => `<div class="legend-item"><span class="legend-swatch" style="background:${colorOf(s.name)}"></span>${escapeHtml(s.name)}</div>`)
        .join("");
    } else {
      legend.innerHTML = `
        <div class="legend-item"><span class="legend-swatch" style="background:var(--series-1)"></span>best-so-far</div>
        <div class="legend-item"><span class="legend-swatch" style="background:var(--series-2)"></span>observed value</div>`;
    }
  }
}

// -- trials table -------------------------------------------------------------

function renderTrials(detail) {
  const panel = document.getElementById("panel-trials");
  const trials = [...(detail.state.trials || [])].reverse();
  if (!trials.length) {
    panel.innerHTML = `<div class="card"><div class="empty-note">No trials yet.</div></div>`;
    return;
  }
  const rows = trials
    .map((t) => {
      const cfg = Object.entries(t.config || {}).map(([k, v]) => `${k}=${typeof v === "number" ? fmtNum(v, 3) : v}`).join(", ");
      const metrics = t.metrics ? Object.entries(t.metrics).map(([k, v]) => `${k}=${typeof v === "number" ? fmtNum(v, 4) : v}`).join(", ") : "—";
      return `<tr>
        <td class="mono">${escapeHtml(t.trial_id)}</td>
        <td><span class="status-dot ${t.status}"></span>${t.status}</td>
        <td class="mono">${escapeHtml(cfg)}</td>
        <td class="mono">${escapeHtml(metrics)}</td>
      </tr>`;
    })
    .join("");
  panel.innerHTML = `
    <div class="card">
      <h3>Trial log (newest first)</h3>
      <table>
        <thead><tr><th>id</th><th>status</th><th>config</th><th>metrics</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

// -- trace -------------------------------------------------------------

function renderTrace(detail) {
  const panel = document.getElementById("panel-trace");
  const trace = detail.trace || [];
  if (!trace.length) {
    let note;
    if (detail.run_kind === "lenz") {
      note = "This run was driven directly through the <code>lenz</code> CLI (scripted or manual) — no Sara agent was involved, so there's no reasoning trace to show. See the Trials tab for what was evaluated.";
    } else {
      note = "No <code>trace.jsonl</code> found for this run. It may predate trace logging, or the workdir was pointed elsewhere with <code>--trace</code>.";
    }
    panel.innerHTML = `<div class="card"><div class="empty-note">${note}</div></div>`;
    return;
  }

  let html = "";
  let turnIndex = 0;
  let i = 0;
  const t0 = trace[0] && trace[0].ts;
  while (i < trace.length) {
    const entry = trace[i];
    const elapsed = typeof entry.ts === "number" && typeof t0 === "number" ? `+${fmtDuration(entry.ts - t0)}` : "";
    if (entry.role === "assistant") {
      turnIndex += 1;
      const toolCalls = entry.tool_calls || [];
      let toolsHtml = "";
      let j = i + 1;
      const results = [];
      while (j < trace.length && trace[j].role === "tool") {
        results.push(trace[j]);
        j += 1;
      }
      for (let k = 0; k < toolCalls.length; k++) {
        const tc = toolCalls[k];
        const cmd = tc.arguments && tc.arguments.cmd ? tc.arguments.cmd : tc.arguments && tc.arguments.path ? `read ${tc.arguments.path}` : JSON.stringify(tc.arguments);
        const lenzCmd = tc.name === "bash" ? extractLenzCommand(cmd) : null;
        const changed = lenzCmd && CHANGE_COMMANDS.has(lenzCmd);
        const result = results[k];
        toolsHtml += `
          <div class="tool-call"><span class="cmd">${escapeHtml(tc.name)}</span>${changed ? ` <span class="change-badge">${escapeHtml(lenzCmd)}</span>` : ""}\n${escapeHtml(cmd)}</div>
          ${result ? `<div class="tool-result">${escapeHtml(truncate(result.content, 1200))}</div>` : ""}
        `;
      }
      html += `
        <div class="trace-turn">
          <div class="trace-turn-header">turn ${turnIndex} · assistant${toolCalls.length ? ` · ${toolCalls.length} tool call${toolCalls.length > 1 ? "s" : ""}` : ""} ${elapsed ? `<span style="margin-left:auto">${elapsed}</span>` : ""}</div>
          <div class="trace-content">
            ${entry.content ? `<div class="trace-reasoning">${escapeHtml(entry.content)}</div>` : ""}
            ${toolsHtml}
          </div>
        </div>`;
      i = j;
    } else if (entry.role === "user") {
      html += `
        <div class="trace-turn">
          <div class="trace-turn-header">instructor ${elapsed ? `<span style="margin-left:auto">${elapsed}</span>` : ""}</div>
          <div class="trace-content"><div class="trace-reasoning">${escapeHtml(truncate(entry.content, 4000))}</div></div>
        </div>`;
      i += 1;
    } else {
      i += 1;
    }
  }
  panel.innerHTML = html;
}

// -- tool use over the campaign -------------------------------------------------------------
// Relative frequency of lenz calls ... over
// normalized trial progress for a single run. A stacked-area chart of which
// `lenz` subcommand Sara called, binned by how far through the campaign's
// observed evaluations she was at the time. Backed by `detail.tool_use`
// (see `_tool_use_events` in viz/server.py) -- data-driven, not hardcoded to
// the specific call-type list, so it reflects whatever this run actually did
// (including cake's evolve-kernels/kernel-population, or a constrained run's
// set-constraints).

const TOOL_USE_BINS = 10;

function binToolUseEvents(events, nBins) {
  const bins = Array.from({ length: nBins }, () => new Map());
  for (const e of events) {
    if (e.progress === null || e.progress === undefined) continue;
    const idx = Math.min(nBins - 1, Math.floor(e.progress * nBins));
    const bin = bins[idx];
    bin.set(e.call_type, (bin.get(e.call_type) || 0) + 1);
  }
  return bins;
}

function renderToolUse(detail) {
  const panel = document.getElementById("panel-toolUse");
  const events = detail.tool_use || [];

  const totals = new Map();
  for (const e of events) totals.set(e.call_type, (totals.get(e.call_type) || 0) + 1);
  // stacking + legend order: most-used call type first (drawn at the bottom of the stack)
  const callTypes = [...totals.keys()].sort((a, b) => totals.get(b) - totals.get(a));
  const colorOf = colorMapFor(callTypes);

  const bins = binToolUseEvents(events, TOOL_USE_BINS);
  const nonEmpty = bins
    .map((bin, i) => ({ i, total: [...bin.values()].reduce((a, v) => a + v, 0) }))
    .filter((b) => b.total > 0);

  const nObserved = (detail.state.trials || []).filter((t) => t.status === "observed").length;
  const nTurns = (detail.trace || []).filter((e) => e.role === "assistant").length;

  const statCards = `
    <div class="stat"><div class="value">${events.length}</div><div class="label">lenz calls</div></div>
    <div class="stat"><div class="value">${nTurns}</div><div class="label">Agent turns</div></div>
    <div class="stat"><div class="value">${nObserved ? fmtNum(events.length / nObserved, 2) : "—"}</div><div class="label">Calls / evaluation</div></div>
    <div class="stat"><div class="value">${callTypes.length}</div><div class="label">Distinct call types</div></div>
  `;

  let chartHtml;
  if (nonEmpty.length < 2) {
    chartHtml = `<div class="empty-note">Not enough spread of calls across the campaign to chart yet.</div>`;
  } else {
    chartHtml = `<div id="tool-use-chart-wrap"></div><div class="legend wrap" id="tool-use-legend"></div>`;
  }

  const breakdownRows = callTypes
    .map((ct) => {
      const count = totals.get(ct);
      const pct = ((100 * count) / events.length).toFixed(1);
      return `<tr><td><span class="legend-swatch" style="background:${colorOf(ct)};display:inline-block;margin-right:6px"></span><span class="mono">${escapeHtml(ct)}</span></td><td>${count}</td><td>${pct}%</td></tr>`;
    })
    .join("");

  const reconfigs = detail.reconfigurations || [];
  const reconfigRows = reconfigs.length
    ? [...reconfigs]
        .reverse()
        .map((r) => {
          const { ts, command, ...payload } = r;
          return `<tr><td class="mono">${escapeHtml(command)}</td><td class="mono">${escapeHtml(JSON.stringify(payload))}</td></tr>`;
        })
        .join("")
    : `<tr><td colspan="2" class="empty-note">No mid-campaign reconfigurations (set-acqf/set-bounds/set-objectives/set-constraints/set-surrogate) -- the backend stayed as configured at <code>create</code>.</td></tr>`;

  const usage = detail.meta && detail.meta.usage;
  let usageHtml;
  if (usage) {
    const total = (usage.input_tokens || 0) + (usage.output_tokens || 0);
    const rows = [
      ["Input tokens", usage.input_tokens],
      ["Output tokens", usage.output_tokens],
      ["Cache read tokens", usage.cache_read_tokens],
      ["Cache creation tokens", usage.cache_creation_tokens],
      ["Total (input + output)", total],
    ]
      .map(([label, v]) => `<tr><td>${escapeHtml(label)}</td><td class="mono">${v === null || v === undefined ? "—" : v.toLocaleString()}</td></tr>`)
      .join("");
    usageHtml = `
      <div class="card">
        <h3>LLM usage</h3>
        <div class="sub" style="margin-bottom:8px">Summed across all ${nTurns} turns. Token cost varies widely by model and reasoning level.</div>
        <table class="kv-table">${rows}</table>
      </div>`;
  } else {
    usageHtml = `
      <div class="card">
        <h3>LLM usage</h3>
        <div class="empty-note">Not available for this run -- either it predates usage tracking, or the provider/gateway didn't report token counts for streaming responses. Rerun to capture it.</div>
      </div>`;
  }

  panel.innerHTML = `
    <div class="card">
      <h3>Tool use over the campaign</h3>
      <div class="sub" style="margin-bottom:12px">Relative frequency of each <code>lenz</code> call type, binned over normalized trial progress. x-axis: fraction of this run's observed evaluations completed so far when each call was made.</div>
      <div class="stat-row" style="margin-bottom:12px">${statCards}</div>
      ${chartHtml}
    </div>
    <div class="card">
      <h3>Call type breakdown</h3>
      <table><thead><tr><th>call type</th><th>count</th><th>% of calls</th></tr></thead><tbody>${breakdownRows}</tbody></table>
    </div>
    <div class="card">
      <h3>Mid-campaign reconfigurations</h3>
      <div class="sub" style="margin-bottom:8px">Backend changes Sara made after <code>create</code> (mid-run problem reformulation).</div>
      <table><thead><tr><th>command</th><th>new value</th></tr></thead><tbody>${reconfigRows}</tbody></table>
    </div>
    ${usageHtml}
  `;

  if (nonEmpty.length < 2) return;

  function draw() {
    const container = document.getElementById("tool-use-chart-wrap");
    if (!container) return;
    const W = container.clientWidth || 600;
    const H = 280;
    const margin = { top: 10, right: 20, bottom: 28, left: 40 };
    const plotW = Math.max(10, W - margin.left - margin.right);
    const plotH = H - margin.top - margin.bottom;

    const xScale = (progress) => margin.left + progress * plotW;
    const yScale = (frac) => margin.top + plotH - frac * plotH;

    const stacks = nonEmpty.map(({ i, total }) => {
      const bin = bins[i];
      let cum = 0;
      const layers = callTypes.map((ct) => {
        const frac = (bin.get(ct) || 0) / total;
        const layer = { y0: cum, y1: cum + frac };
        cum += frac;
        return layer;
      });
      return { x: (i + 0.5) / TOOL_USE_BINS, layers };
    });

    let areas = "";
    callTypes.forEach((ct, idx) => {
      const top = stacks.map((s) => `${xScale(s.x)},${yScale(s.layers[idx].y1)}`).join(" L");
      const bottom = [...stacks].reverse().map((s) => `${xScale(s.x)},${yScale(s.layers[idx].y0)}`).join(" L");
      areas += `<path d="M${top} L${bottom} Z" fill="${colorOf(ct)}" opacity="0.85"></path>`;
    });

    let gridlines = "";
    let yLabels = "";
    for (const frac of [0, 0.25, 0.5, 0.75, 1.0]) {
      const y = yScale(frac);
      gridlines += `<line class="gridline" x1="${margin.left}" x2="${W - margin.right}" y1="${y}" y2="${y}"></line>`;
      yLabels += `<text class="axis-label" x="${margin.left - 8}" y="${y + 3}" text-anchor="end">${frac.toFixed(2)}</text>`;
    }
    let xLabels = "";
    for (const frac of [0, 0.2, 0.4, 0.6, 0.8, 1.0]) {
      const x = xScale(frac);
      const anchor = frac === 0 ? "start" : frac === 1 ? "end" : "middle";
      xLabels += `<text class="axis-label" x="${x}" y="${H - 6}" text-anchor="${anchor}">${frac.toFixed(1)}</text>`;
    }

    container.innerHTML = `
      <svg class="chart" viewBox="0 0 ${W} ${H}">
        ${gridlines}
        <line class="baseline" x1="${margin.left}" x2="${W - margin.right}" y1="${margin.top + plotH}" y2="${margin.top + plotH}"></line>
        ${areas}
        ${yLabels}
        ${xLabels}
        <text class="axis-label" x="${(margin.left + W - margin.right) / 2}" y="${H + 2}" text-anchor="middle">normalized trial progress</text>
        <g id="tool-use-hover"></g>
      </svg>
    `;

    const svg = container.querySelector("svg");
    const hoverLayer = container.querySelector("#tool-use-hover");
    const tooltip = document.getElementById("tooltip");

    svg.addEventListener("mousemove", (evt) => {
      const rect = svg.getBoundingClientRect();
      const mx = ((evt.clientX - rect.left) / rect.width) * W;
      const progress = Math.min(1, Math.max(0, (mx - margin.left) / plotW));
      let nearest = stacks[0];
      let bestDist = Infinity;
      for (const s of stacks) {
        const d = Math.abs(s.x - progress);
        if (d < bestDist) {
          bestDist = d;
          nearest = s;
        }
      }
      hoverLayer.innerHTML = `<line class="crosshair" x1="${xScale(nearest.x)}" x2="${xScale(nearest.x)}" y1="${margin.top}" y2="${margin.top + plotH}"></line>`;
      const tipRows = callTypes
        .map((ct, idx) => {
          const layer = nearest.layers[idx];
          const pct = ((layer.y1 - layer.y0) * 100).toFixed(0);
          if (pct === "0") return "";
          return `<div><span style="color:${colorOf(ct)}">●</span> ${escapeHtml(ct)}: ${pct}%</div>`;
        })
        .join("");
      tooltip.style.display = "block";
      tooltip.style.left = evt.pageX + 14 + "px";
      tooltip.style.top = evt.pageY - 10 + "px";
      tooltip.innerHTML = `<div style="color:var(--text-muted)">progress ≈ ${nearest.x.toFixed(2)}</div>${tipRows}`;
    });
    svg.addEventListener("mouseleave", () => {
      hoverLayer.innerHTML = "";
      tooltip.style.display = "none";
    });
  }

  draw();
  const wrap = document.getElementById("tool-use-chart-wrap");
  if (!wrap._resizeObserver) {
    let raf = null;
    const ro = new ResizeObserver(() => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(draw);
    });
    ro.observe(wrap);
    wrap._resizeObserver = ro;
  }

  document.getElementById("tool-use-legend").innerHTML = callTypes
    .map((ct) => `<div class="legend-item"><span class="legend-swatch" style="background:${colorOf(ct)}"></span>${escapeHtml(ct)} (${totals.get(ct)})</div>`)
    .join("");
}

// -- kernel population -------------------------------------------------------------

function renderKernels(detail) {
  const panel = document.getElementById("panel-kernels");
  const shelf = detail.state.shelf || {};
  const cake = (detail.state.plugins && detail.state.plugins.cake) || shelf;
  const populations = cake.kernel_populations || {};
  const legacy = cake.kernel_population || shelf.kernel_population || [];
  const targets = Object.keys(populations).length
    ? Object.keys(populations)
    : legacy.length
      ? [shelf.objectives?.[0]?.metric || "objective"]
      : [];

  const cards = targets.map((target) => {
    const population = [...(populations[target] || (targets.length === 1 && legacy.length ? legacy : []))].sort(
      (a, b) => (a.bic ?? Infinity) - (b.bic ?? Infinity)
    );
    const bestExpr = population.length ? population[0].expression : null;
    const state = cake.kernel_evolution_states?.[target] || cake.kernel_evolution_state || {};
    const rows = population
      .map(
        (m) => `<tr>
        <td class="mono ${m.expression === bestExpr ? "kernel-best" : ""}">${escapeHtml(m.expression)}${m.expression === bestExpr ? " ★" : ""}</td>
        <td>${fmtNum(m.bic, 2)}</td>
        <td>${m.generation}</td>
      </tr>`
      )
      .join("");
    return `
    <div class="card">
      <h3>${escapeHtml(target)}</h3>
      <div class="stat-row" style="margin-bottom:12px">
        <div class="stat"><div class="value">${state.generation ?? 0}</div><div class="label">Generation</div></div>
        <div class="stat"><div class="value">${state.frozen ? "Yes" : "No"}</div><div class="label">Frozen</div></div>
        <div class="stat"><div class="value" style="font-family:ui-monospace,monospace;font-size:16px">${escapeHtml(bestExpr || "—")}</div><div class="label">Best kernel</div></div>
      </div>
      <table>
        <thead><tr><th>expression</th><th>BIC (lower = better)</th><th>introduced at gen</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="3" class="empty-note">No population yet — evolution hasn’t triggered.</td></tr>'}</tbody>
      </table>
    </div>`;
  }).join("");

  const gens = detail.kernel_generations || [];
  const genRows = [...gens]
    .reverse()
    .map(
      (g) => `<tr>
        <td>${g.generation}</td>
        <td>${escapeHtml(g.target || "—")}</td>
        <td class="mono kernel-best">${escapeHtml(g.best || "—")}</td>
        <td class="mono">${escapeHtml((g.population || []).join(", "))}</td>
      </tr>`
    )
    .join("");

  panel.innerHTML = `
    ${cards || '<div class="card"><div class="empty-note">No kernel populations yet.</div></div>'}
    <div class="card">
      <h3>Evolution history</h3>
      <table>
        <thead><tr><th>gen</th><th>metric</th><th>best</th><th>population</th></tr></thead>
        <tbody>${genRows || '<tr><td colspan="4" class="empty-note">No evolution rounds recorded yet.</td></tr>'}</tbody>
      </table>
    </div>
  `;
}

// -- compare view -------------------------------------------------------------

async function renderCompare() {
  hideAllMainViews();
  const view = document.getElementById("compare-view");
  view.style.display = "block";
  state.mode = "compare";

  const names = [...state.selected];
  const details = await Promise.all(names.map((n) => fetch(`/api/runs/${encodeURIComponent(n)}`).then((r) => r.json())));
  const withConv = details.filter((d) => d.convergence && d.convergence.points.length >= 2);
  const colorOf = colorMapFor(details.map((d) => d.name));

  const rows = details
    .map((d) => {
      const best = d.convergence && d.convergence.points.length ? d.convergence.points[d.convergence.points.length - 1].best : null;
      const nObs = (d.state.trials || []).filter((t) => t.status === "observed").length;
      return `<tr>
        <td><span class="run-dot" style="background:${colorOf(d.name)};display:inline-block;margin-top:0"></span> ${escapeHtml(d.name)}</td>
        <td class="mono">${escapeHtml((d.meta && d.meta.model) || "—")}</td>
        <td>${nObs}</td>
        <td>${fmtNum(best)}</td>
        <td>${fmtDuration(d.duration_seconds)}</td>
      </tr>`;
    })
    .join("");

  view.innerHTML = `
    <div id="main-header">
      <div>
        <h2>Comparing ${details.length} runs</h2>
      </div>
    </div>
    <div class="card">
      <h3>Convergence</h3>
      ${withConv.length >= 1 ? `<div id="chart-wrap"></div><div class="legend" id="chart-legend"></div>` : '<div class="empty-note">None of the selected runs have single-objective convergence data.</div>'}
    </div>
    <div class="card">
      <h3>Summary</h3>
      <table>
        <thead><tr><th>run</th><th>model</th><th>evals</th><th>best</th><th>duration</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;

  if (withConv.length >= 1) {
    renderChart(
      document.getElementById("chart-wrap"),
      withConv.map((d) => ({ name: d.name, convergence: d.convergence })),
      colorOf
    );
  }
}

// -- experiment compare (condition-level true regret) -----------------------------

function withEvalZero(trace) {
  return [Infinity, ...trace];
}

function legendOrigin(corner, x0, y1, x1, y0, legendW, legendH, inset = 4) {
  const left = corner[1] === "l" ? x0 + inset : x1 - legendW - inset;
  const top = corner[0] === "t" ? y1 + inset : y0 - legendH - inset;
  return { left, top };
}

function pickLegendBox(points, x0, y1, x1, y0, legendW, legendH, inset = 4) {
  const order = ["tr", "bl", "br", "tl"];
  let best = order[0];
  let bestHits = Infinity;
  for (const corner of order) {
    const { left, top } = legendOrigin(corner, x0, y1, x1, y0, legendW, legendH, inset);
    const right = left + legendW;
    const bottom = top + legendH;
    let hits = 0;
    for (const [x, y] of points) {
      if (x >= left && x <= right && y >= top && y <= bottom) hits += 1;
    }
    if (hits < bestHits) {
      best = corner;
      bestHits = hits;
    }
  }
  return legendOrigin(best, x0, y1, x1, y0, legendW, legendH, inset);
}

function renderRegretChart(container, traces, colorOf) {
  const labels = Object.keys(traces);
  if (!labels.length) return;
  colorOf = colorOf || colorMapFor(labels);

  function draw() {
    const W = container.clientWidth || 820;
    const H = Math.max(360, Math.round(W * 0.52));
    const pad = { l: 60, r: 20, t: 16, b: 36 };
    const x0 = pad.l;
    const y1 = pad.t;
    const x1 = W - pad.r;
    const y0 = H - pad.b;

    const plotTraces = {};
    for (const label of labels) plotTraces[label] = withEvalZero(traces[label]);
    const finiteVals = labels.flatMap((label) => plotTraces[label].filter((v) => Number.isFinite(v)));
    let yMin = finiteVals.length ? Math.min(0, ...finiteVals) : 0;
    let yMax = finiteVals.length ? Math.max(...finiteVals) : 1;
    if (yMax <= yMin) yMax = yMin + 1;
    const nMax = Math.max(...labels.map((label) => plotTraces[label].length));

    const X = (i) => x0 + ((x1 - x0) * i) / Math.max(nMax - 1, 1);
    const Y = (v) => {
      if (!Number.isFinite(v)) return y1;
      return y1 + (y0 - y1) * (1 - (v - yMin) / (yMax - yMin));
    };

    const padTrace = (trace) => {
      if (trace.length >= nMax) return trace.slice(0, nMax);
      return trace.concat(Array(nMax - trace.length).fill(trace[trace.length - 1]));
    };

    let grid = "";
    for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
      const v = yMin + frac * (yMax - yMin);
      const yy = Y(v);
      grid += `<line class="gridline" x1="${x0}" x2="${x1}" y1="${yy}" y2="${yy}"></line>`;
      grid += `<text class="axis-label" x="${x0 - 8}" y="${yy + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
    }

    let seriesSvg = "";
    const legendRows = [];
    labels.forEach((label, i) => {
      const trace = traces[label];
      const nObs = trace.length;
      const evalNote = nObs < nMax - 1 ? `, ${nObs} eval${nObs === 1 ? "" : "s"}` : "";
      legendRows.push({ label, text: `${label} (best ${fmtNum(trace[trace.length - 1], 4)}${evalNote})`, color: colorOf(label) });
      const padded = padTrace(plotTraces[label]);
      const pts = padded.map((v, j) => `${X(j)},${Y(v)}`).join(" ");
      const color = colorOf(label);
      seriesSvg += `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2"></polyline>`;
      const mx = X(nObs);
      const my = Y(trace[trace.length - 1]);
      const markerR = nObs === 1 ? 6 : 4;
      seriesSvg += `<circle cx="${mx}" cy="${my}" r="${markerR}" fill="${color}" stroke="var(--text-muted)" stroke-width="1"></circle>`;
    });

    const legendRowH = 18;
    const legendPad = 6;
    const legendW = 300;
    const legendH = legendRows.length * legendRowH + legendPad * 2;
    const boxW = legendW + legendPad;
    const curvePts = [];
    for (const label of labels) {
      for (const [j, v] of padTrace(plotTraces[label]).entries()) curvePts.push([X(j), Y(v)]);
    }
    const { left: legendLeft, top: legendTop } = pickLegendBox(curvePts, x0, y1, x1, y0, boxW, legendH);
    let legendSvg = `<rect x="${legendLeft}" y="${legendTop}" width="${boxW}" height="${legendH}" rx="4" fill="var(--text-primary)" fill-opacity="0.08" stroke="var(--border)"></rect>`;
    legendRows.forEach((row, i) => {
      const cy = legendTop + legendPad + i * legendRowH + legendRowH / 2;
      legendSvg += `<circle cx="${legendLeft + legendPad}" cy="${cy}" r="4" fill="${row.color}"></circle>`;
      legendSvg += `<text class="axis-label" x="${legendLeft + legendPad + 10}" y="${cy + 4}">${escapeHtml(row.text)}</text>`;
    });

    container.innerHTML = `
      <svg class="regret-chart" viewBox="0 0 ${W} ${H}">
        ${grid}
        <line class="baseline" x1="${x0}" x2="${x1}" y1="${y0}" y2="${y0}"></line>
        <line class="baseline" x1="${x0}" x2="${x0}" y1="${y0}" y2="${y1}"></line>
        <text class="axis-label" x="${x0}" y="${y0 + 22}">0</text>
        <text class="axis-label" x="${x1}" y="${y0 + 22}" text-anchor="end">${nMax - 1} evals</text>
        ${seriesSvg}
        ${legendSvg}
        <g id="regret-hover"></g>
      </svg>
    `;

    const svg = container.querySelector("svg");
    const hoverLayer = container.querySelector("#regret-hover");
    const tooltip = document.getElementById("tooltip");

    svg.addEventListener("mousemove", (evt) => {
      const rect = svg.getBoundingClientRect();
      const mx = ((evt.clientX - rect.left) / rect.width) * W;
      const evalIdx = Math.round(((mx - x0) / (x1 - x0)) * (nMax - 1));
      const clamped = Math.max(0, Math.min(nMax - 1, evalIdx));
      let tipRows = `<div style="color:var(--text-muted)">eval ${clamped}</div>`;
      let dots = `<line class="crosshair" x1="${X(clamped)}" x2="${X(clamped)}" y1="${y1}" y2="${y0}"></line>`;
      labels.forEach((label) => {
        const trace = plotTraces[label];
        const idx = Math.min(clamped, trace.length - 1);
        const v = trace[idx];
        const color = colorOf(label);
        dots += `<circle class="hover-dot" cx="${X(clamped)}" cy="${Y(v)}" r="4.5" style="fill:${color}"></circle>`;
        const shown = Number.isFinite(v) ? fmtNum(v) : "—";
        tipRows += `<div><span style="color:${color}">●</span> ${escapeHtml(label)}: ${shown}</div>`;
      });
      hoverLayer.innerHTML = dots;
      tooltip.style.display = "block";
      tooltip.style.left = evt.pageX + 14 + "px";
      tooltip.style.top = evt.pageY - 10 + "px";
      tooltip.innerHTML = tipRows;
    });
    svg.addEventListener("mouseleave", () => {
      hoverLayer.innerHTML = "";
      tooltip.style.display = "none";
    });
  }

  draw();
  if (!container._resizeObserver) {
    let raf = null;
    const ro = new ResizeObserver(() => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(draw);
    });
    ro.observe(container);
    container._resizeObserver = ro;
  }
}

function statusLabel(status) {
  if (status === "stopped early") return "stopped early";
  if (status === "running") return "running";
  if (status === "failed") return "failed";
  return "complete";
}

function renderExperimentCompare(detail) {
  const view = document.getElementById("experiment-compare-view");
  const traces = detail.traces || {};
  const conditions = detail.conditions || Object.keys(traces).map((name) => ({
    name,
    n_evals: traces[name].length,
    regret_eval1: traces[name][0],
    best_regret: traces[name][traces[name].length - 1],
    budget: null,
    status: "complete",
  }));

  const colorOf = colorMapFor([
    ...conditions.map((c) => c.name),
    ...Object.keys(traces),
  ]);

  const rows = conditions
    .map((c) => {
      const evals =
        c.budget != null ? `${c.n_evals}/${c.budget}` : String(c.n_evals);
      const label = `<span class="run-dot" style="background:${colorOf(c.name)};display:inline-block;margin-top:0"></span> ${escapeHtml(c.name)}`;
      const nameCell = c.run_name
        ? `<button type="button" class="condition-link" data-run="${escapeHtml(c.run_name)}">${label}</button>`
        : label;
      return `<tr>
        <td>${nameCell}</td>
        <td class="mono">${fmtNum(c.regret_eval1, 4)}</td>
        <td class="mono">${fmtNum(c.best_regret, 6)}</td>
        <td>${escapeHtml(evals)}</td>
        <td>${escapeHtml(statusLabel(c.status))}</td>
      </tr>`;
    })
    .join("");

  view.innerHTML = `
    <div id="main-header">
      <div>
        <h2>${escapeHtml(detail.title)}</h2>
        ${detail.caption ? `<div class="sub experiment-caption">${escapeHtml(detail.caption)}</div>` : ""}
      </div>
      <div class="experiment-actions">
        <button id="refresh-group-btn" title="Reload">Refresh</button>
      </div>
    </div>
    <div class="card">
      <h3>Regret</h3>
      <div class="chart-note">Y-axis: best-so-far true regret vs hidden optimum.</div>
      ${
        conditions.length
          ? `<div id="regret-chart-wrap"></div>`
          : '<div class="empty-note">No data yet.</div>'
      }
    </div>
    <div class="card">
      <h3>Summary</h3>
      ${
        conditions.length
          ? `<table>
        <thead><tr><th>condition</th><th>regret @ 1</th><th>best regret</th><th>evals</th><th>status</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`
          : `<div class="empty-note">No data yet.</div>`
      }
    </div>
  `;

  document.getElementById("refresh-group-btn")?.addEventListener("click", () => selectCompareGroup(detail.name));
  view.querySelectorAll(".condition-link").forEach((btn) => {
    btn.addEventListener("click", () => selectRun(btn.dataset.run));
  });

  if (conditions.length) renderRegretChart(document.getElementById("regret-chart-wrap"), traces, colorOf);
}

// -- boot -------------------------------------------------------------

initTheme();
initSearchAndCompare();
initSidebarNav();
renderGroupPicker();
Promise.allSettled([loadRuns(), loadCompareGroups()]).then(() => {
  readFiltersFromUrl();
  applyFilters();
  const params = new URLSearchParams(location.search);
  const groupParam = params.get("group");
  const compareParam = params.get("compare");
  const raw = decodeURIComponent(location.hash.slice(1));
  const sep = raw.lastIndexOf("|");
  const runName = sep === -1 ? raw : raw.slice(0, sep);
  const tab = sep === -1 ? null : raw.slice(sep + 1);

  if (groupParam && state.compareGroups.some((g) => g.name === groupParam)) {
    applySidebarModeUI("experiments");
    selectCompareGroup(groupParam);
    return;
  }
  if (compareParam) {
    for (const name of compareParam.split(",")) {
      if (state.runs.some((r) => r.name === name)) state.selected.add(name);
    }
    if (state.selected.size >= 2) {
      renderSidebar();
      updateCompareBar();
      renderCompare();
      return;
    }
  }
  if (runName && state.runs.some((r) => r.name === runName)) {
    if (tab) state.activeTab = tab;
    selectRun(runName);
  }
});
