/* Cross-seed mean +/- standard-error regret viewer. Layout and interaction
   mirror plugbo's sara-viz compare chart; the only new thing drawn here is
   the shaded standard-error band per condition. */

(() => {
  const COLORS = {
    vanilla: "var(--cat-4)",
    cake: "var(--cat-1)",
    turbo: "var(--cat-6)",
    pibo: "var(--cat-2)",
    "sara-lenz": "var(--cat-3)",
    "sara-lenz-cake": "var(--cat-7)",
    "sara-lenz-pibo": "var(--cat-8)",
    "sara-lenz-turbo": "var(--cat-8)",
    "sara-only": "var(--cat-5)",
  };
  const FALLBACK = ["var(--cat-1)", "var(--cat-2)", "var(--cat-3)", "var(--cat-4)", "var(--cat-5)", "var(--cat-6)", "var(--cat-7)", "var(--cat-8)"];

  function colorOf(name, i) {
    return COLORS[name] || FALLBACK[i % FALLBACK.length];
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function fmtNum(n, digits = 4) {
    if (n === null || n === undefined || Number.isNaN(n) || !Number.isFinite(n)) return "—";
    if (n === 0) return "0";
    if (Math.abs(n) >= 1e5 || Math.abs(n) < 1e-3) return n.toExponential(2);
    let s = n.toFixed(digits);
    if (s.includes(".")) s = s.replace(/0+$/, "").replace(/\.$/, "");
    return s;
  }

  function padFlat(trace, n) {
    if (trace.length >= n) return trace.slice(0, n);
    return trace.concat(Array(n - trace.length).fill(trace[trace.length - 1]));
  }

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

  const tooltip = document.getElementById("tooltip");
  window.addEventListener("scroll", () => { tooltip.style.display = "none"; }, { passive: true });

  const hidden = new Set();
  let currentConditions = [];

  function draw() {
    const container = document.getElementById("chart-container");
    const conditions = currentConditions;
    if (!conditions.length) {
      container.innerHTML = "<p class='empty-note'>No scoreable conditions in this group.</p>";
      return;
    }
    const visible = conditions.filter((c) => !hidden.has(c.name));
    const active = visible.length ? visible : conditions;

    const W = Math.max(container.clientWidth || 820, 320);
    const H = Math.max(520, Math.round(W * 0.66));
    const pad = { l: 58, r: 18, t: 16, b: 36 };
    const x0 = pad.l, y1 = pad.t, x1 = W - pad.r, y0 = H - pad.b;

    const nMax = Math.max(...conditions.map((c) => c.mean.length)) + 1;

    const plotMean = {}, plotUpper = {}, plotLower = {};
    for (const c of active) {
      const mean = withEvalZero(c.mean);
      const upper = withEvalZero(c.mean.map((m, i) => m + c.stderr[i]));
      const lower = withEvalZero(c.mean.map((m, i) => m - c.stderr[i]));
      plotMean[c.name] = mean;
      plotUpper[c.name] = upper;
      plotLower[c.name] = lower;
    }

    // Regret traces span orders of magnitude (a big warm-start miss down to
    // a near-zero final gap), so a linear axis leaves the whole converged
    // tail crushed against the bottom. Log scale spreads it out; nonpositive
    // points (a lower SE band dipping through zero, or a literal 0 regret)
    // get floored to half the smallest positive value actually on screen.
    const finiteVals = active.flatMap((c) => plotMean[c.name].filter((v) => Number.isFinite(v)));
    const positiveVals = finiteVals.filter((v) => v > 0);
    let yMin = positiveVals.length ? Math.min(...positiveVals) : 1e-6;
    let yMax = positiveVals.length ? Math.max(...positiveVals) : 1;
    if (yMax <= yMin) yMax = yMin * 10;
    const floor = yMin * 0.5;
    const logMin = Math.log10(floor);
    const logMax = Math.log10(yMax);

    const X = (i) => x0 + ((x1 - x0) * i) / Math.max(nMax - 1, 1);
    const Y = (v) => {
      if (!Number.isFinite(v)) return y1;
      const lv = Math.log10(Math.max(v, floor));
      return y1 + (y0 - y1) * (1 - (lv - logMin) / (logMax - logMin));
    };

    let grid = "";
    for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
      const v = Math.pow(10, logMin + frac * (logMax - logMin));
      const yy = y1 + (y0 - y1) * (1 - frac);
      grid += `<line class="gridline" x1="${x0}" x2="${x1}" y1="${yy}" y2="${yy}"></line>`;
      grid += `<text class="axis-label" x="${x0 - 8}" y="${yy + 4}" text-anchor="end">${fmtNum(v, 3)}</text>`;
    }

    let seriesSvg = "";
    const legendRows = [];
    const curvePts = [];
    conditions.forEach((c, i) => {
      const color = colorOf(c.name, i);
      const nEvals = c.mean.length;
      const seNote = ` ±${fmtNum(c.best_stderr, 4)}`;
      const seedNote = `, ${c.n_seeds} seed${c.n_seeds === 1 ? "" : "s"}`;
      legendRows.push({ name: c.name, text: `${c.name} (final ${fmtNum(c.best_mean, 4)}${seNote}${seedNote})`, color, faded: hidden.has(c.name) });
      if (hidden.has(c.name)) return;
      const nMean = padFlat(plotMean[c.name], nMax);
      const nUpper = padFlat(plotUpper[c.name], nMax);
      const nLower = padFlat(plotLower[c.name], nMax);
      const bandPts = nUpper.map((v, j) => `${X(j)},${Y(v)}`).join(" L ") +
        " L " + nLower.slice().reverse().map((v, j) => `${X(nMax - 1 - j)},${Y(v)}`).join(" L ");
      seriesSvg += `<path d="M ${bandPts} Z" fill="${color}" class="band"></path>`;
      const pts = nMean.map((v, j) => `${X(j)},${Y(v)}`).join(" ");
      seriesSvg += `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2"></polyline>`;
      seriesSvg += `<circle cx="${X(nEvals)}" cy="${Y(c.mean[nEvals - 1])}" r="4" fill="${color}" class="end-marker"></circle>`;
      nMean.forEach((v, j) => curvePts.push([X(j), Y(v)]));
    });

    const legendRowH = 18, legendPad = 6, legendW = 330;
    const legendH = legendRows.length * legendRowH + legendPad * 2;
    const boxW = legendW + legendPad;
    const { left: legendLeft, top: legendTop } = pickLegendBox(curvePts, x0, y1, x1, y0, boxW, legendH);

    let legendSvg = `<rect class="legend-box" x="${legendLeft}" y="${legendTop}" width="${boxW}" height="${legendH}" rx="4"></rect>`;
    legendRows.forEach((row, i) => {
      const cy = legendTop + legendPad + i * legendRowH + legendRowH / 2;
      const opacity = row.faded ? "0.35" : "1";
      legendSvg += `<g class="legend-row" data-name="${escapeHtml(row.name)}" opacity="${opacity}" style="cursor:pointer">`;
      legendSvg += `<rect x="${legendLeft}" y="${cy - legendRowH / 2}" width="${boxW}" height="${legendRowH}" fill="transparent"></rect>`;
      legendSvg += `<circle cx="${legendLeft + legendPad + 6}" cy="${cy}" r="4" fill="${row.color}"></circle>`;
      legendSvg += `<text class="axis-label" x="${legendLeft + legendPad + 16}" y="${cy + 4}">${escapeHtml(row.text)}</text>`;
      legendSvg += `</g>`;
    });

    container.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mean best-so-far true regret, +/- standard error across seeds.">
        ${grid}
        <line class="baseline" x1="${x0}" x2="${x1}" y1="${y0}" y2="${y0}"></line>
        <line class="baseline" x1="${x0}" x2="${x0}" y1="${y0}" y2="${y1}"></line>
        <text class="axis-label" x="${x0}" y="${y0 + 22}">0</text>
        <text class="axis-label" x="${x1}" y="${y0 + 22}" text-anchor="end">${nMax - 1} evals</text>
        ${seriesSvg}
        ${legendSvg}
        <g class="hover-layer"></g>
      </svg>
    `;

    const svg = container.querySelector("svg");
    const hoverLayer = container.querySelector(".hover-layer");

    svg.querySelectorAll(".legend-row").forEach((row) => {
      row.addEventListener("click", (evt) => {
        evt.stopPropagation();
        const name = row.getAttribute("data-name");
        if (hidden.has(name)) hidden.delete(name);
        else if (hidden.size < conditions.length - 1) hidden.add(name);
        draw();
      });
    });

    svg.addEventListener("mousemove", (evt) => {
      const rect = svg.getBoundingClientRect();
      const mx = ((evt.clientX - rect.left) / rect.width) * W;
      const evalIdx = Math.round(((mx - x0) / (x1 - x0)) * (nMax - 1));
      const clamped = Math.max(0, Math.min(nMax - 1, evalIdx));
      let tipRows = `<div class="tip-step">eval ${clamped}</div>`;
      let dots = `<line class="crosshair" x1="${X(clamped)}" x2="${X(clamped)}" y1="${y1}" y2="${y0}"></line>`;
      active.forEach((c) => {
        const mean = plotMean[c.name];
        const idx = Math.min(clamped, mean.length - 1);
        const v = mean[idx];
        const se = plotUpper[c.name][idx] - v;
        const color = colorOf(c.name, conditions.findIndex((x) => x.name === c.name));
        dots += `<circle cx="${X(clamped)}" cy="${Y(v)}" r="4.5" style="fill:${color}"></circle>`;
        tipRows += `<div><span style="color:${color}">●</span> ${escapeHtml(c.name)}: ${fmtNum(v)} ± ${fmtNum(se)}</div>`;
      });
      hoverLayer.innerHTML = dots;
      tooltip.style.display = "block";
      tooltip.style.left = `${evt.clientX + 14}px`;
      tooltip.style.top = `${evt.clientY - 10}px`;
      tooltip.innerHTML = tipRows;
    });
    svg.addEventListener("mouseleave", () => {
      hoverLayer.innerHTML = "";
      tooltip.style.display = "none";
    });
  }

  function renderSummary(conditions) {
    const body = document.getElementById("summary-body");
    body.innerHTML = conditions
      .map(
        (c) => `<tr>
          <td>${escapeHtml(c.name)}</td>
          <td>${c.seeds.join(", ")}</td>
          <td>${fmtNum(c.best_mean)} &plusmn; ${fmtNum(c.best_stderr)}</td>
          <td>${escapeHtml(c.status)}</td>
        </tr>`
      )
      .join("");
  }

  async function loadGroup(name) {
    const res = await fetch(`/api/merge-groups/${encodeURIComponent(name)}`);
    const data = await res.json();
    hidden.clear();
    currentConditions = data.conditions || [];
    document.getElementById("empty-main").style.display = "none";
    document.getElementById("group-view").style.display = "block";
    document.getElementById("group-title").textContent = data.heading || data.name;
    document.getElementById("group-caption").textContent = data.caption || "";
    draw();
    renderSummary(currentConditions);
  }

  async function boot() {
    const res = await fetch("/api/merge-groups");
    const groups = await res.json();
    const list = document.getElementById("group-list");
    if (!groups.length) {
      document.getElementById("empty-state").style.display = "block";
      return;
    }
    list.innerHTML = groups
      .map(
        (g, i) => `<li data-name="${escapeHtml(g.name)}" data-idx="${i}">
          <div class="group-item-title">${escapeHtml(g.heading || g.name)}</div>
          <div class="group-item-sub">seeds ${g.seeds.join(", ")}</div>
        </li>`
      )
      .join("");
    list.querySelectorAll("li").forEach((li) => {
      li.addEventListener("click", () => {
        list.querySelectorAll("li").forEach((el) => el.classList.remove("active"));
        li.classList.add("active");
        loadGroup(li.getAttribute("data-name"));
      });
    });
    list.querySelector("li").click();

    window.addEventListener("resize", () => draw());
  }

  document.getElementById("theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const current = root.getAttribute("data-theme");
    root.setAttribute("data-theme", current === "dark" ? "light" : "dark");
    draw();
  });

  boot();
})();
