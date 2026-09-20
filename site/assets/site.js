(() => {
  const root = document.getElementById("rsi-academic");
  const { base, math, methods, videos, images, all } = window.DATALITE;
  const state = { track: "math", math: "opsd", method: "opsd" };
  const esc = (value) =>
    String(value).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  function bar(row, unit) {
    const [label, value, kind] = row;
    const shown =
      unit === "normalized" ? value.toFixed(5) : value.toFixed(2) + "%";
    const width = unit === "normalized" ? value * 100 : value;
    return (
      '<div class="rs-bar-row"><div class="rs-bar-label"><span>' +
      esc(label) +
      "</span><b>" +
      shown +
      '</b></div><div class="rs-bar-track" role="img" aria-label="' +
      esc(label) +
      ": " +
      shown +
      '"><div class="rs-bar-mark rs-' +
      kind +
      '" style="width:' +
      width +
      '%"></div></div></div>'
    );
  }
  root.querySelector("#rs-image-charts").innerHTML = images
    .map(
      (x) =>
        '<div class="rs-chart-panel"><h3>' +
        esc(x.title) +
        '</h3><p class="rs-chart-sub">Normalized transfer composite</p>' +
        x.rows.map((r) => bar(r, "normalized")).join("") +
        '<div class="rs-chart-axis"><span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span></div><p class="rs-chart-label">Normalized score · ' +
        x.gain +
        "</p></div>",
    )
    .join("");
  root.querySelector("#rs-video-charts").innerHTML = videos
    .map(
      ([name, b, f]) =>
        '<div class="rs-video-model"><div class="rs-video-head">' +
        esc(name) +
        "<span>+" +
        (f - b).toFixed(2) +
        " pp</span></div>" +
        [
          ["Base", b, "base"],
          ["SFT", f, "ours"],
        ]
          .map(
            ([label, value, kind]) =>
              '<div class="rs-paired-row"><span>' +
              esc(label) +
              '</span><div class="rs-bar-track" role="img" aria-label="' +
              esc(name) +
              " " +
              esc(label) +
              ": " +
              value.toFixed(2) +
              ' percent"><div class="rs-bar-mark rs-' +
              kind +
              '" style="width:' +
              value +
              '%"></div></div><b>' +
              value.toFixed(2) +
              "</b></div>",
          )
          .join("") +
        "</div>",
    )
    .join("");
  root.querySelector("#rs-all-rows").innerHTML = all
    .map(
      ([name, metric, result, id]) =>
        '<tr><td><a href="' +
        base +
        "results/submissions/" +
        id +
        '/result.json" target="_blank" rel="noopener noreferrer">' +
        esc(name) +
        "</a></td><td>" +
        esc(metric) +
        "</td><td>" +
        esc(result) +
        "</td></tr>",
    )
    .join("");
  function render() {
    root.querySelectorAll("[data-track]").forEach((b) => {
      b.setAttribute("aria-selected", String(b.dataset.track === state.track));
      b.tabIndex = b.dataset.track === state.track ? 0 : -1;
    });
    ["math", "image", "video", "all"].forEach((k) => {
      root.querySelector("#rs-panel-" + k).hidden = k !== state.track;
    });
    root.querySelector("#rs-math-selector").value = state.math;
    const m = math[state.math];
    root.querySelector("#rs-math-model").textContent = m.model;
    root.querySelector("#rs-math-metric").textContent = m.metric;
    root.querySelector("#rs-math-bars").innerHTML = m.rows
      .map((r) => bar(r, "percent"))
      .join("");
    root.querySelector("#rs-math-gain").textContent = m.gain;
    root.querySelector("#rs-math-gain-label").textContent = m.label;
    root.querySelector("#rs-math-explanation").textContent = m.explanation;
    root.querySelector("#rs-math-context").textContent = m.context;
    root.querySelector("#rs-math-protocol").textContent = m.protocol;
    root.querySelector("#rs-math-source").href = base + m.source;
    const method = methods[state.method];
    ["title", "description", "feedback", "budget", "available"].forEach((k) => {
      root.querySelector("#rs-method-" + k).textContent = method[k];
    });
    root.querySelector("#rs-method-source").href = base + method.path;
    root.querySelectorAll("[data-method]").forEach((b) => {
      b.setAttribute("aria-pressed", String(b.dataset.method === state.method));
    });
  }

  render();
  root.querySelectorAll("[data-track]").forEach((b) =>
    b.addEventListener("click", () => {
      state.track = b.dataset.track;
      render();
    }),
  );
  root.querySelector("#rs-math-selector").addEventListener("change", (e) => {
    state.math = e.target.value;
    render();
  });
  root.querySelectorAll("[data-method]").forEach((b) =>
    b.addEventListener("click", () => {
      state.method = b.dataset.method;
      render();
    }),
  );
  root.querySelectorAll("[data-jump]").forEach((b) =>
    b.addEventListener("click", () => {
      root
        .querySelector("#" + b.dataset.jump)
        .scrollIntoView({ block: "start", behavior: "auto" });
    }),
  );
  root.querySelectorAll("[data-show-track]").forEach((b) =>
    b.addEventListener("click", () => {
      state.track = b.dataset.showTrack;
      if (state.track === "math") state.math = "opsd";
      render();
      root
        .querySelector("#rs-results")
        .scrollIntoView({ block: "start", behavior: "auto" });
    }),
  );
  // Arrow-key navigation follows the WAI-ARIA tabs pattern.
  root.querySelectorAll("[data-track]").forEach((tab, index, tabs) => {
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft")
        next = (index + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) {
        event.preventDefault();
        tabs[next].click();
        tabs[next].focus();
      }
    });
  });
})();
