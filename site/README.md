# DataLite-RSI homepage

The production homepage lives in this repository and is hosted at
<https://haolpku.github.io/DataLite-RSI/> using GitHub Pages.

## Edit and preview

```bash
python scripts/build_site.py
python -m http.server 8000 --directory _site
```

Open <http://localhost:8000>. The output is ignored by Git; source files are:

- `index.template.html`: semantic page markup and build-time placeholders.
- `assets/site.css`: responsive light/dark styles.
- `assets/site.js`: local result tabs, math selector and method details.
- `content.json`: display order, protocol labels and editorial explanations.
- `../scripts/build_site.py`: reads source result manifests and generates the
  README table, chart payload, accessible static table and standalone HTML.

All chart scores and headline comparisons are derived from result manifests.
Narrative descriptions and research limitations are editorial content: review
these whenever a protocol, budget or source result changes. Counts come from
registered manifests. Numerical data is also available as `results.json` in the
built site. Sources link to the full Git commit used for each build.

Add a new result to both its registry location and `content.json`; the build
rejects missing or duplicate listings. Run the build and commit the refreshed
README. CI uses `--check` to catch stale generated summary data.

## Deploy

`.github/workflows/pages.yml` validates, tests and builds on main pushes, then
publishes only `_site/`. Pull requests run the same site build and repository
checks without deploying. GitHub Pages must use **GitHub Actions** as its source.
Repository content, environment files and source artifacts are not uploaded to
Pages. No separate website repository or access token is required.

The design follows the academic-project layout requested by the maintainer,
inspired by [K12-KGraph](https://haolpku.github.io/K12-KGraph-page/). It uses no
content, illustrations, authorship, or logos from that project. Inter is loaded
from Google Fonts with system-font fallback; all functional assets are local.

The static result table is readable without JavaScript. Interactive charts and
method selection require JavaScript. Tabs support arrow/Home/End navigation;
layouts respond to small screens and the system color scheme.
