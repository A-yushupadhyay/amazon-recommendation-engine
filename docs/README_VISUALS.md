# Visual assets: how they are made and maintained

All visuals are hand-editable SVG text files in `docs/assets/`. They use no scripts, no external fonts and no external images, so GitHub renders them from the repository itself.

| File | Shows | Numbers come from |
|---|---|---|
| `metrics-hero.svg` | Four headline results strip | README "Results" tables |
| `architecture.svg` | Offline pipeline, online serving, fallback paths (animated dots) | Architecture section |
| `model-quality.svg` | Precision@10 and recall@10, CF vs baseline | `ml/similarity_engine.py` output |
| `tail-latency.svg` | p50, p99 before and after fix (log scale) | Locust runs |
| `ab-test.svg` | CTR bars and required sample size | `ml/ab_test_simulation.py` output |

`docs/architecture.mmd` is the Mermaid source of the text version of the architecture diagram in the README.

## Updating a number

1. Open the SVG in any text editor and search for the old value.
2. Change the label and, for bars, the `height` or `width` attribute. Bar sizes are proportional to the value (for `tail-latency.svg`, to the base-10 logarithm of the value).
3. Update the matching README table so the text and the picture agree.
4. Open the file in a browser to preview it.

## Animation

Animations are small CSS and SMIL effects: bars grow once on load, tiles fade in, dashed connectors move, and dots travel along the three serving paths. They carry no information that the static picture lacks. Every file includes a `prefers-reduced-motion` rule that turns the CSS animations off. Each SVG also has a `<title>` and `<desc>` for screen readers, and the README gives full alt text and a table next to every chart.

## Rendering notes

- GitHub shows SVGs added with `<img>` or Markdown image syntax, including their animation.
- If an SVG ever fails to load, the alt text and the tables beside it still carry the content.
- Check the page in a private browser window after each commit, since GitHub caches images aggressively.
