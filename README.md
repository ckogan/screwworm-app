# Screwworm zone-freedom app (demo)

Interactive what-if app for zone freedom from New World screwworm. It runs entirely in the browser (Shinylive); there is no server.

**Live:** https://ckogan.github.io/screwworm-app/

- Demo mode: the reviewer notes are hidden.
- Priors are placeholders from a first-pass literature review, and the zone presets use a guessed initial state. Results are for exploring the model, not for decisions.
- Source: `src/` holds the app, the scenario presets and the closed-form model. Demo mode is on by default.

Rebuild: `pip install shinylive && shinylive export src docs && touch docs/.nojekyll`
