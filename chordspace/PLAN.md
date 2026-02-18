# ChordSpace — Phased Implementation Plan

## What Is ChordSpace?

ChordSpace is a browser-based instrument for exploring harmony visually and sonically. You see every chord in 12-tone equal temperament laid out on a 2D map — a Tonnetz hexagonal grid where spatial proximity means musical proximity. Click a chord to hear it. The map is painted as a consonance/dissonance heatmap that shifts in real time when you change the playback timbre, because the same chord sounds consonant on a Rhodes and rough on a square wave. Navigate by voice-leading distance, neo-Riemannian transformations (P, L, R), or just drag around and listen.

The core idea: **harmony has a shape, and that shape changes with timbre.** ChordSpace makes both visible and playable.

---

## Phase 1: Core Data Layer ✅ DONE (`src/core.ts`)

### 1.1 Pitch-Class Set Engine ✅
- [x] 12-bit integer representation, all bitwise ops
- [x] `transposePCS`, `invertPCS`, `popcount`, `pcsToBits`, `bitsToArray`
- [x] `intervalVector` — bitwise method
- [x] `commonTones` — popcount of AND

### 1.2 Classification & Enumeration ✅
- [x] `normalForm`, `primeForm`, `primeFormBits` — Rahn convention
- [x] Full Forte catalog: all 224 TnI set classes for cardinality 2–7
- [x] `FORTE_CLASSES`, `PCS_TO_CLASS` lookup

### 1.3 Chord Naming ✅
- [x] 26 chord templates × 12 roots → `CHORD_VOCABULARY` (~300 entries)
- [x] `nameChord(bits)`, `allChordNames(bits)` — ranked lookup

### 1.4 Data Structures ✅
- [x] `SetClass`, `ChordEntry`, `ChordName`, `Triad` interfaces

---

## Phase 2: Voice Leading & Navigation ✅ DONE (`src/navigation.ts`)

### 2.1 Voice Leading Distance ✅
- [x] `minVLD(a, b)` — L1 norm with cyclic rotation
- [x] `vldBits(a, b)` — convenience wrapper

### 2.2 Pre-computed Distance Matrix ✅
- [x] `buildDistanceMatrix()` — M×M Float32Array for full chord vocabulary
- [x] `findNearestFromMatrix()` — fast row scan

### 2.3 Neo-Riemannian Transformations ✅
- [x] `applyP`, `applyL`, `applyR`
- [x] Compound operations: LP, PL, RP, PR
- [x] `plrNeighbors(root, isMajor)`

### 2.4 Parsimonious Neighbor Enumeration ✅
- [x] `parsimoniousNeighbors(bits, maxMoves)` — generate all chords reachable by moving voices ±1 semitone

### 2.5 Combined Ranking ✅
- [x] Weighted score: VLD + common tones + consonance + PLR bonus

### 2.6 Classical MDS ✅
- [x] `classicalMDS(matrix)` — power iteration for top 2 eigenvectors of double-centered D²

---

## Phase 3: Dissonance & Consonance Engine ✅ DONE (`src/dissonance.ts`)

### 3.1 Vassilakis Roughness Model ✅
- [x] `vassilakisPair` with exact published constants
- [x] AF-degree exponent 3.11

### 3.2 Overtone Profile System ✅
- [x] 12 timbres: sine, piano, rhodes, organ, saw, square, bell, wurlitzer, plucked, clavinet, subbass, brass

### 3.3 Consonance Score Normalization ✅
- [x] `consonanceScore` — pairs-normalized, 0–1 range
- [x] `precomputeTriadDissonance` for Tonnetz heatmap

---

## Phase 4: Synthesis Engine ✅ DONE (`src/synth.ts`)

### 4.1 Multi-Method Synthesizer ✅
- [x] Additive synthesis (sine oscillators matching timbre partial profile)
- [x] Wavetable synthesis (PeriodicWave from timbre partials)
- [x] Subtractive synthesis (dual detuned sawtooth → biquad LPF with filter envelope)
- [x] FM synthesis (2-operator carrier + modulator with index decay)

### 4.2 ADSR Envelope ✅
- [x] Full ADSR with macro-driven attack time

### 4.3 Effects Chain ✅
- [x] Chorus (modulated delay, 0.5Hz LFO, 3ms depth)
- [x] Reverb (1.5s convolution with generated exponential decay impulse)

### 4.4 Macro Controls ✅
- [x] Brightness (partial rolloff / filter cutoff)
- [x] Warmth (even harmonic boost / detune)
- [x] Attack (ADSR attack time)
- [x] Body (mid-partial boost / resonance)
- [x] Air (reverb send)

---

## Phase 5: 2D Visualization ✅ DONE

### 5.1 Tonnetz ✅ (`src/tonnetz.ts`)
- [x] Hexagonal grid renderer, pc = (q*7 + r*4) % 12
- [x] Major/minor triangle rendering with consonance heatmap
- [x] Click-to-audition, PLR neighbor highlighting

### 5.2 2D Chord Scatter Map ✅ (`src/chordmap.ts`)
- [x] X=consonance, Y=cardinality bands with evenness spread
- [x] 3 view modes: named chords, Forte classes, all PCS
- [x] Zoom/pan, hover tooltips, click-to-audition
- [x] Color-coded by cardinality
- [x] Key filter visual dimming
- [x] Search highlighting (yellow glow)
- [x] Progression trail rendering

### 5.3 Annular Projection ✅ (`src/annular.ts`)
- [x] Tymoczko polar plot: angle = PC sum mod 12, radius = 1 - evenness
- [x] Concentric rings, clock labels, cardinality color coding
- [x] Click-to-audition, hover tooltips

### 5.4 View Switching ✅
- [x] Map / Tonnetz / Annular toggle with canvas show/hide

---

## Phase 6: Audio Visualization ✅ DONE

- [x] Web Audio API multi-method synthesis at 48kHz
- [x] Click-to-audition with synth method + macro selection
- [x] Waveform oscilloscope (time-domain display)
- [x] Spectrum analyzer (frequency-domain bar graph)
- [x] AnalyserNode integration for real-time visualization

---

## Phase 7: Voicing Engine ✅ DONE (`src/core.ts`)

### 7.1 Voicing Algorithms ✅
- [x] Close position with inversion support
- [x] Drop-2
- [x] Shell (root + 3rd + 7th)
- [x] Spread (bass isolated, upper voices high)
- [x] Rootless Type A (3-5-7-9)
- [x] Rootless Type B (7-9-3-5)
- [x] Quartal (stacked 4ths)

### 7.2 Register Constraints ✅
- [x] `applyRegisterConstraints()` — minimum spacing by register (<C3: 7st, C3-C5: 3st, >C5: 1st)
- [x] `applyDoublingRules()` — avoid doubled 3rd in upper register
- [x] Toggle-able via UI checkbox

---

## Phase 8: Progression & Search ✅ DONE

### 8.1 Progression Sequencer ✅
- [x] Click-to-build progression with VLD display between chords
- [x] Play/stop with tempo control (40-240 BPM)
- [x] Loop mode
- [x] Undo (remove last) and clear
- [x] Visual trail on chord map

### 8.2 MIDI Export ✅
- [x] Export progression as MIDI file (format 0, tempo-aware)
- [x] Download as .mid file

### 8.3 Search ✅
- [x] Name search (Cmaj7, dim, etc.)
- [x] Forte number search (4-20, etc.)
- [x] Interval pattern search (0 4 7)
- [x] Note name search (C E G)

### 8.4 Key Filter ✅
- [x] 12 roots × 12 scales (major, minor, dorian, mixolydian, lydian, phrygian, harmonic minor, melodic minor, pentatonic, blues, whole tone, chromatic)
- [x] Visual dimming of out-of-key chords

---

## Phase 9: Navigation-First Revamp ✅ DONE

### 9.1 Constellation View ✅ (`src/constellation.ts`)
- [x] Center chord with ranked neighbors in animated radial layout
- [x] Uses `combinedRanking()` — weighted VLD + common tones + consonance + PLR bonus
- [x] Click neighbor to navigate, right-click to add to progression
- [x] Connection lines: thickness/opacity encode VLD, gold dots mark PLR relationships
- [x] `randomChord()` for exploration starting point
- [x] Key filter integration (rebuilds candidate pool)

### 9.2 Simplified UI ✅
- [x] Constellation as default view (starts on C major)
- [x] 4-view toggle: Explore / Map / Tonnetz / Annular
- [x] Random button (gold accent, navigates constellation)
- [x] "more" toggle hides secondary controls (view mode, root filter, inversion, cardinality, register)
- [x] Always-visible progression bar at bottom with pills, arrows, VLD display
- [x] Per-chord remove buttons in progression
- [x] Click progression pills to audition

### 9.3 Bug Fixes ✅
- [x] MIDI export uses actual BPM from tempo control (was hard-coded 120)
- [x] Unified `updateInfoPanel(bits, midi)` works across all views
- [x] Deleted dead `audio.ts` (superseded by `synth.ts`)

---

## Architecture

```
src/
  core.ts         — PCS engine, Forte catalog, chord naming, voicing, register constraints
  dissonance.ts   — Vassilakis roughness, 12 timbres, consonance scoring
  navigation.ts   — VLD, PLR, distance matrix, MDS, parsimonious neighbors, combined ranking
  synth.ts        — Multi-method synthesizer (additive/wavetable/subtractive/FM) + effects
  chordmap.ts     — 2D scatter map renderer with zoom/pan
  tonnetz.ts      — Tonnetz hexagonal grid renderer
  annular.ts      — Annular (Tymoczko polar) projection renderer
  constellation.ts — Navigation-first constellation view (center chord + ranked neighbors)
  app.ts          — App shell, state, control wiring, oscilloscope, MIDI export
index.html        — UI layout, toolbar, sidebar, canvases
```
