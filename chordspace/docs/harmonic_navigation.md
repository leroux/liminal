# ChordSpace: technical foundations for harmonic navigation

**Voice leading geometry, neo-Riemannian transformations, and the Tonnetz provide a mathematically rigorous yet computationally lightweight foundation for building a browser-based chord exploration tool.** The core insight from two decades of mathematical music theory is that chords live in well-defined geometric spaces where physical proximity corresponds to musical proximity — and the entire computational problem is small enough that brute-force approaches run in under a millisecond in a browser. This report synthesizes the key algorithms, formulas, data structures, and academic frameworks needed to implement ChordSpace, drawing primarily from Dmitri Tymoczko's voice leading geometry, Richard Cohn's neo-Riemannian theory, and Douthett and Steinbach's work on parsimonious graphs.

---

## 1. Measuring distance between chords with voice leading metrics

The fundamental question for ChordSpace is: given two chords, how "far apart" are they? Tymoczko's geometric framework (Science, 2006) models an n-note chord as a point in the orbifold **T^n/S_n** — the n-dimensional torus modulo the symmetric group. A voice leading between two chords is a path in this space, and its length under a chosen metric defines the distance.

### The displacement multiset and Lp norms

Given two ordered n-note chords A = (a₁, …, aₙ) and B = (b₁, …, bₙ), the **displacement multiset** is {|b₁ − a₁|, |b₂ − a₂|, …, |bₙ − aₙ|}. For pitch-class space, each displacement uses the circular distance: ‖x‖₁₂ = min(x mod 12, 12 − (x mod 12)). Three distance norms are musically meaningful:

| Metric | Formula | Musical meaning |
|--------|---------|-----------------|
| **L1 (taxicab)** | Σ\|dᵢ\| | Total semitones moved across all voices |
| **L2 (Euclidean)** | √(Σdᵢ²) | Penalizes large leaps in individual voices |
| **L∞ (Chebyshev)** | max(\|dᵢ\|) | Largest single-voice displacement |

**Tymoczko primarily recommends L1** for practical work because it is simplest to compute, correlates well with musical intuition ("total keyboard distance"), and is the standard in the literature. However, L1 has a limitation: it treats **{2, 0}** (one voice moves 2 semitones) as equal to **{1, 1}** (two voices each move 1 semitone), when listeners perceive the distributed motion as smoother. L2 strictly satisfies this "distribution constraint" — (1,1) has L2 distance ≈ 1.41 versus 2.0 for (2,0) — making it the better choice when penalizing large individual leaps matters. For ChordSpace, **L1 should be the default with L2 available as an option**.

### The crossing-free theorem and optimal assignment

For unordered chords, one must find the voice assignment (permutation) that minimizes total distance — formally a minimum-cost bipartite matching problem solvable by the Hungarian algorithm in O(n³). But Tymoczko proved a powerful shortcut: **there always exists a minimal voice leading that is crossing-free** (Theorem 1, Science 2006). This means sorting both chords by pitch and pairing them in order yields the optimal assignment in pitch space. In pitch-class space, only n cyclic rotations need checking:

```javascript
function minVoiceLeadingDistance(A, B) {
  // A, B: sorted pitch-class arrays of equal length n
  const n = A.length;
  let minDist = Infinity;
  for (let r = 0; r < n; r++) {
    let dist = 0;
    for (let i = 0; i < n; i++) {
      let d = Math.abs(A[i] - B[(i + r) % n]);
      d = Math.min(d, 12 - d);
      dist += d;
    }
    minDist = Math.min(minDist, dist);
  }
  return minDist;
}
```

This runs in **O(n²)** — for triads, just 9 arithmetic operations. Even without optimization, computing voice leading distance between any two chords is sub-microsecond work.

### Handling chords of different sizes

Measuring distance between a triad and a seventh chord (3 notes to 4 notes) is one of the hardest problems in voice leading theory. Callender, Quinn, and Tymoczko's OPTIC framework (Science, 2008) introduces **C-equivalence** (cardinality equivalence), which identifies chords related by note duplication — so {C, E, G} becomes C-equivalent to {C, C, E, G}. The practical algorithm: duplicate each note in the smaller chord to match the larger chord's cardinality, try all possible duplications, compute standard VLD for each, and take the minimum. For a triad-to-tetrachord comparison, this means testing C(3,1) = 3 duplications — trivially fast. An alternative is the **Hausdorff metric**: d_H(A,B) = max(max_{a∈A} min_{b∈B} |a−b|, max_{b∈B} min_{a∈A} |a−b|), which avoids the note-duplication step but is less musically intuitive.

### Weighted voice leading for perceptual accuracy

Standard mathematical frameworks treat all voices equally, but David Huron's perceptual research (Music Perception, 2001) demonstrates that **bass movement is perceptually foundational** and **outer voices (soprano and bass) are more prominent** than inner voices due to auditory masking effects. A weighted distance formula accommodates this:

```
d_weighted(A, B) = Σᵢ wᵢ · |bᵢ − aᵢ|
```

Reasonable defaults for SATB: w_bass = 1.5, w_inner = 1.0, w_soprano = 1.3. This breaks Tymoczko's orbifold geometry (which requires equal weights), so ChordSpace should offer both unweighted (mathematically rigorous) and weighted (perceptually tuned) modes as a user toggle.

---

## 2. Neo-Riemannian transformations as a chord navigation grammar

Neo-Riemannian theory provides the most elegant framework for suggesting "next chord" options. Its three core operations — P, L, and R — generate a complete network connecting all 24 major and minor triads through minimal voice motion, capturing harmonic relationships (especially chromatic mediants) that functional harmony cannot explain.

### Formal definitions of P, L, and R

All three operations are **involutions** (self-inverse: applying twice returns to the original). Each preserves exactly two common tones and moves one voice by semitone (P, L) or whole tone (R):

**P (Parallel)** preserves root and fifth, moves the third by 1 semitone:
- Major → minor: {r, r+4, r+7} → {r, r+3, r+7} (e.g., C major ↔ C minor)

**L (Leading-tone exchange)** preserves the minor third interval, moves one voice by 1 semitone:
- Major → minor: lower the root by semitone; new root = old third (e.g., C major → E minor: C drops to B)
- Minor → major: raise the fifth by semitone (e.g., E minor → C major: B rises to C)

**R (Relative)** preserves the major third interval, moves one voice by 2 semitones:
- Major → minor: raise the fifth by whole step (e.g., C major → A minor: G rises to A)
- Minor → major: lower the root by whole step (e.g., A minor → C major: A drops to G)

The algorithmic implementation is compact:

```javascript
function applyP(root, isMajor) {
  return { root, isMajor: !isMajor };
}
function applyL(root, isMajor) {
  return isMajor
    ? { root: (root + 4) % 12, isMajor: false }
    : { root: (root + 8) % 12, isMajor: true };
}
function applyR(root, isMajor) {
  return isMajor
    ? { root: (root + 9) % 12, isMajor: false }
    : { root: (root + 3) % 12, isMajor: true };
}
```

### The PLR group and its network structure

The PLR group is isomorphic to the **dihedral group D₁₂** of order 24, acting simply transitively on the 24 major/minor triads — meaning any triad can reach any other through a unique minimal PLR sequence. The resulting graph has each triad-vertex connected to exactly 3 neighbors (one per operation). The compound operations reveal deeper structure:

- **PL cycle** (length 6): generates **hexatonic systems** — 4 non-overlapping cycles of 6 triads each, all using pitch classes from a hexatonic scale. Cohn's "maximally smooth cycles" (Music Analysis, 1996).
- **PR cycle** (length 8): generates **octatonic systems** — 3 cycles of 8 triads, using octatonic scale pitch classes.
- **LR cycle** (length 24): a single Hamiltonian cycle traversing all 24 triads.

### Compound operations capture chromatic mediants

The real power of neo-Riemannian theory for ChordSpace lies in **compound operations** that model the chromatic mediant relationships ubiquitous in film music, late-Romantic harmony, and modern pop:

| Compound | Steps | Example from C major | Common tones | Total VLD |
|----------|-------|---------------------|--------------|-----------|
| LP | 2 | E major | 1 (E) | 2 semitones |
| PL | 2 | A♭ major | 1 (G/A♭) | 2 semitones |
| RP | 2 | A major | 1 (E) | 3 semitones |
| PR | 2 | E♭ major | 1 (G/E♭) | 3 semitones |
| N (=RLP) | 3 | F minor | 1 (C) | 3 semitones |
| S (=LPR) | 3 | C♯ minor | 1 (E) | 3 semitones |
| H (=LPL) | 3 | A♭ minor | 0 | 3 semitones |

The **hexatonic pole** (H = LPL) is the maximally distant chord within a hexatonic cycle — sharing zero common tones but requiring only 3 semitones of total motion. It produces the "uncanny" or eerie effect used extensively by Wagner, Liszt, and in contemporary film scoring.

### Extensions for seventh chords

Adrian Childs (Journal of Music Theory, 1998) developed transformations for **dominant seventh and half-diminished seventh chords** (set class [0258]), which are inversionally related just as major and minor triads are. Two pitch classes are held constant while two move by semitone, creating a network analogous to the triadic Tonnetz. Douthett and Steinbach's **P_{m,n} framework** (1998) generalizes further: two chords are P_{m,n}-related if m voices move by semitone and n by whole tone. For seventh chords, the key relations are P_{1,0} (3 common tones, 1 semitone motion), P_{2,0} (2 common tones, 2 semitone motions), and P_{3,0} (1 common tone, 3 semitone motions — the relation underlying the Tristan Prelude). Boris Kerkez (Bridges, 2012) extended the PLR group itself to major/minor seventh chords via a PS-group also isomorphic to D₁₂.

For ChordSpace, the practical recommendation is to implement PLR for triads directly (simple root/type computations), then extend to seventh chords using the P_{m,n} enumeration algorithm, which generates all neighbors where m voices move by semitone and n by whole tone.

---

## 3. The Tonnetz and orbifolds as navigable geometric models

### Tonnetz structure and coordinate system

The Tonnetz (Euler, 1739; revived by Cohn, 1997) is a **triangular lattice** where vertices represent pitch classes and three edge-directions encode consonant intervals: **perfect fifths** (horizontal), **major thirds** (one diagonal), and **minor thirds** (the other diagonal). Triads appear as triangles — downward-pointing for major, upward-pointing for minor. **Adjacent triangles sharing an edge correspond to PLR-related triads** (2 common tones), making the Tonnetz a spatial encoding of the neo-Riemannian network.

In 12-tone equal temperament, the infinite lattice wraps because fifths cycle after 12 steps, major thirds after 3, and minor thirds after 4. The resulting topology is a **torus (T²)** — the fundamental domain is a parallelogram containing all 12 pitch classes, with opposite edges identified. For rendering, a practical coordinate scheme uses the fifth (interval 7) and major third (interval 4) as basis vectors:

```javascript
function tonnetzCoords(col, row) {
  const hexWidth = 60, hexHeight = 52;
  return {
    x: col * hexWidth + (row % 2) * hexWidth / 2,
    y: row * hexHeight * 0.75,
    pitchClass: (col * 7 + row * 4) % 12
  };
}
```

Displaying ~3×3 copies of the fundamental domain gives users the visual sense of the toroidal wraparound while keeping the interface navigable.

### The chicken-wire torus and cube dance

Douthett and Steinbach (1998) introduced the **geometric dual** of the Tonnetz — the **chicken-wire torus** — where nodes represent triads (not pitch classes) and edges represent PLR operations. This hexagonal lattice, also toroidal, is the natural "chord map" for navigation: each triad-node connects to exactly 3 PLR neighbors. Their **Cube Dance** graph augments this by including **augmented triads as hub nodes**, each connecting to 6 triads via P_{1,0} relations. For seventh chords, the **Power Towers** graph connects dominant, half-diminished, minor, and diminished seventh chords via P_{1,0} and P_{2,0} relations.

### Orbifolds for voice leading geometry

Tymoczko's orbifold model provides a complementary geometric framework. An n-note chord is a point in **T^n/S_n**: the n-torus (each axis is circular pitch-class space) quotiented by the symmetric group (because voice order doesn't matter). The resulting spaces have rich topology — **2-note chords live on a Möbius strip**, **3-note chords on a twisted triangular prism** — with mirror boundaries at the singularities (chords with doubled notes). Voice leadings are line segments in these spaces, and their lengths under the chosen metric define distance.

A key property: **chords that evenly divide the octave** (augmented triads, diminished seventh chords) sit at the center of these orbifolds, enabling efficient voice leading to many transpositions. Major and minor triads, as near-equal divisions, cluster near the center — which is why triadic harmony supports such smooth voice leading.

### Visualizable projections for the browser

For ChordSpace, three visualization strategies are practical:

**Primary: 2D Tonnetz hexagonal grid.** This is the most intuitive, well-understood model with numerous existing implementations. Render with HTML5 Canvas, SVG, or D3.js. Triads are highlighted triangles; the active chord and its neighbors are color-coded by distance or transformation type.

**Secondary: Tymoczko's annular projection.** A 2D polar plot where angular position encodes pitch-class sum (transposition level) and radial position encodes evenness (how closely the chord approximates an equal division of the octave). This works for any chord cardinality and elegantly shows how augmented triads and diminished seventh chords sit at the center.

**Optional: 3D toroidal Tonnetz or MDS embedding.** Using Three.js/WebGL, render the torus directly with pitch classes mapped to its surface, or pre-compute a multidimensional scaling (MDS) embedding of all chords based on their pairwise voice leading distances and display the resulting 3D point cloud. Krumhansl and Kessler (1982) showed MDS recovers the toroidal structure from perceptual data; Aminian et al. (2020) confirmed Tonnetz-based MDS produces interpretable 2D/3D layouts.

The DFT-based **Fourier phase space** approach (Yust, 2018; Amiot) offers a mathematically principled alternative: map each pitch-class set to its discrete Fourier transform coefficients and use the phases of coefficients 3 and 5 as (x, y) coordinates — this directly recovers the Tonnetz layout with no heuristic choices.

---

## 4. Algorithms for efficient neighborhood search

### Why brute force wins for this problem

The fundamental insight for ChordSpace's implementation is that **the chord vocabulary in 12-TET is tiny by computational standards**. There are only C(12,3) = 220 trichords, C(12,4) = 495 tetrachords, and a practical jazz/pop vocabulary of ~500 named chord types. Computing VLD between two triads takes 9 arithmetic operations. A full brute-force scan of 500 chords takes ~350K operations — **under 1 millisecond in any modern browser**. Spatial indexing structures like k-d trees or ball trees are overkill.

The recommended strategy is **pre-computation**: build the complete M×M distance matrix at application initialization. For 500 chords, this is 250K entries (~1MB), computable in under 10ms. After that, finding the N closest neighbors to any chord is a single row scan plus sort — O(M + M log M), well under 0.1ms.

```typescript
class ChordSpace {
  chords: Chord[];
  distanceMatrix: Float32Array;
  
  constructor(vocabulary: Chord[]) {
    this.chords = vocabulary;
    const M = vocabulary.length;
    this.distanceMatrix = new Float32Array(M * M);
    for (let i = 0; i < M; i++)
      for (let j = i + 1; j < M; j++) {
        const d = minVoiceLeadingDistance(
          vocabulary[i].pitchClasses,
          vocabulary[j].pitchClasses
        );
        this.distanceMatrix[i * M + j] = d;
        this.distanceMatrix[j * M + i] = d;
      }
  }
  
  findNearest(sourceIdx: number, n: number): number[] {
    const M = this.chords.length;
    const row = this.distanceMatrix.slice(
      sourceIdx * M, sourceIdx * M + M
    );
    return Array.from(row.keys())
      .filter(i => i !== sourceIdx)
      .sort((a, b) => row[a] - row[b])
      .slice(0, n);
  }
}
```

### Enumerating parsimonious neighbors algorithmically

For P_{m,n} neighbors (Douthett and Steinbach's framework), generate candidates by choosing which voices move and in which direction:

```javascript
function parsimoniousNeighbors(chord, m, n) {
  const results = new Set();
  const len = chord.length;
  // Choose m+n voices to move
  for (const moving of combinations(range(len), m + n)) {
    // Partition into m semitone-movers and n wholetone-movers
    for (const semitoneVoices of combinations(moving, m)) {
      const wholeToneVoices = moving.filter(v => !semitoneVoices.includes(v));
      // Try all direction combinations (2^(m+n) possibilities)
      for (let dirs = 0; dirs < (1 << (m + n)); dirs++) {
        const newChord = [...chord];
        let bit = 0;
        for (const v of semitoneVoices)
          newChord[v] = (chord[v] + ((dirs >> bit++) & 1 ? 1 : 11)) % 12;
        for (const v of wholeToneVoices)
          newChord[v] = (chord[v] + ((dirs >> bit++) & 1 ? 2 : 10)) % 12;
        const sorted = [...new Set(newChord)].sort((a, b) => a - b);
        if (sorted.length === len) results.add(sorted.join(','));
      }
    }
  }
  return [...results].map(s => s.split(',').map(Number));
}
```

For P_{1,0} on a triad: 3 voices × 2 directions = 6 candidates, yielding typically **4–6 valid neighbors** (including the PLR-related triads). This computation is essentially instantaneous.

### Common-tone computation with bitwise operations

Representing chords as **12-bit chroma vectors** (bit i = 1 if pitch class i is present) enables blazing-fast common-tone queries:

```javascript
function chromaVector(pitchClasses) {
  let v = 0;
  for (const pc of pitchClasses) v |= (1 << pc);
  return v;
}

function commonToneCount(a, b) {
  return popcount(a & b); // bitwise AND + count set bits
}
```

This runs in **O(1) per comparison** — millions per second. To find all chords sharing exactly k pitch classes with a source chord, scan the vocabulary with a single bitwise AND and popcount per chord.

### Parameterizing smooth versus dramatic voice leading

A "smoothness slider" (0.0 = parsimonious, 1.0 = dramatic) maps to a **voice-leading distance radius**. Research from Music Theory Online (Smither, 2019) provides calibrated thresholds: voice leading is considered "smooth" when no individual voice exceeds 2 semitones and total displacement stays under 4 semitones.

Recommended threshold mapping for triads:

| Slider position | Category | Max individual voice | Max total VLD |
|----------------|----------|---------------------|---------------|
| 0–25% | Parsimonious | 1 semitone | 1–2 |
| 25–50% | Smooth | 2 semitones | 2–4 |
| 50–75% | Moderate | 3 semitones | 4–7 |
| 75–100% | Dramatic | 6 semitones | 7–12 |

The slider simply adjusts the filter threshold applied to the pre-computed distance matrix, making the UI update trivially fast.

### A combined ranking formula for musical relevance

Pure voice-leading distance misses musical context. A chord one semitone away might be a dissonant cluster rather than a musically useful progression. The recommended combined scoring formula balances proximity with musicality:

```javascript
function combinedScore(source, target, weights = {
  vld: 1.0, commonTones: 0.5, consonance: 0.3, neoRiemannian: 0.8
}) {
  const vld = voiceLeadingDistance(source, target);
  const common = commonToneCount(source.chroma, target.chroma);
  const consonance = chordConsonanceScore(target);
  const nrBonus = isNeoRiemannianRelated(source, target) ? 1.0 : 0.0;
  
  return weights.vld * vld
       - weights.commonTones * common
       - weights.consonance * consonance
       - weights.neoRiemannian * nrBonus;
}
```

Consonance can be scored using empirical interval dissonance values (perfect fifth = very consonant at 0.1, tritone = highly dissonant at 0.9, based on roughness models from Cubarsí, 2019). The neo-Riemannian bonus flags chords reachable by named PLR operations, which are almost always musically interesting progressions.

---

## 5. Recommended data structures and architecture

### Core TypeScript types

```typescript
type PitchClass = 0|1|2|3|4|5|6|7|8|9|10|11;

interface Chord {
  pitchClasses: PitchClass[];      // sorted, e.g., [0, 4, 7]
  chroma: number;                   // 12-bit vector for fast bitwise ops
  root: PitchClass;
  quality: string;                  // "major" | "minor" | "dom7" | ...
  name: string;                     // "Cmaj" | "Am7" | ...
}

interface VoiceLeadingResult {
  chord: Chord;
  distance: number;                 // VLD (L1 by default)
  commonTones: number;
  displacement: number[];           // per-voice motion
  neoRiemannianOp?: string;         // "P" | "L" | "R" | "LP" | ...
  combinedScore: number;
}
```

### Performance budget

| Operation | Complexity | Wall time (500 chords) |
|-----------|-----------|----------------------|
| VLD between two triads | O(n²), n=3 | < 1μs |
| Pre-compute full distance matrix | O(M²·n²) | ~5ms |
| Query N nearest neighbors | O(M log M) | < 0.1ms |
| Enumerate P_{1,0} neighbors | O(2n) | < 0.01ms |
| Common-tone filter (bitwise) | O(M) | < 0.05ms |
| Full combined ranking | O(M log M) | < 0.5ms |

**Every operation comfortably fits within a single animation frame** (16ms budget at 60fps), meaning the entire neighborhood search and ranking pipeline can run synchronously on user interaction without any perceivable delay.

### Existing open-source foundations

Several libraries and tools provide building blocks. **tonal.js** (npm: `tonal`) is the most comprehensive TypeScript music theory library, offering chord parsing, interval computation, and a `@tonaljs/voice-leading` module. For Tonnetz visualization, **cifkao/tonnetz-viz** and **codedot/tonnetz** are mature web implementations using Web MIDI and Canvas/SVG. The **guinnesschen/harmonic-keyboard** project implements Wasserstein (Earth Mover's) distance for optimal voice leading in a React app. In Python, **cuthbertLab/music21** provides a `voiceLeading` module for motion analysis, and **marcobn/musicntwrk** builds musical space networks. Tymoczko's own **ChordGeometries** software (Java/Max) renders chords and voice leadings in 3D geometric spaces.

---

## Conclusion

The mathematical music theory literature provides ChordSpace with a complete, implementable toolkit. Three findings stand out as particularly important for the technical specification. First, **the computational problem is small**: with fewer than 1000 chords in any practical vocabulary, pre-computing a full distance matrix at load time eliminates the need for spatial indexing entirely — a simple Float32Array and array sort handles everything in real time. Second, **neo-Riemannian transformations provide a natural "grammar" for chord suggestion** that is both mathematically principled (the PLR group acts simply transitively on triads) and musically powerful (capturing the chromatic mediant relationships that define film and late-Romantic harmony). These can be computed with pure arithmetic — no lookup tables required. Third, **the 2D Tonnetz is the ideal primary visualization**: it spatially encodes exactly the relationships users care about (common tones, PLR operations, parsimonious voice leading), has a well-understood hexagonal rendering, and multiple open-source implementations exist as reference. The annular projection and 3D toroidal view can serve as secondary perspectives for users who want to explore transpositional and orbifold structure. The combined ranking formula — weighting voice leading distance, common tones, consonance, and neo-Riemannian relationships — bridges the gap between "technically close" and "musically interesting," giving users a navigable, meaningful harmonic space.