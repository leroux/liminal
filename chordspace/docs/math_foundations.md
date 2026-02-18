# Mathematical and music-theoretic foundations for ChordSpace

**ChordSpace can algorithmically generate every possible chord in 12-tone equal temperament by combining three layers: pitch-class set enumeration (224 unique set classes from 4,096 subsets of Z₁₂), a mapping layer that assigns chord names to roughly 30 of those classes, and a voicing engine that explodes each abstract chord into concrete MIDI realizations — 150 voicings for a simple triad, billions for a 13th chord.** This report provides the complete mathematical framework, enumeration counts, algorithms, and data structures needed to build this system. The three layers correspond to increasingly concrete representations: abstract equivalence classes, named harmonic objects with roots, and physical sound events placed in specific registers.

---

## 1. Pitch-class sets as the atomic units of harmony

### The Z₁₂ universe

In 12-tone equal temperament, every pitch reduces to one of 12 pitch classes: C=0, C♯=1, D=2, D♯=3, E=4, F=5, F♯=6, G=7, G♯=8, A=9, A♯=10, B=11. A pitch-class set (PCS) is any subset of Z₁₂ = {0,1,2,...,11}. The total number of possible subsets is **2¹² = 4,096**, ranging from the empty set to the full chromatic aggregate.

Allen Forte's 1973 *The Structure of Atonal Music* introduced a classification system that groups these 4,096 subsets into equivalence classes under transposition and inversion. Two sets belong to the same **set class** if one can be transformed into the other by Tₙ (adding n mod 12 to every element) or TₙI (inverting then transposing). Each class receives a **Forte number** in the format *n-m*, where *n* is cardinality and *m* is an ordinal. An optional "Z" prefix indicates Z-related pairs — sets sharing identical interval vectors but not related by transposition or inversion.

### Enumeration: from raw subsets to equivalence classes

The binomial coefficient C(12,n) gives raw subset counts. Applying group actions collapses these dramatically:

| Cardinality | Raw subsets C(12,n) | Tₙ classes (necklaces) | TₙI classes (Forte) |
|:-----------:|:-------------------:|:----------------------:|:--------------------:|
| 0 | 1 | 1 | 1 |
| 1 | 12 | 1 | 1 |
| 2 | 66 | 6 | 6 |
| 3 | 220 | 19 | 12 |
| 4 | 495 | 43 | 29 |
| 5 | 792 | 66 | 38 |
| 6 | 924 | 80 | 50 |
| 7 | 792 | 66 | 38 |
| 8 | 495 | 43 | 29 |
| 9 | 220 | 19 | 12 |
| 10 | 66 | 6 | 6 |
| 11 | 12 | 1 | 1 |
| 12 | 1 | 1 | 1 |
| **Total** | **4,096** | **352** | **224** |

The symmetry between cardinality *n* and *12−n* arises from complementation: every set class at cardinality *n* has a complement at cardinality *12−n*. Forte's published catalog covers cardinalities 3–9, yielding **208 set classes**. The full count including trivial cardinalities is **224 TₙI classes** (bracelets) or **352 Tₙ classes** (necklaces). The distinction matters: under Tₙ alone, major and minor triads are different classes; under TₙI, they collapse into a single class (3-11) because inversion maps one to the other.

### Burnside's lemma proves the counts

Burnside's lemma states that the number of distinct orbits under a group *G* acting on a set *X* equals **(1/|G|) × Σ|Fix(g)|** summed over all group elements. For transposition-only equivalence, the group is Z₁₂ (order 12). For transposition-plus-inversion, it is the dihedral group D₁₂ (order 24).

**Transposition orbits.** The permutation Tₖ decomposes Z₁₂ into cycles of length 12/gcd(k,12). A subset of size *n* is fixed by Tₖ only if *n* is divisible by the cycle length, and the count equals C(c, n/ℓ) where *c* is the number of cycles and ℓ is cycle length.

Worked example for **trichords** (n=3): T₀ fixes all C(12,3)=220 subsets. T₁,T₅,T₇,T₁₁ each have one cycle of length 12 — since 3 is not divisible by 12, they fix 0 subsets. T₂,T₁₀ produce 2 cycles of length 6 — 3 is not divisible by 6, so 0 fixed. T₃,T₉ produce 3 cycles of length 4 — 0 fixed. T₄,T₈ produce 4 cycles of length 3 — C(4,1)=4 fixed each, giving 8 total. T₆ produces 6 cycles of length 2 — 0 fixed. Sum = 220+8 = **228**, divided by 12 = **19 Tₙ trichord classes**.

Worked example for **tetrachords** (n=4): T₀ fixes 495. T₃,T₉ each fix C(3,1)=3 (total 6). T₆ fixes C(6,2)=15. All others fix 0. Sum = 495+6+15 = **516**, divided by 12 = **43 Tₙ tetrachord classes**.

**Adding inversion** requires counting fixed points of each TₖI operator. Even inversions (T₀I, T₂I, ..., T₁₀I) each fix 2 singleton pitch classes and 5 swapped pairs. Odd inversions fix 0 singletons and 6 swapped pairs. For tetrachords: even inversions each fix C(2,0)·C(5,2) + C(2,2)·C(5,1) = 10+5 = 15, times 6 = 90. Odd inversions each fix C(6,2)=15, times 6 = 90. Total inversion fixed points = 180. Grand total = (516+180)/24 = **696/24 = 29 TₙI tetrachord classes**.

For **hexachords**: transposition sum = 960 (yielding 80 necklaces), inversion sum = 240, grand total = 1200/24 = **50 Forte hexachord classes**.

---

## 2. Algorithms for normal form, prime form, and interval vectors

### Computing normal form

Normal form is the most compact circular arrangement of a PCS, transposed to start at 0. The algorithm:

```
ALGORITHM NormalForm(S):
  1. Sort S ascending: s₀ < s₁ < ... < sₖ₋₁
  2. Generate k circular rotations:
     R₀ = [s₀, s₁, ..., sₖ₋₁]
     Rⱼ = [sⱼ, sⱼ₊₁, ..., sⱼ₋₁+12]  (add 12 to wrapped elements)
  3. For each Rⱼ, compute span = last − first
  4. Select rotation(s) with minimum span
  5. Break ties: compare (2nd − 1st), then (3rd − 1st), etc.
     Choose the rotation with smaller values (most packed left)
  6. Transpose result so first element = 0
```

Example: {2, 6, 9} → rotations [2,6,9] span 7, [6,9,14] span 8, [9,14,18] span 9. Winner: [2,6,9], transposed to **[0,4,7]** — the prime form of the major triad.

### Computing prime form

Two competing conventions exist. The **Rahn convention** (now standard, used by Straus's textbook) computes normal form of both the set and its inversion, then selects the lexicographically smaller result:

```
ALGORITHM PrimeForm(S):
  A = NormalForm(S), transposed to 0
  S' = {(12 − s) mod 12 | s ∈ S}
  B = NormalForm(S'), transposed to 0
  return lexicographically smaller of A and B
```

The **Forte convention** differs in tie-breaking: it selects the form most packed to the left (largest gaps pushed right). Only **17 of 352 Tₙ classes** produce different prime forms under the two systems — for example, 5-20 yields [0,1,5,6,8] under Rahn but [0,1,3,7,8] under Forte.

### Interval vectors reveal harmonic character

The interval vector (IV) is a 6-element array **⟨ic₁, ic₂, ic₃, ic₄, ic₅, ic₆⟩** counting occurrences of each interval class. Interval class = min(d, 12−d) where d is the directed interval between two pitch classes.

```
ALGORITHM IntervalVector(S):
  iv = [0,0,0,0,0,0]
  for each pair (pᵢ, pⱼ) where i < j:
    d = |pᵢ − pⱼ| mod 12
    ic = min(d, 12 − d)
    iv[ic − 1] += 1
  return iv
```

The sum of all entries always equals C(n,2). Key examples: the major/minor triad (3-11) has IV = **⟨0,0,1,1,1,0⟩** — one each of minor 3rd, major 3rd, and perfect 5th, which is why triads sound harmonically balanced. The all-interval tetrachord (4-Z15) has IV = **⟨1,1,1,1,1,1⟩** — every interval class exactly once. The whole-tone scale (6-35) has IV = ⟨0,6,0,6,0,3⟩ — completely missing odd interval classes, explaining its floating, directionless quality.

### Z-relations: same sound, different structure

Z-related set classes share identical interval vectors without being related by transposition or inversion. The total count across all cardinalities:

| Cardinality | Z-pairs | Notable example |
|:-----------:|:-------:|:---------------|
| 4 | 1 | 4-Z15 / 4-Z29 (all-interval tetrachords) |
| 5 | 3 | 5-Z12/5-Z36, 5-Z17/5-Z37, 5-Z18/5-Z38 |
| 6 | 15 | Including 6-Z19/6-Z44, 6-Z28/6-Z49 |
| 7 | 3 | Complements of cardinality-5 pairs |
| 8 | 1 | Complement of cardinality-4 pair |

In total, **23 Z-related pairs** (46 individual set classes). No Z-relations exist below cardinality 4. Among hexachords specifically, 20 of the 50 classes are self-complementary while the remaining 30 form 15 Z-related pairs with their complements — a consequence of the hexachord theorem (a hexachord always shares its interval vector with its complement).

---

## 3. The 12-bit integer representation powers efficient computation

A pitch-class set maps naturally to a **12-bit integer** where bit *i* is 1 if and only if pitch class *i* belongs to the set. The C major triad {0,4,7} becomes 2⁰ + 2⁴ + 2⁷ = **145** (binary: 000010010001). This representation enables blazing-fast operations:

| Operation | Implementation | Example |
|:----------|:--------------|:--------|
| Union S∪T | `S \| T` | — |
| Intersection S∩T | `S & T` | — |
| Complement S̄ | `S ^ 0xFFF` | — |
| Membership p∈S | `(S >> p) & 1` | — |
| Cardinality | `popcount(S)` | — |
| Transposition Tₙ | `((S << n) \| (S >> (12-n))) & 0xFFF` | Rotate left by n |
| Inversion I₀ | Reverse bit order within 12 bits | Bit i → bit (12−i)%12 |

To enumerate all 224 Forte set classes, iterate through integers 0–4095, compute each prime form using bitwise rotation and inversion, and collect unique results. This runs in O(4096 × 24) operations — under a millisecond on any modern processor. The interval vector can also be computed bitwise: for each ic from 1 to 6, transpose S by ic semitones, AND with the original, and popcount the result (dividing by 2 for ic=6 since the tritone is self-complementary).

```
function intervalVectorBitwise(S):
  iv = new Array(6)
  for ic = 1 to 5:
    iv[ic-1] = popcount(S & transpose(S, ic))
  iv[5] = popcount(S & transpose(S, 6)) / 2
  return iv
```

---

## 4. Mapping Forte set classes to named chords

### Comprehensive chord interval catalog

The table below documents every standard Western chord type with its semitone structure from the root, pitch classes assuming C root, and Forte classification:

**Triads (cardinality 3):**

| Chord type | Semitones from root | PCs (C root) | Forte # |
|:-----------|:-------------------|:-------------|:--------|
| Major | [0, 4, 7] | C-E-G | 3-11B |
| Minor | [0, 3, 7] | C-E♭-G | 3-11A |
| Diminished | [0, 3, 6] | C-E♭-G♭ | 3-10 |
| Augmented | [0, 4, 8] | C-E-G♯ | 3-12 |
| Sus2 | [0, 2, 7] | C-D-G | 3-9 |
| Sus4 | [0, 5, 7] | C-F-G | 3-9 |

Major and minor triads collapse into Forte 3-11 because inversion maps [0,4,7] to [0,3,7]. Sus2 and sus4 share Forte 3-9 for the same reason. The augmented triad (3-12) divides the octave into three equal major thirds, producing **3-fold transpositional symmetry** — only 4 distinct augmented triads exist.

**Seventh chords (cardinality 4):**

| Chord type | Semitones | PCs (C root) | Forte # | Notes |
|:-----------|:---------|:-------------|:--------|:------|
| Major 7th | [0, 4, 7, 11] | C-E-G-B | 4-20 | Self-inversional |
| Minor 7th | [0, 3, 7, 10] | C-E♭-G-B♭ | 4-26 | = Major 6th inverted |
| Dominant 7th | [0, 4, 7, 10] | C-E-G-B♭ | 4-27B | Shares class with half-dim |
| Half-dim 7th | [0, 3, 6, 10] | C-E♭-G♭-B♭ | 4-27A | Inversional partner of dom7 |
| Diminished 7th | [0, 3, 6, 9] | C-E♭-G♭-B♭♭ | 4-28 | 4-fold symmetry: only 3 exist |
| Minor-major 7th | [0, 3, 7, 11] | C-E♭-G-B | 4-19A | Shares class with aug-maj7 |
| Aug-major 7th | [0, 4, 8, 11] | C-E-G♯-B | 4-19B | Inversional partner of mM7 |
| Augmented 7th | [0, 4, 8, 10] | C-E-G♯-B♭ | 4-24 | — |

The dominant 7th and half-diminished 7th share Forte 4-27 — a critical observation that reveals the deep structural kinship between these functionally different chords. The diminished 7th (4-28) has maximum symmetry among tetrachords, invariant under T₃, T₆, and T₉; consequently **only 3 distinct diminished 7th chords exist** across all 12 keys.

**Extended, added-tone, and altered chords:**

| Chord type | Semitones | PCs (C root) | Forte # |
|:-----------|:---------|:-------------|:--------|
| Major 6th / add13 | [0, 4, 7, 9] | {0,4,7,9} | 4-26 |
| Minor 6th | [0, 3, 7, 9] | {0,3,7,9} | 4-27 |
| add9 | [0, 2, 4, 7] | {0,2,4,7} | 4-22A |
| 7sus4 | [0, 5, 7, 10] | {0,5,7,10} | 4-23 |
| 7♭5 / French 6th | [0, 4, 6, 10] | {0,4,6,10} | 4-25 |
| Dominant 9th | [0, 2, 4, 7, 10] | {0,2,4,7,10} | 5-34 |
| Major 9th | [0, 2, 4, 7, 11] | {0,2,4,7,11} | 5-27B |
| Minor 9th | [0, 2, 3, 7, 10] | {0,2,3,7,10} | 5-25B |
| 6/9 | [0, 2, 4, 7, 9] | {0,2,4,7,9} | **5-35** |
| 7♭9 | [0, 1, 4, 7, 10] | {0,1,4,7,10} | 5-31B |
| 7♯9 ("Hendrix") | [0, 3, 4, 7, 10] | {0,3,4,7,10} | 5-32A |
| 7♯11 | [0, 4, 6, 7, 10] | {0,4,6,7,10} | 5-28 |

A remarkable finding: the 6/9 chord maps to **Forte 5-35** [0,2,4,7,9], which is identical to the major pentatonic scale — the most evenly distributed 5-note set. At the extreme end, a full major 13th chord {0,2,4,5,7,9,11} equals the entire diatonic scale (Forte 7-35). At cardinality 7, the distinction between "chord" and "scale" dissolves entirely.

**Augmented 6th and special chords:**

| Chord type | Semitones from bass | Forte # | Enharmonic equivalent |
|:-----------|:-------------------|:--------|:---------------------|
| Italian 6th | [0, 4, 10] | 3-8 | Incomplete dom7 |
| French 6th | [0, 4, 6, 10] | 4-25 | Dominant 7♭5 |
| German 6th | [0, 4, 7, 10] | 4-27B | Dominant 7th |
| Mystic chord (Scriabin) | [0, 2, 4, 6, 9, 10] | 6-34 | Stacked 4ths from C |
| Petrushka chord | [0, 1, 4, 6, 7, 10] | 6-30 | Two major triads a tritone apart |
| Tristan chord | [0, 3, 6, 10] | 4-27A | Half-diminished 7th |

### How many sets have names versus none

Of the **208 Forte set classes** at cardinalities 3–9, approximately **30 have at least one standard chord name** in jazz or classical harmony. Another **15–20** have scale names (diatonic, pentatonic, octatonic, whole-tone). The remaining **~160 set classes (~77%)** have no conventional tonal name — they exist as sonorities used in atonal and post-tonal composition but lack any chord symbol. The naming "sweet spot" is cardinalities 3–4, where roughly half of trichords and a third of tetrachords carry standard names. By cardinality 5, fewer than 20% do.

### Enharmonic equivalence creates systematic naming ambiguity

The same pitch-class set frequently supports multiple valid chord names, creating a fundamental challenge for any naming algorithm:

**Symmetry-induced equivalence.** The diminished 7th's 4-fold symmetry means Cdim7 = E♭dim7 = F♯dim7 = Adim7 — four names for {0,3,6,9}. The augmented triad's 3-fold symmetry gives Caug = Eaug = G♯aug. These are not "different chords with the same notes" but rather the same abstract object viewed from different roots.

**Root-dependent reinterpretation.** C6 = {0,4,7,9} = Am7 = {9,0,4,7}. Both are Forte 4-26. Similarly, Cm6 = Am7♭5, and this pattern generalizes: **every minor 7th chord is enharmonically a major 6th chord** of its relative major (a minor 3rd up). The German augmented 6th is pitch-class identical to a dominant 7th — this enharmonic pivot has driven modulation techniques for centuries.

**Rootless voicings create cross-chord equivalence.** Removing the root from Cmaj9 {0,2,4,7,11} yields {2,4,7,11} = Em7. This principle underpins jazz upper-structure theory, where a pianist's left hand plays what is functionally a different chord than what the bass implies.

For ChordSpace, the implementation strategy is a **lookup table mapping each 12-bit PCS to a ranked list of interpretations**, scored by commonness. The bass note or user-specified root disambiguates when multiple names are valid.

---

## 5. From abstract sets to concrete voicings: combinatorial explosion and constraint filtering

### The voicing count formula

A pitch-class set is abstract — {0,4,7} says nothing about which octave each note occupies. A **voicing** is a concrete realization: specific MIDI note numbers. Given a chord with *k* distinct pitch classes and a MIDI range [low, high], each pitch class *p* has count(p) = ⌊(high − first(p))/12⌋ + 1 available octave placements, where first(p) is the lowest MIDI note ≥ low with pitch class p.

**Without doubling** (exactly one instance of each pitch class): total voicings = **∏ count(pᵢ)**

**With doubling** (each pitch class appears at least once, possibly in multiple octaves): total voicings = **∏ (2^count(pᵢ) − 1)**

Worked example for C major triad in MIDI 36–96: C has 6 instances, E has 5, G has 5. Without doubling: **6 × 5 × 5 = 150 voicings**. With doubling: 63 × 31 × 31 = **60,543 voicings**. For a dominant 13th chord (7 pitch classes) in the same range, the no-doubling count is 6 × 5⁶ = **93,750**, and with doubling the number exceeds **55 billion**. This combinatorial explosion makes constraint-based filtering not optional but essential.

### Algorithmic definitions of voicing types

**Close position.** All chord tones within a single octave span, sorted ascending. Generated by placing each interval above a base MIDI note, wrapping any notes exceeding 11 semitones above the bass back into the octave. Span ≤ 11 semitones. Each close-position voicing has *k* inversions (rotations), produced by moving the lowest note up an octave.

**Drop-2.** Start from a close-position voicing sorted descending. Subtract 12 from the 2nd note from the top. Re-sort ascending. This creates a characteristic span of roughly a 10th and is the most common jazz guitar voicing type. Each 4-note chord has 4 drop-2 voicings (one per inversion as starting point).

```
function drop2(closeVoicing):
  desc = closeVoicing.sortDescending()
  desc[1] -= 12
  return desc.sortAscending()
```

**Drop-3.** Same procedure but dropping the 3rd note from the top. Produces wider bass separation. On guitar, typically played across string sets (6,5,4,2) or (5,4,3,1), skipping one internal string.

**Drop-2-4.** Drop both the 2nd and 4th notes from the top. Very wide, orchestral-feeling voicings.

**Shell voicing.** The minimum viable harmonic skeleton: root + 3rd + 7th only. The 5th is omitted (it adds little color) and extensions are left to context. Two variants exist: root-3rd-7th (Bud Powell style) and root-7th-3rd (inverted shell with 7th below 3rd for a darker sound).

```
function shellVoicing(root, quality, octave):
  base = root + octave * 12
  third = (quality.includes('minor')) ? 3 : 4
  seventh = (quality.includes('maj7')) ? 11 : 10
  return [base, base + third, base + seventh].sort()
```

**Rootless voicings.** Omit the root (assumed in the bass). Jazz piano standard formulas:
- Type A (major/minor): 3-5-7-9
- Type B (major/minor): 7-9-3-5
- Type A (dominant): 3-13-♭7-9
- Type B (dominant): ♭7-9-3-13

A practical "rule of thumb" constrains the top note of a left-hand rootless voicing to fall between **MIDI 60 (C4) and 72 (C5)** for optimal clarity.

**Quartal voicing.** Chord tones rearranged in stacked perfect 4ths (5 semitones) or augmented 4ths (6 semitones), drawn from a parent scale. The algorithm iterates through permutations of pitch classes, seeking arrangements where successive intervals are 5 or 6 semitones. Not every chord type admits a quartal voicing; the technique works best with modal harmony (the "So What" voicing — three perfect 4ths plus a major 3rd — is the canonical example).

**Spread voicing.** Bass note isolated below MIDI 48; upper structure distributed across MIDI 60–84. The algorithm places the root (or 5th) in the low register and arranges remaining tones in the treble, ensuring no adjacent upper voices exceed an octave apart.

---

## 6. Register constraints from psychoacoustics to playability

### Critical bandwidth determines minimum spacing

The **critical bandwidth** — the frequency range within which two simultaneous tones produce audible roughness — widens relative to pitch in lower registers. The ERB formula (Glasberg & Moore, 1990) gives ERB(f) = 24.7 × (4.37f/1000 + 1) Hz. Converted to musically useful thresholds:

| Register | MIDI range | Min. adjacent interval | Musical equivalent |
|:---------|:----------|:----------------------|:-------------------|
| Below C2 | < 36 | **7 semitones** | Perfect 5th |
| C2–C3 | 36–48 | **5 semitones** | Perfect 4th |
| C3–C4 | 48–60 | **4 semitones** | Major 3rd |
| C4–C5 | 60–72 | **3 semitones** | Minor 3rd |
| Above C5 | > 72 | **2 semitones** | Major 2nd |

This explains why a major 3rd sounds rich at C4 but muddy at C2 — at 100 Hz the critical bandwidth spans nearly the entire fundamental frequency, while at 400 Hz it covers only 27%. The implementation should encode this as a function:

```
function registerMinInterval(midiNote):
  if midiNote < 36: return 7
  if midiNote < 48: return 5
  if midiNote < 60: return 4
  if midiNote < 72: return 3
  return 2
```

### Doubling and voice-leading heuristics

Professional arrangers follow consistent rules. **The root is always safe to double**, the 5th can be doubled if the root is present, but **the 3rd and 7th should rarely or never be doubled** — they color the harmony too strongly and the 7th is a tendency tone requiring resolution. In classical harmony, the leading tone must never be doubled. For voicing generation, a scoring function should penalize doublings of 3rds and 7ths while rewarding root doublings.

### Guitar-specific physical constraints

Guitar voicing generation is a constraint-satisfaction problem with physical parameters: standard tuning (MIDI 40-45-50-55-59-64 for strings 6–1), a **maximum 4-fret stretch** between lowest and highest fretted positions, at most **4 fretting fingers** (with barre counting as one), and a strong preference against internal muted strings (muting string 5 while playing strings 6 and 4 requires awkward technique).

```
function guitarVoicings(pitchClassSet, tuning, maxStretch=4):
  results = []
  for fretWindow in 0..12:
    for each string in 0..5:
      options = [MUTED] ∪ {fret : 0 ≤ fret ≤ 20, (tuning[string]+fret)%12 ∈ pitchClassSet}
    for each combination of string assignments:
      if frettedSpan ≤ maxStretch
         AND uniqueFretPositions ≤ 4
         AND noInternalMutedStrings
         AND playedPitchClasses ⊇ pitchClassSet:
        results.push(combination)
  return results
```

Research by Tuohy & Potter at UGA demonstrated that a genetic algorithm with fitness components for hand movement, fret distance, and string distance achieves 91.1% agreement with human-created tablature — suggesting that a scoring function combining these factors can rank guitar voicings effectively.

---

## 7. Data structures tying it all together

### Core type definitions

```typescript
interface PitchClassSet {
  bits: number;              // 12-bit integer (0x000–0xFFF)
  cardinality: number;       // popcount(bits)
  primeForm: number;         // canonical 12-bit representation
  forteNumber: string;       // "3-11", "4-Z15", etc.
  intervalVector: [number, number, number, number, number, number];
}

interface Chord {
  pcs: PitchClassSet;
  root: number;              // 0–11
  quality: string;           // "maj7", "m7b5", "dom13"
  symbol: string;            // "Cmaj7", "F#m7b5"
  intervals: number[];       // sorted semitones from root: [0, 4, 7, 11]
}

interface Voicing {
  midi: number[];            // sorted ascending MIDI notes
  chord: Chord;
  type: VoicingType;         // Close | Drop2 | Drop3 | Shell | Quartal | ...
  bass: number;              // lowest MIDI note
  span: number;              // highest − lowest
  adjacentIntervals: number[]; // intervals between consecutive voices
}

interface VoicingConstraints {
  midiRange: [number, number];
  maxSpan: number;
  maxVoices: number;
  allowDoubling: boolean;
  instrument: 'piano' | 'guitar' | 'orchestral';
  registerMinInterval: (midi: number) => number;
  // Guitar-specific
  tuning?: number[];
  maxFretStretch?: number;
  allowInternalMutes?: boolean;
}
```

### The master lookup table

The central data structure is a **HashMap from 12-bit PCS integer to a ranked list of chord interpretations**. Since there are only 4,096 possible keys, this table can be pre-computed at application startup in milliseconds. Each entry stores all valid (root, quality) pairs sorted by commonness:

```typescript
const chordLookup: Map<number, ChordInterpretation[]> = new Map();

interface ChordInterpretation {
  root: number;
  quality: string;
  commonness: number;   // 1.0 = primary interpretation
  intervals: number[];
}

// Example: bits 145 → [{root:0, quality:"major", commonness:1.0},
//                       {root:4, quality:"min(1st inv)", commonness:0.3}]
```

A parallel **HashMap from Forte number to set class metadata** supports the theoretical analysis layer, and a **pre-computed array of all 224 prime forms** with their properties enables instant set-class identification. The voicing generation pipeline flows: user selects chord → lookup PCS bits → apply voicing algorithm → filter by constraints → score by register rules → rank and present.

---

## Conclusion: three layers, one engine

ChordSpace's architecture maps directly to three mathematical layers. The **enumeration layer** exploits the 12-bit integer representation and Burnside's lemma to guarantee completeness — all 224 Forte set classes (or 352 under transposition only) are discoverable through a single pass over 4,096 integers. The **naming layer** confronts the fundamental asymmetry that only ~30 of 208 cataloged set classes carry standard chord names, and that enharmonic equivalence makes the mapping from pitch-class sets to names inherently one-to-many; a ranked interpretation table with bass-note disambiguation resolves this. The **voicing layer** tames combinatorial explosion (150 to 55 billion voicings per chord type) through structured algorithms (drop-2, shell, quartal, rootless) and psychoacoustically grounded constraints (critical bandwidth thresholds, doubling rules, instrument-specific physical limits).

The key architectural insight is that these three layers compose cleanly: enumerate → name → voice. The 12-bit integer representation threads through all three, making transposition a bit rotation, set intersection a bitwise AND, and prime form computation a 24-iteration loop. For a browser-based engine, the entire Forte catalog fits in a few kilobytes, voicing algorithms run in microseconds per chord, and the constraint-filtering pipeline can be parallelized across Web Workers for real-time interaction. The mathematical foundations are not merely theoretical scaffolding — they are the implementation itself.