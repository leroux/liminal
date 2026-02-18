// ─── Pitch-class set engine ─────────────────────────────────────────
// Every PCS is a 12-bit integer (u16). Bit i = 1 iff pitch class i is present.
// C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11

export const NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

export function pcsToBits(pcs: number[]): number {
  let bits = 0;
  for (const pc of pcs) bits |= 1 << (pc % 12);
  return bits;
}

export function bitsToArray(bits: number): number[] {
  const result: number[] = [];
  for (let i = 0; i < 12; i++) {
    if (bits & (1 << i)) result.push(i);
  }
  return result;
}

export function popcount(x: number): number {
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

export function transposePCS(bits: number, n: number): number {
  n = ((n % 12) + 12) % 12;
  if (n === 0) return bits;
  return ((bits << n) | (bits >> (12 - n))) & 0xFFF;
}

export function invertPCS(bits: number): number {
  let result = 0;
  for (let i = 0; i < 12; i++) {
    if (bits & (1 << i)) result |= 1 << ((12 - i) % 12);
  }
  return result;
}

export function intervalVector(bits: number): [number, number, number, number, number, number] {
  const iv: [number, number, number, number, number, number] = [0, 0, 0, 0, 0, 0];
  for (let ic = 1; ic <= 5; ic++) {
    iv[ic - 1] = popcount(bits & transposePCS(bits, ic));
  }
  iv[5] = popcount(bits & transposePCS(bits, 6)) / 2;
  return iv;
}

export function commonTones(a: number, b: number): number {
  return popcount(a & b);
}

// ─── Normal form & prime form ───────────────────────────────────────

function normalFormArray(pcs: number[]): number[] {
  if (pcs.length === 0) return [];
  const sorted = [...pcs].sort((a, b) => a - b);
  const k = sorted.length;
  let best: number[] | null = null;

  for (let r = 0; r < k; r++) {
    const rotation: number[] = [];
    for (let i = 0; i < k; i++) {
      let val = sorted[(r + i) % k];
      if (i > 0 && val <= rotation[0]) val += 12;
      rotation.push(val);
    }
    for (let i = 1; i < k; i++) {
      if (rotation[i] < rotation[i - 1]) rotation[i] += 12;
    }

    const span = rotation[k - 1] - rotation[0];

    if (best === null) {
      best = rotation;
    } else {
      const bestSpan = best[k - 1] - best[0];
      if (span < bestSpan) {
        best = rotation;
      } else if (span === bestSpan) {
        for (let i = 1; i < k; i++) {
          const a = rotation[i] - rotation[0];
          const b = best[i] - best[0];
          if (a < b) { best = rotation; break; }
          if (a > b) break;
        }
      }
    }
  }

  const offset = best![0];
  return best!.map(x => (x - offset) % 12);
}

export function primeForm(bits: number): number[] {
  const pcs = bitsToArray(bits);
  if (pcs.length === 0) return [];
  const a = normalFormArray(pcs);
  const invBits = invertPCS(bits);
  const invPcs = bitsToArray(invBits);
  const b = normalFormArray(invPcs);

  for (let i = 0; i < a.length; i++) {
    if (a[i] < b[i]) return a;
    if (a[i] > b[i]) return b;
  }
  return a;
}

export function primeFormBits(bits: number): number {
  return pcsToBits(primeForm(bits));
}

// ─── Chord naming ───────────────────────────────────────────────────

export interface ChordName {
  root: number;
  quality: string;
  symbol: string;
}

interface ChordTemplate {
  intervals: number[];
  quality: string;
  suffix: string;
}

const CHORD_TEMPLATES: ChordTemplate[] = [
  // Triads
  { intervals: [0, 4, 7], quality: 'major', suffix: '' },
  { intervals: [0, 3, 7], quality: 'minor', suffix: 'm' },
  { intervals: [0, 3, 6], quality: 'diminished', suffix: 'dim' },
  { intervals: [0, 4, 8], quality: 'augmented', suffix: 'aug' },
  { intervals: [0, 2, 7], quality: 'sus2', suffix: 'sus2' },
  { intervals: [0, 5, 7], quality: 'sus4', suffix: 'sus4' },
  // Seventh chords
  { intervals: [0, 4, 7, 11], quality: 'major7', suffix: 'maj7' },
  { intervals: [0, 3, 7, 10], quality: 'minor7', suffix: 'm7' },
  { intervals: [0, 4, 7, 10], quality: 'dominant7', suffix: '7' },
  { intervals: [0, 3, 6, 10], quality: 'half-dim7', suffix: 'm7b5' },
  { intervals: [0, 3, 6, 9], quality: 'diminished7', suffix: 'dim7' },
  { intervals: [0, 3, 7, 11], quality: 'minor-major7', suffix: 'mMaj7' },
  { intervals: [0, 4, 8, 11], quality: 'aug-major7', suffix: 'augMaj7' },
  { intervals: [0, 4, 8, 10], quality: 'augmented7', suffix: 'aug7' },
  // Added-tone & extended
  { intervals: [0, 4, 7, 9], quality: 'major6', suffix: '6' },
  { intervals: [0, 3, 7, 9], quality: 'minor6', suffix: 'm6' },
  { intervals: [0, 2, 4, 7], quality: 'add9', suffix: 'add9' },
  { intervals: [0, 5, 7, 10], quality: '7sus4', suffix: '7sus4' },
  { intervals: [0, 4, 6, 10], quality: '7b5', suffix: '7b5' },
  // 9ths
  { intervals: [0, 2, 4, 7, 10], quality: 'dominant9', suffix: '9' },
  { intervals: [0, 2, 4, 7, 11], quality: 'major9', suffix: 'maj9' },
  { intervals: [0, 2, 3, 7, 10], quality: 'minor9', suffix: 'm9' },
  { intervals: [0, 1, 4, 7, 10], quality: '7b9', suffix: '7b9' },
  { intervals: [0, 3, 4, 7, 10], quality: '7#9', suffix: '7#9' },
  { intervals: [0, 4, 6, 7, 10], quality: '7#11', suffix: '7#11' },
  { intervals: [0, 2, 4, 7, 9], quality: '6/9', suffix: '6/9' },
];

export { CHORD_TEMPLATES };

// Pre-computed: for each of the 4096 PCS bit patterns, store ranked chord names
const chordLookup = new Map<number, ChordName[]>();

function buildChordLookup(): void {
  for (const tpl of CHORD_TEMPLATES) {
    for (let root = 0; root < 12; root++) {
      const bits = pcsToBits(tpl.intervals.map(i => (i + root) % 12));
      const name: ChordName = {
        root,
        quality: tpl.quality,
        symbol: NOTE_NAMES[root] + tpl.suffix,
      };
      const existing = chordLookup.get(bits);
      if (existing) {
        existing.push(name);
      } else {
        chordLookup.set(bits, [name]);
      }
    }
  }
}

buildChordLookup();

export function nameChord(bits: number): ChordName | null {
  const names = chordLookup.get(bits);
  return names ? names[0] : null;
}

export function allChordNames(bits: number): ChordName[] {
  return chordLookup.get(bits) || [];
}

// ─── Forte catalog: all 224 set classes ─────────────────────────────

export interface SetClass {
  id: number;               // ordinal within this cardinality
  cardinality: number;
  primeFormBits: number;
  primeFormArray: number[];
  iv: [number, number, number, number, number, number];
  members: number[];         // all transpositions (PCS bits) in this class
  names: ChordName[];        // chord names for any member (empty if unnamed)
  multiplicity: number;      // how many distinct transpositions exist
}

export interface ForteEntry {
  bits: number;
  cardinality: number;
  setClass: SetClass;
  names: ChordName[];
}

// Enumerate all 224 TnI set classes
function buildForteCatalog(): { classes: SetClass[]; lookup: Map<number, SetClass> } {
  const primeToClass = new Map<number, SetClass>();
  const pcsToClass = new Map<number, SetClass>();
  let ordinals = new Map<number, number>(); // cardinality → next ordinal

  for (let bits = 0; bits < 4096; bits++) {
    const card = popcount(bits);
    if (card < 2 || card > 7) continue; // skip trivial (0,1) and large (8+, complements of small)

    const pf = primeFormBits(bits);
    if (primeToClass.has(pf)) {
      // Add this PCS as a member of existing class
      const sc = primeToClass.get(pf)!;
      sc.members.push(bits);
      pcsToClass.set(bits, sc);
      continue;
    }

    // New set class
    const id = (ordinals.get(card) || 0) + 1;
    ordinals.set(card, id);

    const pfArray = primeForm(bits);
    const iv = intervalVector(pf);

    // Collect all chord names for any transposition of this class
    const names: ChordName[] = [];
    // We'll fill names after collecting all members

    const sc: SetClass = {
      id,
      cardinality: card,
      primeFormBits: pf,
      primeFormArray: pfArray,
      iv,
      members: [bits],
      names,
      multiplicity: 0,
    };

    primeToClass.set(pf, sc);
    pcsToClass.set(bits, sc);
  }

  // Finalize: collect names, compute multiplicity
  const classes: SetClass[] = [];
  for (const sc of primeToClass.values()) {
    // Deduplicate members
    sc.members = [...new Set(sc.members)];
    sc.multiplicity = sc.members.length;

    // Collect all chord names across all members
    for (const memberBits of sc.members) {
      const n = allChordNames(memberBits);
      sc.names.push(...n);
    }

    classes.push(sc);
  }

  // Sort by cardinality then ordinal
  classes.sort((a, b) => a.cardinality - b.cardinality || a.id - b.id);

  return { classes, lookup: pcsToClass };
}

const forteCatalog = buildForteCatalog();
export const FORTE_CLASSES = forteCatalog.classes;
export const PCS_TO_CLASS = forteCatalog.lookup;

// ─── Full chord vocabulary: every named chord × 12 roots ────────────

export interface ChordEntry {
  bits: number;
  root: number;
  quality: string;
  symbol: string;
  intervals: number[];
  cardinality: number;
  iv: [number, number, number, number, number, number];
  setClass: SetClass | undefined;
}

function buildChordVocabulary(): ChordEntry[] {
  const entries: ChordEntry[] = [];
  const seen = new Set<string>(); // deduplicate by bits+root

  for (const tpl of CHORD_TEMPLATES) {
    for (let root = 0; root < 12; root++) {
      const pcs = tpl.intervals.map(i => (i + root) % 12);
      const bits = pcsToBits(pcs);
      const key = `${bits}-${root}`;
      if (seen.has(key)) continue;
      seen.add(key);

      entries.push({
        bits,
        root,
        quality: tpl.quality,
        symbol: NOTE_NAMES[root] + tpl.suffix,
        intervals: tpl.intervals,
        cardinality: tpl.intervals.length,
        iv: intervalVector(bits),
        setClass: PCS_TO_CLASS.get(bits),
      });
    }
  }

  return entries;
}

export const CHORD_VOCABULARY = buildChordVocabulary();

// ─── Triad utilities (kept for Tonnetz view) ────────────────────────

export interface Triad {
  root: number;
  isMajor: boolean;
  bits: number;
  name: string;
}

export function makeTriad(root: number, isMajor: boolean): Triad {
  const third = isMajor ? 4 : 3;
  const bits = pcsToBits([root, (root + third) % 12, (root + 7) % 12]);
  const suffix = isMajor ? '' : 'm';
  return { root, isMajor, bits, name: NOTE_NAMES[root] + suffix };
}

export const ALL_TRIADS: Triad[] = [];
for (let root = 0; root < 12; root++) {
  ALL_TRIADS.push(makeTriad(root, true));
  ALL_TRIADS.push(makeTriad(root, false));
}

// ─── MIDI / frequency helpers ───────────────────────────────────────

export function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function pcToMidi(pc: number, octave: number): number {
  return pc + (octave + 1) * 12;
}

// ─── Voicing engine ─────────────────────────────────────────────────

export type InversionType = 'root' | '1st' | '2nd' | '3rd' | '4th';

// Close-position voicing with inversion support
export function closeVoicing(bits: number, baseOctave: number = 4, inversion: number = 0): number[] {
  const pcs = bitsToArray(bits);
  if (pcs.length === 0) return [];

  const name = nameChord(bits);
  const root = name ? name.root : pcs[0];

  // Build ascending from root
  const midi: number[] = [];
  for (const pc of pcs) {
    let note = pcToMidi(pc, baseOctave);
    const rootMidi = pcToMidi(root, baseOctave);
    while (note < rootMidi) note += 12;
    midi.push(note);
  }
  midi.sort((a, b) => a - b);

  // Apply inversion: move bottom N notes up an octave
  const inv = Math.min(inversion, midi.length - 1);
  for (let i = 0; i < inv; i++) {
    midi[i] += 12;
  }
  midi.sort((a, b) => a - b);

  return midi;
}

// Drop-2 voicing: start from close position descending, drop 2nd from top by octave
export function drop2Voicing(bits: number, baseOctave: number = 4, inversion: number = 0): number[] {
  const close = closeVoicing(bits, baseOctave, inversion);
  if (close.length < 4) return close;
  const desc = [...close].sort((a, b) => b - a);
  desc[1] -= 12;
  return desc.sort((a, b) => a - b);
}

// Shell voicing: root + 3rd + 7th (skip 5th)
export function shellVoicing(bits: number, baseOctave: number = 4): number[] {
  const name = nameChord(bits);
  if (!name) return closeVoicing(bits, baseOctave);

  const root = name.root;
  const pcs = bitsToArray(bits);
  if (pcs.length < 3) return closeVoicing(bits, baseOctave);

  // Find third (3 or 4 semitones from root) and seventh (10 or 11)
  const rootMidi = pcToMidi(root, baseOctave);
  const result = [rootMidi];

  for (const offset of [3, 4]) {
    const pc = (root + offset) % 12;
    if (pcs.includes(pc)) {
      result.push(rootMidi + offset);
      break;
    }
  }

  for (const offset of [10, 11]) {
    const pc = (root + offset) % 12;
    if (pcs.includes(pc)) {
      result.push(rootMidi + offset);
      break;
    }
  }

  if (result.length < 3) return closeVoicing(bits, baseOctave);
  return result.sort((a, b) => a - b);
}

// Spread voicing: bass isolated low, upper voices spread higher
export function spreadVoicing(bits: number, baseOctave: number = 3): number[] {
  const close = closeVoicing(bits, baseOctave + 1);
  if (close.length < 2) return close;

  const name = nameChord(bits);
  const root = name ? name.root : bitsToArray(bits)[0];
  const bassMidi = pcToMidi(root, baseOctave);

  return [bassMidi, ...close.slice(1)];
}

// Rootless Type A voicing: 3rd, 5th, 7th, 9th (from bottom)
export function rootlessAVoicing(bits: number, baseOctave: number = 4): number[] {
  const name = nameChord(bits);
  if (!name) return closeVoicing(bits, baseOctave);

  const root = name.root;
  const pcs = bitsToArray(bits);
  if (pcs.length < 3) return closeVoicing(bits, baseOctave);

  const rootMidi = pcToMidi(root, baseOctave);
  const result: number[] = [];

  // 3rd
  for (const offset of [3, 4]) {
    const pc = (root + offset) % 12;
    if (pcs.includes(pc)) { result.push(rootMidi + offset); break; }
  }
  // 5th (or b5/#5)
  for (const offset of [6, 7, 8]) {
    const pc = (root + offset) % 12;
    if (pcs.includes(pc)) { result.push(rootMidi + offset); break; }
  }
  // 7th
  for (const offset of [10, 11]) {
    const pc = (root + offset) % 12;
    if (pcs.includes(pc)) { result.push(rootMidi + offset); break; }
  }
  // 9th (2 semitones above root, up an octave)
  const ninth = (root + 2) % 12;
  if (pcs.includes(ninth)) {
    result.push(rootMidi + 14); // 2 + 12
  } else {
    // add 9th even if not in chord for color
    result.push(rootMidi + 14);
  }

  if (result.length < 3) return closeVoicing(bits, baseOctave);
  return result.sort((a, b) => a - b);
}

// Rootless Type B voicing: 7th, 9th, 3rd, 5th (from bottom)
export function rootlessBVoicing(bits: number, baseOctave: number = 3): number[] {
  const name = nameChord(bits);
  if (!name) return closeVoicing(bits, baseOctave);

  const root = name.root;
  const pcs = bitsToArray(bits);
  if (pcs.length < 3) return closeVoicing(bits, baseOctave);

  const baseMidi = pcToMidi(root, baseOctave);
  const result: number[] = [];

  // 7th in low octave
  for (const offset of [10, 11]) {
    const pc = (root + offset) % 12;
    if (pcs.includes(pc)) { result.push(baseMidi + offset); break; }
  }
  // 9th
  result.push(baseMidi + 14); // root + 2 + 12
  // 3rd up an octave
  for (const offset of [3, 4]) {
    const pc = (root + offset) % 12;
    if (pcs.includes(pc)) { result.push(baseMidi + offset + 12); break; }
  }
  // 5th up an octave
  for (const offset of [6, 7, 8]) {
    const pc = (root + offset) % 12;
    if (pcs.includes(pc)) { result.push(baseMidi + offset + 12); break; }
  }

  if (result.length < 3) return closeVoicing(bits, baseOctave);
  return result.sort((a, b) => a - b);
}

// Quartal voicing: stack notes in 4ths (5 semitones) where possible
export function quartalVoicing(bits: number, baseOctave: number = 4): number[] {
  const pcs = bitsToArray(bits);
  if (pcs.length < 3) return closeVoicing(bits, baseOctave);

  // Try all starting PCs, pick the arrangement with most 4th intervals
  let bestVoicing: number[] = [];
  let bestFourths = -1;

  for (const startPc of pcs) {
    const voicing: number[] = [pcToMidi(startPc, baseOctave)];
    const remaining = pcs.filter(pc => pc !== startPc);

    let currentMidi = voicing[0];
    const used = new Set([startPc]);

    for (let i = 0; i < remaining.length; i++) {
      // Find next note that's closest to a P4 (5 semitones) above current
      let bestNext = -1;
      let bestDist = Infinity;
      for (const pc of pcs) {
        if (used.has(pc)) continue;
        let midi = pcToMidi(pc, baseOctave);
        while (midi <= currentMidi) midi += 12;
        const interval = midi - currentMidi;
        const distFrom4th = Math.abs(interval - 5);
        const distFrom4thOctave = Math.abs(interval - 17); // 4th + octave
        const d = Math.min(distFrom4th, distFrom4thOctave);
        if (d < bestDist) {
          bestDist = d;
          bestNext = midi;
          // Also try wrapping
        }
      }
      if (bestNext >= 0) {
        voicing.push(bestNext);
        used.add(bestNext % 12);
        currentMidi = bestNext;
      }
    }

    // Count actual P4 intervals (5 semitones)
    let fourths = 0;
    for (let i = 1; i < voicing.length; i++) {
      const interval = voicing[i] - voicing[i - 1];
      if (interval === 5 || interval === 6) fourths++; // P4 or tritone
    }

    if (fourths > bestFourths || (fourths === bestFourths && voicing.length > bestVoicing.length)) {
      bestFourths = fourths;
      bestVoicing = voicing;
    }
  }

  return bestVoicing.length > 0 ? bestVoicing : closeVoicing(bits, baseOctave);
}

// ─── Register constraints ────────────────────────────────────────────
// Apply minimum spacing rules by register for cleaner voicings.
// Below C3 (MIDI 48): min 7 semitones between adjacent voices (avoids muddiness)
// C3–C5 (48–72): min 3 semitones
// Above C5 (72+): min 1 semitone (close clusters ok)

export function applyRegisterConstraints(midi: number[]): number[] {
  if (midi.length < 2) return midi;
  const sorted = [...midi].sort((a, b) => a - b);
  const result = [sorted[0]];

  for (let i = 1; i < sorted.length; i++) {
    let note = sorted[i];
    const prev = result[result.length - 1];
    const minInterval = prev < 48 ? 7 : prev < 72 ? 3 : 1;
    while (note - prev < minInterval && note < 108) {
      note += 12;
    }
    result.push(note);
  }

  return result;
}

// Avoid doubling the 3rd in upper register (sounds harsh)
export function applyDoublingRules(midi: number[], bits: number): number[] {
  if (midi.length < 4) return midi;
  const name = nameChord(bits);
  if (!name) return midi;

  const root = name.root;
  // Find the 3rd (major or minor)
  const pcs = bitsToArray(bits);
  let thirdPc = -1;
  for (const offset of [3, 4]) {
    const pc = (root + offset) % 12;
    if (pcs.includes(pc)) { thirdPc = pc; break; }
  }
  if (thirdPc < 0) return midi;

  // Count 3rd occurrences; if doubled above C5, drop one
  const thirdNotes = midi.filter(n => n % 12 === thirdPc && n > 72);
  if (thirdNotes.length > 1) {
    // Remove the highest duplicate 3rd
    const removeNote = thirdNotes[thirdNotes.length - 1];
    const idx = midi.lastIndexOf(removeNote);
    const result = [...midi];
    result.splice(idx, 1);
    return result;
  }

  return midi;
}

// ─── Evenness (how evenly notes divide the octave) ──────────────────
// Uses magnitude of the cardinality-th DFT coefficient, normalized to 0–1.
// 1.0 = perfectly even (augmented triad, dim7), 0.0 = maximally clustered.

export function evenness(bits: number): number {
  const pcs = bitsToArray(bits);
  const n = pcs.length;
  if (n < 2) return 1;

  // Compute DFT magnitude at frequency index = n (the "evenness" coefficient)
  // For a perfectly even n-note chord, this coefficient is maximal
  let re = 0, im = 0;
  for (const pc of pcs) {
    const angle = 2 * Math.PI * pc * n / 12;
    re += Math.cos(angle);
    im += Math.sin(angle);
  }
  const mag = Math.sqrt(re * re + im * im) / n;
  return mag;
}

// ─── Catalog stats ──────────────────────────────────────────────────

export function catalogStats(): {
  totalPCS: number;
  forteClasses: number;
  namedTypes: number;
  totalNamed: number;
  byCardinality: Map<number, number>;
} {
  const byCard = new Map<number, number>();
  for (const sc of FORTE_CLASSES) {
    byCard.set(sc.cardinality, (byCard.get(sc.cardinality) || 0) + 1);
  }
  const namedTypes = new Set(CHORD_TEMPLATES.map(t => t.quality)).size;

  return {
    totalPCS: 4096,
    forteClasses: FORTE_CLASSES.length,
    namedTypes,
    totalNamed: CHORD_VOCABULARY.length,
    byCardinality: byCard,
  };
}
