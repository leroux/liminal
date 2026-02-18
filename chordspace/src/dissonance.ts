// ─── Vassilakis roughness model ─────────────────────────────────────
// Computes perceptual roughness/dissonance from partial frequencies & amplitudes.

import { midiToFreq } from './core.js';

export interface Partial {
  freq: number;
  amp: number;
}

// Overtone profile: array of (frequency_ratio, relative_amplitude)
export interface OvertoneProfile {
  name: string;
  partials: [number, number][]; // [ratio, amplitude]
}

// ─── Built-in timbre profiles ───────────────────────────────────────

export const TIMBRES: Record<string, OvertoneProfile> = {
  sine: {
    name: 'Sine',
    partials: [[1, 1.0]],
  },
  piano: {
    name: 'Piano',
    partials: [
      [1, 1.0], [2, 0.7], [3, 0.45], [4, 0.3], [5, 0.2],
      [6, 0.15], [7, 0.1], [8, 0.08], [9, 0.05], [10, 0.03],
      [11, 0.02], [12, 0.015],
    ],
  },
  rhodes: {
    name: 'Rhodes',
    partials: [
      [1, 1.0], [2, 0.6], [3, 0.25], [4, 0.15], [5, 0.08],
      [6, 0.04], [7, 0.02], [8, 0.01],
    ],
  },
  organ: {
    name: 'Organ',
    partials: [
      [0.5, 0.5], [1, 1.0], [1.5, 0.3], [2, 0.8], [3, 0.6],
      [4, 0.5], [6, 0.3], [8, 0.2],
    ],
  },
  saw: {
    name: 'Saw Pad',
    partials: [
      [1, 1.0], [2, 0.5], [3, 0.333], [4, 0.25], [5, 0.2],
      [6, 0.167], [7, 0.143], [8, 0.125], [9, 0.111], [10, 0.1],
      [11, 0.091], [12, 0.083],
    ],
  },
  square: {
    name: 'Square',
    partials: [
      [1, 1.0], [3, 0.333], [5, 0.2], [7, 0.143],
      [9, 0.111], [11, 0.091], [13, 0.077],
    ],
  },
  bell: {
    name: 'Bell',
    partials: [
      [1, 1.0], [2.0, 0.6], [2.756, 0.4], [4.0, 0.25],
      [5.404, 0.2], [6.0, 0.15], [8.933, 0.1],
    ],
  },
  wurlitzer: {
    name: 'Wurlitzer',
    partials: [
      [1, 1.0], [2, 0.55], [3, 0.35], [4, 0.2], [5, 0.12],
      [6, 0.06], [7, 0.03],
    ],
  },
  plucked: {
    name: 'Plucked String',
    partials: [
      [1, 1.0], [2, 0.5], [3, 0.33], [4, 0.18], [5, 0.12],
      [6, 0.08], [7, 0.06], [8, 0.04], [9, 0.03], [10, 0.02],
    ],
  },
  clavinet: {
    name: 'Clavinet',
    partials: [
      [1, 0.7], [2, 1.0], [3, 0.8], [4, 0.5], [5, 0.4],
      [6, 0.3], [7, 0.25], [8, 0.15], [9, 0.1],
    ],
  },
  subbass: {
    name: 'Sub Bass',
    partials: [
      [1, 1.0], [2, 0.3], [3, 0.05],
    ],
  },
  brass: {
    name: 'Brass Stab',
    partials: [
      [1, 1.0], [2, 0.85], [3, 0.7], [4, 0.55], [5, 0.4],
      [6, 0.3], [7, 0.22], [8, 0.15], [9, 0.1], [10, 0.07],
    ],
  },
};

export const DEFAULT_TIMBRE = 'piano';

// ─── Vassilakis pairwise roughness ──────────────────────────────────

function vassilakisPair(f1: number, f2: number, a1: number, a2: number): number {
  if (f1 > f2) { [f1, f2] = [f2, f1]; [a1, a2] = [a2, a1]; }
  const df = f2 - f1;
  if (df < 0.001) return 0;

  const aMin = Math.min(a1, a2);
  const aMax = Math.max(a1, a2);
  if (aMin < 0.001) return 0;

  const s = 0.24 / (0.0207 * f1 + 18.96);
  const Z = Math.exp(-3.5 * s * df) - Math.exp(-5.75 * s * df);

  // Vassilakis amplitude terms
  const spl = Math.pow(aMin * aMax, 0.1);
  const afDegree = Math.pow(2 * aMin / (aMin + aMax), 3.11);

  return spl * 0.5 * afDegree * Math.abs(Z);
}

// ─── Chord dissonance scoring ───────────────────────────────────────

// Expand a set of MIDI notes + overtone profile into a full partial list
function expandPartials(midiNotes: number[], profile: OvertoneProfile): Partial[] {
  const result: Partial[] = [];
  for (const midi of midiNotes) {
    const f0 = midiToFreq(midi);
    for (const [ratio, amp] of profile.partials) {
      const freq = f0 * ratio;
      if (freq < 20 || freq > 20000) continue; // audible range
      result.push({ freq, amp });
    }
  }
  return result;
}

// Total roughness for a chord: sum of all pairwise partial interactions
export function chordRoughness(midiNotes: number[], profile: OvertoneProfile): number {
  const partials = expandPartials(midiNotes, profile);
  let total = 0;

  for (let i = 0; i < partials.length; i++) {
    for (let j = i + 1; j < partials.length; j++) {
      total += vassilakisPair(
        partials[i].freq, partials[j].freq,
        partials[i].amp, partials[j].amp,
      );
    }
  }
  return total;
}

// Normalized consonance score: 0 = dissonant, 1 = consonant
// Uses unison as reference (minimum roughness for the timbre)
// and a tritone cluster as near-maximum roughness
export function consonanceScore(midiNotes: number[], profile: OvertoneProfile): number {
  if (midiNotes.length < 2) return 1.0;
  const roughness = chordRoughness(midiNotes, profile);

  // Normalize against number of note pairs to make comparable across cardinalities
  const pairs = midiNotes.length * (midiNotes.length - 1) / 2;
  const normalized = roughness / pairs;

  // Empirical scaling: map to 0–1 range.
  // Minor second at C4 with piano timbre ≈ 0.15 roughness per pair
  // Perfect fifth ≈ 0.01
  const maxExpected = 0.12;
  const score = 1.0 - Math.min(normalized / maxExpected, 1.0);
  return Math.max(0, Math.min(1, score));
}

// ─── Precomputed dissonance for triads ──────────────────────────────

export interface TriadDissonance {
  bits: number;
  roughness: number;
  consonance: number;
}

// Compute dissonance for all 24 major/minor triads in a reference voicing
export function precomputeTriadDissonance(
  profile: OvertoneProfile,
  baseOctave: number = 4,
): Map<number, TriadDissonance> {
  const results = new Map<number, TriadDissonance>();

  for (let root = 0; root < 12; root++) {
    for (const isMajor of [true, false]) {
      const third = isMajor ? 4 : 3;
      const pcs = [root, (root + third) % 12, (root + 7) % 12];
      const bits = pcs.reduce((b, pc) => b | (1 << pc), 0);

      // Close voicing in baseOctave
      const baseMidi = root + (baseOctave + 1) * 12;
      const midi = [baseMidi, baseMidi + third, baseMidi + 7];

      const roughness = chordRoughness(midi, profile);
      const consonance = consonanceScore(midi, profile);

      results.set(bits, { bits, roughness, consonance });
    }
  }
  return results;
}
