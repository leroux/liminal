// ─── Voice leading & neo-Riemannian navigation ─────────────────────

import { bitsToArray, popcount, ALL_TRIADS, type Triad } from './core.js';
import { consonanceScore, type OvertoneProfile } from './dissonance.js';

// ─── Voice leading distance ─────────────────────────────────────────

// Circular pitch-class distance
function pcDist(a: number, b: number): number {
  const d = Math.abs(a - b) % 12;
  return Math.min(d, 12 - d);
}

// Minimum voice leading distance (L1) between two equal-cardinality PCS arrays
export function minVLD(a: number[], b: number[]): number {
  const n = a.length;
  if (n !== b.length) return Infinity;
  const sa = [...a].sort((x, y) => x - y);
  const sb = [...b].sort((x, y) => x - y);

  let minDist = Infinity;
  for (let r = 0; r < n; r++) {
    let dist = 0;
    for (let i = 0; i < n; i++) {
      dist += pcDist(sa[i], sb[(i + r) % n]);
    }
    minDist = Math.min(minDist, dist);
  }
  return minDist;
}

// VLD between two PCS bit patterns
export function vldBits(a: number, b: number): number {
  return minVLD(bitsToArray(a), bitsToArray(b));
}

// ─── Neo-Riemannian transformations ─────────────────────────────────

export interface PLRResult {
  root: number;
  isMajor: boolean;
  op: string;
}

export function applyP(root: number, isMajor: boolean): PLRResult {
  return { root, isMajor: !isMajor, op: 'P' };
}

export function applyL(root: number, isMajor: boolean): PLRResult {
  if (isMajor) {
    return { root: (root + 4) % 12, isMajor: false, op: 'L' };
  } else {
    return { root: (root + 8) % 12, isMajor: true, op: 'L' };
  }
}

export function applyR(root: number, isMajor: boolean): PLRResult {
  if (isMajor) {
    return { root: (root + 9) % 12, isMajor: false, op: 'R' };
  } else {
    return { root: (root + 3) % 12, isMajor: true, op: 'R' };
  }
}

// All single-step PLR neighbors of a triad
export function plrNeighbors(root: number, isMajor: boolean): PLRResult[] {
  return [
    applyP(root, isMajor),
    applyL(root, isMajor),
    applyR(root, isMajor),
  ];
}

// Compound operations
export function applyLP(root: number, isMajor: boolean): PLRResult {
  const l = applyL(root, isMajor);
  const p = applyP(l.root, l.isMajor);
  return { ...p, op: 'LP' };
}

export function applyPL(root: number, isMajor: boolean): PLRResult {
  const p = applyP(root, isMajor);
  const l = applyL(p.root, p.isMajor);
  return { ...l, op: 'PL' };
}

export function applyRP(root: number, isMajor: boolean): PLRResult {
  const r = applyR(root, isMajor);
  const p = applyP(r.root, r.isMajor);
  return { ...p, op: 'RP' };
}

export function applyPR(root: number, isMajor: boolean): PLRResult {
  const p = applyP(root, isMajor);
  const r = applyR(p.root, p.isMajor);
  return { ...r, op: 'PR' };
}

// ─── Pre-computed distance matrix ───────────────────────────────────

export interface DistanceMatrix {
  indices: Map<number, number>;  // bits → row/col index
  bitsList: number[];            // index → bits
  labels: string[];              // index → label
  data: Float32Array;            // M×M flattened
  size: number;
}

export function buildDistanceMatrix(
  chords: { bits: number; label: string }[],
): DistanceMatrix {
  // Group by cardinality for meaningful VLD — only compare same cardinality
  const size = chords.length;
  const data = new Float32Array(size * size);
  const indices = new Map<number, number>();
  const bitsList: number[] = [];
  const labels: string[] = [];

  for (let i = 0; i < size; i++) {
    indices.set(chords[i].bits, i);
    bitsList.push(chords[i].bits);
    labels.push(chords[i].label);
  }

  // Compute pairwise VLD (symmetric, so only compute upper triangle)
  for (let i = 0; i < size; i++) {
    const cardI = popcount(bitsList[i]);
    for (let j = i + 1; j < size; j++) {
      const cardJ = popcount(bitsList[j]);
      // Cross-cardinality gets Infinity
      const d = cardI === cardJ ? vldBits(bitsList[i], bitsList[j]) : 100;
      data[i * size + j] = d;
      data[j * size + i] = d;
    }
  }

  return { indices, bitsList, labels, data, size };
}

// Fast nearest-neighbor lookup from pre-computed matrix
export function findNearestFromMatrix(
  matrix: DistanceMatrix,
  sourceBits: number,
  maxResults: number = 10,
): { bits: number; label: string; distance: number; commonTones: number }[] {
  const idx = matrix.indices.get(sourceBits);
  if (idx === undefined) return [];

  const { size, data, bitsList, labels } = matrix;
  const row = idx * size;

  // Collect all distances from this row
  const results: { bits: number; label: string; distance: number; commonTones: number }[] = [];
  for (let j = 0; j < size; j++) {
    if (j === idx) continue;
    const d = data[row + j];
    if (d >= 100) continue; // skip cross-cardinality
    results.push({
      bits: bitsList[j],
      label: labels[j],
      distance: d,
      commonTones: popcount(sourceBits & bitsList[j]),
    });
  }

  results.sort((a, b) => a.distance - b.distance);
  return results.slice(0, maxResults);
}

// ─── Parsimonious neighbor enumeration ──────────────────────────────
// Generate all chords reachable by moving m voices by 1 semitone

export function parsimoniousNeighbors(
  bits: number,
  maxMoves: number = 1,
): { bits: number; moves: number }[] {
  const pcs = bitsToArray(bits);
  const n = pcs.length;
  const results = new Map<number, number>(); // bits → number of moves

  // Generate all combinations of voices to move
  function enumerate(voiceIdx: number, movesLeft: number, currentPcs: number[]): void {
    if (movesLeft === 0 || voiceIdx >= n) {
      const newBits = currentPcs.reduce((b, pc) => b | (1 << pc), 0);
      if (newBits !== bits && popcount(newBits) === n) {
        const totalMoves = maxMoves - movesLeft;
        const existing = results.get(newBits);
        if (existing === undefined || totalMoves < existing) {
          results.set(newBits, totalMoves);
        }
      }
      // Continue with remaining voices even if no moves left
      if (voiceIdx < n) {
        enumerate(voiceIdx + 1, movesLeft, currentPcs);
      }
      return;
    }

    // Option 1: don't move this voice
    enumerate(voiceIdx + 1, movesLeft, currentPcs);

    // Option 2: move up by semitone
    const up = (currentPcs[voiceIdx] + 1) % 12;
    if (!currentPcs.includes(up)) {
      const copy = [...currentPcs];
      copy[voiceIdx] = up;
      enumerate(voiceIdx + 1, movesLeft - 1, copy);
    }

    // Option 3: move down by semitone
    const down = (currentPcs[voiceIdx] + 11) % 12;
    if (!currentPcs.includes(down)) {
      const copy = [...currentPcs];
      copy[voiceIdx] = down;
      enumerate(voiceIdx + 1, movesLeft - 1, copy);
    }
  }

  enumerate(0, maxMoves, [...pcs]);

  return Array.from(results.entries())
    .map(([b, m]) => ({ bits: b, moves: m }))
    .sort((a, b) => a.moves - b.moves);
}

// ─── Combined ranking score ─────────────────────────────────────────

export interface RankedNeighbor {
  bits: number;
  label: string;
  score: number;       // lower = better neighbor
  vld: number;
  commonTones: number;
  consonance: number;
  isPLR: boolean;
}

export function combinedRanking(
  sourceBits: number,
  candidates: { bits: number; label: string }[],
  profile: OvertoneProfile,
  weights: {
    vld?: number;           // default 1.0
    commonTones?: number;   // default -0.5 (rewards common tones)
    consonance?: number;    // default -0.3 (rewards consonance)
    plrBonus?: number;      // default -0.8 (rewards PLR relation)
  } = {},
): RankedNeighbor[] {
  const w = {
    vld: weights.vld ?? 1.0,
    commonTones: weights.commonTones ?? -0.5,
    consonance: weights.consonance ?? -0.3,
    plrBonus: weights.plrBonus ?? -0.8,
  };

  const sourceCard = popcount(sourceBits);

  // Build PLR set for triads
  const plrSet = new Set<number>();
  if (sourceCard === 3) {
    // Check all PLR results for all 24 triads
    for (const triad of ALL_TRIADS) {
      if (triad.bits === sourceBits) {
        const neighbors = plrNeighbors(triad.root, triad.isMajor);
        for (const n of neighbors) {
          const third = n.isMajor ? 4 : 3;
          const nBits = (1 << n.root) | (1 << ((n.root + third) % 12)) | (1 << ((n.root + 7) % 12));
          plrSet.add(nBits);
        }
      }
    }
  }

  const results: RankedNeighbor[] = [];

  for (const cand of candidates) {
    if (cand.bits === sourceBits) continue;
    const candCard = popcount(cand.bits);
    if (candCard !== sourceCard) continue;

    const vld = vldBits(sourceBits, cand.bits);
    const ct = popcount(sourceBits & cand.bits);
    const midi = bitsToArray(cand.bits).map(pc => pc + 60); // C4 voicing for scoring
    const cons = consonanceScore(midi, profile);
    const isPLR = plrSet.has(cand.bits);

    const score =
      w.vld * vld +
      w.commonTones * ct +
      w.consonance * cons +
      (isPLR ? w.plrBonus : 0);

    results.push({
      bits: cand.bits,
      label: cand.label,
      score,
      vld,
      commonTones: ct,
      consonance: cons,
      isPLR,
    });
  }

  results.sort((a, b) => a.score - b.score);
  return results;
}

// ─── Classical MDS for 2D layout from distance matrix ───────────────

export function classicalMDS(matrix: DistanceMatrix): { x: Float32Array; y: Float32Array } {
  const n = matrix.size;
  if (n === 0) return { x: new Float32Array(0), y: new Float32Array(0) };
  if (n === 1) return { x: new Float32Array([0]), y: new Float32Array([0]) };

  const D = matrix.data;

  // Step 1: Compute D² element-wise
  const D2 = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) {
    D2[i] = D[i] * D[i];
  }

  // Step 2: Double centering → B = -0.5 * H * D² * H where H = I - (1/n) * 11'
  // B_ij = -0.5 * (D²_ij - rowMean_i - colMean_j + grandMean)
  const rowMeans = new Float64Array(n);
  const colMeans = new Float64Array(n);
  let grandMean = 0;

  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += D2[i * n + j];
    rowMeans[i] = sum / n;
  }
  for (let j = 0; j < n; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += D2[i * n + j];
    colMeans[j] = sum / n;
  }
  for (let i = 0; i < n; i++) grandMean += rowMeans[i];
  grandMean /= n;

  const B = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      B[i * n + j] = -0.5 * (D2[i * n + j] - rowMeans[i] - colMeans[j] + grandMean);
    }
  }

  // Step 3: Power iteration for top 2 eigenvectors
  const x = new Float32Array(n);
  const y = new Float32Array(n);

  function powerIteration(mat: Float64Array, sz: number, deflateVec?: Float64Array): { vec: Float64Array; val: number } {
    let v = new Float64Array(sz);
    for (let i = 0; i < sz; i++) v[i] = Math.random() - 0.5;

    let eigenvalue = 0;
    for (let iter = 0; iter < 100; iter++) {
      // Multiply: w = B * v
      const w = new Float64Array(sz);
      for (let i = 0; i < sz; i++) {
        let sum = 0;
        for (let j = 0; j < sz; j++) sum += mat[i * sz + j] * v[j];
        w[i] = sum;
      }

      // Deflate if needed
      if (deflateVec) {
        let dot = 0;
        for (let i = 0; i < sz; i++) dot += w[i] * deflateVec[i];
        for (let i = 0; i < sz; i++) w[i] -= dot * deflateVec[i];
      }

      // Normalize
      let norm = 0;
      for (let i = 0; i < sz; i++) norm += w[i] * w[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-10) break;

      eigenvalue = norm;
      for (let i = 0; i < sz; i++) v[i] = w[i] / norm;
    }

    return { vec: v, val: eigenvalue };
  }

  const { vec: v1, val: e1 } = powerIteration(B, n);
  const { vec: v2, val: e2 } = powerIteration(B, n, v1);

  const s1 = Math.sqrt(Math.max(0, e1));
  const s2 = Math.sqrt(Math.max(0, e2));

  for (let i = 0; i < n; i++) {
    x[i] = v1[i] * s1;
    y[i] = v2[i] * s2;
  }

  return { x, y };
}

// ─── Find nearest chords by VLD from a pool ─────────────────────────

export interface VLDNeighbor {
  bits: number;
  label: string;
  distance: number;
  commonTones: number;
}

export function findNearestByVLD(
  sourceBits: number,
  pool: { bits: number; label: string }[],
  maxResults: number = 8,
): VLDNeighbor[] {
  const sourceCard = popcount(sourceBits);
  const results: VLDNeighbor[] = [];

  for (const item of pool) {
    if (item.bits === sourceBits) continue;
    const itemCard = popcount(item.bits);
    if (itemCard !== sourceCard) continue;
    const dist = vldBits(sourceBits, item.bits);
    const ct = popcount(sourceBits & item.bits);
    results.push({ bits: item.bits, label: item.label, distance: dist, commonTones: ct });
  }

  results.sort((a, b) => a.distance - b.distance);
  return results.slice(0, maxResults);
}

// ─── Distance matrix for all 24 triads ──────────────────────────────

export interface TriadDistance {
  triad: Triad;
  distance: number;
  commonTones: number;
  plrOp: string | null;
}

export function triadDistances(source: Triad): TriadDistance[] {
  const neighbors = plrNeighbors(source.root, source.isMajor);
  const plrMap = new Map<string, string>();
  for (const n of neighbors) {
    const key = `${n.root}-${n.isMajor}`;
    plrMap.set(key, n.op);
  }

  return ALL_TRIADS.map(t => {
    const key = `${t.root}-${t.isMajor}`;
    return {
      triad: t,
      distance: vldBits(source.bits, t.bits),
      commonTones: popcount(source.bits & t.bits),
      plrOp: plrMap.get(key) || null,
    };
  }).sort((a, b) => a.distance - b.distance);
}
