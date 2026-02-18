// ─── Multi-method polyphonic synthesizer ─────────────────────────────
// Methods: additive, wavetable, subtractive, FM
// Effects: chorus, reverb send
// Proper ADSR envelope

import { midiToFreq } from './core.js';

// ─── Just intonation ────────────────────────────────────────────────
// 5-limit ratios for each semitone above a reference note.
// The bass note stays 12-TET; upper voices tune to pure intervals above it.

const JI_RATIOS = [
  1 / 1,   // 0  unison
  16 / 15, // 1  minor 2nd
  9 / 8,   // 2  major 2nd
  6 / 5,   // 3  minor 3rd
  5 / 4,   // 4  major 3rd
  4 / 3,   // 5  perfect 4th
  45 / 32, // 6  tritone
  3 / 2,   // 7  perfect 5th
  8 / 5,   // 8  minor 6th
  5 / 3,   // 9  major 6th
  9 / 5,   // 10 minor 7th
  15 / 8,  // 11 major 7th
];

function midiToJustFreq(midi: number, bassMidi: number): number {
  const bassFreq = midiToFreq(bassMidi);
  const diff = midi - bassMidi;
  const octaves = Math.floor(diff / 12);
  const semitone = ((diff % 12) + 12) % 12;
  return bassFreq * JI_RATIOS[semitone] * Math.pow(2, octaves);
}

export type Tuning = 'tet' | 'just';
import { type OvertoneProfile, TIMBRES, DEFAULT_TIMBRE } from './dissonance.js';

// ─── ADSR envelope ──────────────────────────────────────────────────

export interface ADSRParams {
  attack: number;   // seconds
  decay: number;    // seconds
  sustain: number;  // 0–1
  release: number;  // seconds
}

const DEFAULT_ADSR: ADSRParams = {
  attack: 0.02,
  decay: 0.1,
  sustain: 0.7,
  release: 0.3,
};

// ─── Synthesis methods ──────────────────────────────────────────────

export type SynthMethod = 'additive' | 'wavetable' | 'subtractive' | 'fm';

// ─── Macro controls ─────────────────────────────────────────────────

export interface MacroParams {
  brightness: number;  // 0–1, controls filter cutoff / partial rolloff
  warmth: number;      // 0–1, gentle LPF + even harmonic boost
  attack: number;      // 0–1, envelope attack time
  body: number;        // 0–1, mid-partial boost
  air: number;         // 0–1, HF emphasis + reverb send
}

const DEFAULT_MACROS: MacroParams = {
  brightness: 0.5,
  warmth: 0.5,
  attack: 0.3,
  body: 0.5,
  air: 0.3,
};

// ─── Multi-method synth ─────────────────────────────────────────────

interface ActiveVoice {
  nodes: AudioNode[];
  masterGain: GainNode;
}

export class MultiSynth {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private activeVoices: ActiveVoice[] = [];

  // Effects chain nodes
  private chorusNode: DelayNode | null = null;
  private chorusLFO: OscillatorNode | null = null;
  private chorusGain: GainNode | null = null;
  private reverbNode: ConvolverNode | null = null;
  private reverbGain: GainNode | null = null;
  private dryGain: GainNode | null = null;

  private _timbre: OvertoneProfile = TIMBRES[DEFAULT_TIMBRE];
  private _method: SynthMethod = 'additive';
  private _tuning: Tuning = 'tet';
  private _adsr: ADSRParams = { ...DEFAULT_ADSR };
  private _macros: MacroParams = { ...DEFAULT_MACROS };

  get timbre(): OvertoneProfile { return this._timbre; }
  set timbre(t: OvertoneProfile) { this._timbre = t; }

  get method(): SynthMethod { return this._method; }
  set method(m: SynthMethod) { this._method = m; }

  get tuning(): Tuning { return this._tuning; }
  set tuning(t: Tuning) { this._tuning = t; }

  get adsr(): ADSRParams { return this._adsr; }
  get macros(): MacroParams { return this._macros; }

  setMacro(key: keyof MacroParams, value: number): void {
    this._macros[key] = Math.max(0, Math.min(1, value));
    this.applyMacros();
  }

  private applyMacros(): void {
    // Map attack macro to ADSR
    this._adsr.attack = 0.005 + this._macros.attack * 2.0; // 5ms to 2s

    // Map air to reverb send
    if (this.reverbGain) {
      this.reverbGain.gain.value = this._macros.air * 0.5;
    }
    if (this.dryGain) {
      this.dryGain.gain.value = 1.0 - this._macros.air * 0.2;
    }
  }

  private ensureContext(): AudioContext {
    if (!this.ctx) {
      this.ctx = new AudioContext({ sampleRate: 48000 });
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.3;

      // Build effects chain
      this.buildEffectsChain();

      this.masterGain.connect(this.ctx.destination);
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
    return this.ctx;
  }

  private buildEffectsChain(): void {
    const ctx = this.ctx!;

    // Dry path
    this.dryGain = ctx.createGain();
    this.dryGain.gain.value = 0.85;
    this.dryGain.connect(this.masterGain!);

    // Chorus: modulated delay line
    this.chorusNode = ctx.createDelay(0.05);
    this.chorusNode.delayTime.value = 0.012;
    this.chorusGain = ctx.createGain();
    this.chorusGain.gain.value = 0.3;

    this.chorusLFO = ctx.createOscillator();
    this.chorusLFO.frequency.value = 0.5;
    const chorusDepth = ctx.createGain();
    chorusDepth.gain.value = 0.003; // 3ms modulation depth
    this.chorusLFO.connect(chorusDepth);
    chorusDepth.connect(this.chorusNode.delayTime);
    this.chorusLFO.start();

    this.chorusNode.connect(this.chorusGain);
    this.chorusGain.connect(this.masterGain!);

    // Reverb: simple convolution with generated impulse
    this.reverbNode = ctx.createConvolver();
    this.reverbGain = ctx.createGain();
    this.reverbGain.gain.value = 0.15;

    // Generate simple reverb impulse
    const sampleRate = ctx.sampleRate;
    const length = sampleRate * 1.5; // 1.5 second tail
    const impulse = ctx.createBuffer(2, length, sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const data = impulse.getChannelData(ch);
      for (let i = 0; i < length; i++) {
        data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (sampleRate * 0.4));
      }
    }
    this.reverbNode.buffer = impulse;

    this.reverbNode.connect(this.reverbGain);
    this.reverbGain.connect(this.masterGain!);
  }

  // Get the effect input node (voices connect here)
  private getEffectInput(): AudioNode {
    return this.dryGain!;
  }

  playChord(midiNotes: number[], duration: number = 2.0): void {
    const ctx = this.ensureContext();
    this.stopAll();

    const numNotes = midiNotes.length;
    const noteGainValue = 0.6 / Math.sqrt(numNotes);
    const bassMidi = Math.min(...midiNotes);

    for (const midi of midiNotes) {
      const freq = this._tuning === 'just'
        ? midiToJustFreq(midi, bassMidi)
        : midiToFreq(midi);

      let voice: ActiveVoice;

      switch (this._method) {
        case 'wavetable':
          voice = this.createWavetableVoice(ctx, freq, noteGainValue, duration);
          break;
        case 'subtractive':
          voice = this.createSubtractiveVoice(ctx, freq, noteGainValue, duration);
          break;
        case 'fm':
          voice = this.createFMVoice(ctx, freq, noteGainValue, duration);
          break;
        default:
          voice = this.createAdditiveVoice(ctx, freq, noteGainValue, duration);
      }

      this.activeVoices.push(voice);
    }

    // Also send to chorus and reverb
    setTimeout(() => {
      this.activeVoices = this.activeVoices.filter(v => {
        try { v.masterGain.gain.value; return true; } catch { return false; }
      });
    }, (duration + 0.5) * 1000);
  }

  // ─── Additive synthesis ─────────────────────────────────────────

  private createAdditiveVoice(ctx: AudioContext, f0: number, gain: number, dur: number): ActiveVoice {
    const voice: ActiveVoice = {
      nodes: [],
      masterGain: ctx.createGain(),
    };

    this.applyEnvelope(voice.masterGain, gain, dur, ctx.currentTime);
    voice.masterGain.connect(this.getEffectInput());
    voice.masterGain.connect(this.chorusNode!);
    voice.masterGain.connect(this.reverbNode!);

    // Apply brightness as partial rolloff
    const brightnessRolloff = 0.5 + this._macros.brightness * 1.5;

    for (const [ratio, amp] of this._timbre.partials) {
      const freq = f0 * ratio;
      if (freq > 18000 || freq < 20) continue;

      const osc = ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.value = freq;

      const g = ctx.createGain();
      // Apply brightness rolloff to higher partials
      const rolloff = Math.pow(1 / ratio, 1.0 / brightnessRolloff);
      g.gain.value = amp * rolloff;

      // Apply warmth: boost even harmonics
      if (this._macros.warmth > 0.5 && Math.round(ratio) % 2 === 0) {
        g.gain.value *= 1 + (this._macros.warmth - 0.5) * 0.6;
      }

      // Apply body: boost mid partials (2nd-5th)
      if (ratio >= 2 && ratio <= 5) {
        g.gain.value *= 1 + (this._macros.body - 0.5) * 0.4;
      }

      osc.connect(g);
      g.connect(voice.masterGain);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + dur + 0.5);
      voice.nodes.push(osc, g);
    }

    return voice;
  }

  // ─── Wavetable synthesis ────────────────────────────────────────

  private createWavetableVoice(ctx: AudioContext, f0: number, gain: number, dur: number): ActiveVoice {
    const voice: ActiveVoice = {
      nodes: [],
      masterGain: ctx.createGain(),
    };

    this.applyEnvelope(voice.masterGain, gain, dur, ctx.currentTime);
    voice.masterGain.connect(this.getEffectInput());
    voice.masterGain.connect(this.chorusNode!);
    voice.masterGain.connect(this.reverbNode!);

    // Build custom PeriodicWave from timbre partials
    const maxHarmonic = Math.floor(18000 / f0);
    const real = new Float32Array(maxHarmonic + 1);
    const imag = new Float32Array(maxHarmonic + 1);
    real[0] = 0; imag[0] = 0;

    const brightnessRolloff = 0.5 + this._macros.brightness * 1.5;

    for (const [ratio, amp] of this._timbre.partials) {
      const harmIdx = Math.round(ratio);
      if (harmIdx > 0 && harmIdx <= maxHarmonic) {
        const rolloff = Math.pow(1 / Math.max(ratio, 1), 1.0 / brightnessRolloff);
        imag[harmIdx] = amp * rolloff;
      }
    }

    const wave = ctx.createPeriodicWave(real, imag, { disableNormalization: false });
    const osc = ctx.createOscillator();
    osc.setPeriodicWave(wave);
    osc.frequency.value = f0;

    // Slight detune for warmth
    if (this._macros.warmth > 0.5) {
      const detune = (this._macros.warmth - 0.5) * 8; // up to 4 cents
      osc.detune.value = detune;
    }

    osc.connect(voice.masterGain);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + dur + 0.5);
    voice.nodes.push(osc);

    return voice;
  }

  // ─── Subtractive synthesis ──────────────────────────────────────

  private createSubtractiveVoice(ctx: AudioContext, f0: number, gain: number, dur: number): ActiveVoice {
    const voice: ActiveVoice = {
      nodes: [],
      masterGain: ctx.createGain(),
    };

    this.applyEnvelope(voice.masterGain, gain, dur, ctx.currentTime);
    voice.masterGain.connect(this.getEffectInput());
    voice.masterGain.connect(this.chorusNode!);
    voice.masterGain.connect(this.reverbNode!);

    // Saw oscillator
    const osc = ctx.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.value = f0;

    // Second oscillator slightly detuned for thickness
    const osc2 = ctx.createOscillator();
    osc2.type = 'sawtooth';
    osc2.frequency.value = f0;
    osc2.detune.value = 7 + this._macros.warmth * 10; // 7-17 cents

    const oscGain = ctx.createGain();
    oscGain.gain.value = 0.5;
    const osc2Gain = ctx.createGain();
    osc2Gain.gain.value = 0.35;

    // Low-pass filter
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    // Cutoff controlled by brightness
    const baseCutoff = f0 * 2;
    const maxCutoff = Math.min(f0 * 12, 16000);
    filter.frequency.value = baseCutoff + this._macros.brightness * (maxCutoff - baseCutoff);
    filter.Q.value = 0.5 + this._macros.body * 4; // resonance from body

    // Filter envelope: sweep cutoff down
    const now = ctx.currentTime;
    const filterAttack = 0.01 + this._macros.attack * 0.5;
    filter.frequency.setValueAtTime(filter.frequency.value * 2, now);
    filter.frequency.exponentialRampToValueAtTime(filter.frequency.value, now + filterAttack);

    osc.connect(oscGain);
    osc2.connect(osc2Gain);
    oscGain.connect(filter);
    osc2Gain.connect(filter);
    filter.connect(voice.masterGain);

    osc.start(now);
    osc2.start(now);
    osc.stop(now + dur + 0.5);
    osc2.stop(now + dur + 0.5);

    voice.nodes.push(osc, osc2, oscGain, osc2Gain, filter);

    return voice;
  }

  // ─── FM synthesis ───────────────────────────────────────────────

  private createFMVoice(ctx: AudioContext, f0: number, gain: number, dur: number): ActiveVoice {
    const voice: ActiveVoice = {
      nodes: [],
      masterGain: ctx.createGain(),
    };

    this.applyEnvelope(voice.masterGain, gain, dur, ctx.currentTime);
    voice.masterGain.connect(this.getEffectInput());
    voice.masterGain.connect(this.chorusNode!);
    voice.masterGain.connect(this.reverbNode!);

    // 2-operator FM: carrier + modulator
    const carrier = ctx.createOscillator();
    carrier.type = 'sine';
    carrier.frequency.value = f0;

    // Modulator
    const modulator = ctx.createOscillator();
    modulator.type = 'sine';
    // C:M ratio based on timbre — use 1:1 for bell-like, 1:2 for piano-like
    const cmRatio = this._timbre.partials.length > 6 ? 1.0 : 2.0;
    modulator.frequency.value = f0 * cmRatio;

    // Modulation depth (index) controlled by brightness
    const modIndex = 0.5 + this._macros.brightness * 5;
    const modGain = ctx.createGain();
    modGain.gain.value = f0 * modIndex;

    // FM envelope: index decays over time (like real instruments)
    const now = ctx.currentTime;
    modGain.gain.setValueAtTime(f0 * modIndex * 2, now);
    modGain.gain.exponentialRampToValueAtTime(f0 * modIndex * 0.3, now + dur * 0.7);

    modulator.connect(modGain);
    modGain.connect(carrier.frequency);
    carrier.connect(voice.masterGain);

    carrier.start(now);
    modulator.start(now);
    carrier.stop(now + dur + 0.5);
    modulator.stop(now + dur + 0.5);

    voice.nodes.push(carrier, modulator, modGain);

    return voice;
  }

  // ─── ADSR envelope ──────────────────────────────────────────────

  private applyEnvelope(gainNode: GainNode, peakGain: number, duration: number, startTime: number): void {
    const { attack, decay, sustain, release } = this._adsr;
    const sustainGain = peakGain * sustain;
    const releaseStart = startTime + duration - release;

    gainNode.gain.setValueAtTime(0, startTime);
    gainNode.gain.linearRampToValueAtTime(peakGain, startTime + attack);
    gainNode.gain.linearRampToValueAtTime(sustainGain, startTime + attack + decay);
    gainNode.gain.setValueAtTime(sustainGain, Math.max(releaseStart, startTime + attack + decay));
    gainNode.gain.linearRampToValueAtTime(0, startTime + duration);
  }

  stopAll(): void {
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    for (const voice of this.activeVoices) {
      voice.masterGain.gain.cancelScheduledValues(now);
      voice.masterGain.gain.setValueAtTime(voice.masterGain.gain.value, now);
      voice.masterGain.gain.linearRampToValueAtTime(0, now + 0.03);
      for (const node of voice.nodes) {
        if (node instanceof OscillatorNode) {
          try { node.stop(now + 0.05); } catch { /* already stopped */ }
        }
      }
    }
    this.activeVoices = [];
  }

  // ─── Analyser node for visualizations ─────────────────────────

  getAnalyserNode(): AnalyserNode | null {
    if (!this.ctx) return null;
    const analyser = this.ctx.createAnalyser();
    analyser.fftSize = 2048;
    this.masterGain!.connect(analyser);
    return analyser;
  }

  getAudioContext(): AudioContext | null {
    return this.ctx;
  }
}
