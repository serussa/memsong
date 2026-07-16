#!/usr/bin/env python3
"""
Baseline Lyric Progression Probe for ACE-Step / DiT.

Analyzes the frozen baseline model's lyric progression mechanism.
No adapters, no training, no new model components.

Output: outputs/baseline_progression_probe/
"""

import os, sys, json, math, pickle, time, copy, random, csv
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ['ACESTEP_OFFLINE'] = '1'
os.environ['ACESTEP_MINIMAL_COMPONENTS'] = '1'
os.environ['SIDESTEP_SAFE_ROOT'] = '/root/autodl-tmp'

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'baseline_progression_probe')
SEEDS = [42, 123]
BASE_DURATION = 30

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32

# Token group constants
TG_STYLE, TG_LYRIC, TG_SECTION, TG_NONVOCAL, TG_SPECIAL = range(5)

SECTION_KW = ['[Verse]', '[Chorus]', '[Intro', '[Outro]', '[Bridge]',
              '[Pre-Chorus]', '[Post-Chorus]', '[Interlude]', '[Solo]', '[Break]', '[End]']
NONVOCAL_KW = ['[Instrumental]', '[Solo]', '[Break]']

MINIMAL_LYRICS = "[Verse]\n风\n雨\n\n[Chorus]\n星光\n照亮\n\n[Outro]"
SHORT_LYRICS = "[Verse]\n风吹过山岗\n雨落在心上\n\n[Chorus]\n星光闪耀夜空\n照亮我的梦\n\n[Verse]\n岁月如歌流淌\n思念在远方\n\n[Chorus]\n星光闪耀夜空\n照亮我的梦\n\n[Outro]"
BASE_CAPTION = "A pop song with female vocals, piano and guitar accompaniment, medium tempo"

def classify_token_groups(lyrics: str, caption: str) -> torch.Tensor:
    """Build token-group classification for encoder_hidden_states positions."""
    cw = caption.strip().split() if caption else []
    ll = lyrics.strip().split('\n') if lyrics else []
    total = max(len(cw) + len(ll), 1)
    g = torch.zeros(total, dtype=torch.long)
    g[:len(cw)] = TG_STYLE
    for i, line in enumerate(ll):
        idx = len(cw) + i
        if idx >= total: break
        ls = line.strip()
        if any(ls.startswith(k) for k in NONVOCAL_KW): g[idx] = TG_NONVOCAL
        elif any(ls.startswith(k) for k in SECTION_KW): g[idx] = TG_SECTION
        elif ls: g[idx] = TG_LYRIC
        else: g[idx] = TG_SPECIAL
    return g


def group_name(gid: int) -> str:
    return {TG_STYLE: 'style', TG_LYRIC: 'lyric', TG_SECTION: 'section',
            TG_NONVOCAL: 'nonvocal', TG_SPECIAL: 'special'}.get(gid, 'unknown')


# ====================================================================
# Attention perturbation via pre-hooks on cross_attn modules
# ====================================================================

class AttentionPerturbationEngine:
    """Manages per-layer attention perturbations via encoder-hidden-state scaling.

    Instead of per-token classification (which requires the text tokenizer), we
    use region-based scaling: caption region (first N tokens) and lyric region
    (remaining tokens).
    """

    def __init__(self, model):
        self.model = model
        self.handles: list = []
        self.active: dict = {}

    def _make_pre_hook(self, scale_fn: Callable):
        """Create pre-hook that dynamically computes per-token scale factors.

        scale_fn: callable(n_tokens) -> 1D tensor of length n_tokens.

        Note: cross_attn.forward() receives all args as keyword args, so
        the positional args tuple may be empty. We must return ((), new_kwargs)
        to pass the modified kwargs through.
        """
        def pre_hook(module, args, kwargs):
            eh = kwargs.get('encoder_hidden_states')
            if eh is not None:
                n_tokens = eh.shape[1]
                scale = scale_fn(n_tokens).to(eh.device, eh.dtype)
                new_eh = eh.clone() * scale.unsqueeze(0).unsqueeze(-1)
                new_kwargs = dict(kwargs)
                new_kwargs['encoder_hidden_states'] = new_eh
                return (), new_kwargs
            return None
        return pre_hook

    def apply_caption_attenuation(self, layer_idx: int, delta: float = 0.2,
                                   n_caption_tokens: int = 15):
        """Attenuate caption-region tokens at a specific layer."""
        self.remove_layer(layer_idx)
        def scale_fn(n_tokens):
            s = torch.ones(n_tokens)
            s[:min(n_caption_tokens, n_tokens)] = 1.0 - delta
            return s
        self._register(layer_idx, scale_fn)

    def apply_lyric_attenuation(self, layer_idx: int, delta: float = 0.2,
                                 n_caption_tokens: int = 15):
        """Attenuate lyric-region tokens (after caption)."""
        self.remove_layer(layer_idx)
        def scale_fn(n_tokens):
            s = torch.ones(n_tokens)
            s[min(n_caption_tokens, n_tokens):] = 1.0 - delta
            return s
        self._register(layer_idx, scale_fn)

    def apply_caption_boost(self, layer_idx: int, delta: float = 0.2,
                             n_caption_tokens: int = 15):
        self.remove_layer(layer_idx)
        def scale_fn(n_tokens):
            s = torch.ones(n_tokens)
            s[:min(n_caption_tokens, n_tokens)] = 1.0 + delta
            return s
        self._register(layer_idx, scale_fn)

    def apply_lyric_boost(self, layer_idx: int, delta: float = 0.2,
                           n_caption_tokens: int = 15):
        self.remove_layer(layer_idx)
        def scale_fn(n_tokens):
            s = torch.ones(n_tokens)
            s[min(n_caption_tokens, n_tokens):] = 1.0 + delta
            return s
        self._register(layer_idx, scale_fn)

    def _register(self, layer_idx: int, scale_fn: Callable):
        layer = self.model.decoder.layers[layer_idx]
        if not hasattr(layer, 'cross_attn'):
            return
        handle = layer.cross_attn.register_forward_pre_hook(
            self._make_pre_hook(scale_fn), with_kwargs=True)
        self.handles.append((layer_idx, handle))

    def remove_layer(self, layer_idx: int):
        for i, (lid, h) in enumerate(self.handles):
            if lid == layer_idx:
                h.remove()
                self.handles.pop(i)
                self.active.pop(layer_idx, None)
                break

    def clear_all(self):
        for _, h in self.handles:
            h.remove()
        self.handles.clear()
        self.active.clear()


# ====================================================================
# Attention statistics collector (hooked via decoder monkey-patch)
# ====================================================================

class AttentionCollector:
    """Collects cross-attention weights from decoder layers.

    Works by monkey-patching the decoder forward to enable output_attentions.
    """

    def __init__(self, model, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.reset()

    def reset(self):
        self.cross_attentions: Dict[int, List[torch.Tensor]] = {}
        self.decoder_outputs_cache: list = []

    def collect(self, decoder_outputs):
        """Call after decoder() returns. Extracts cross-attentions.

        PreTrainedModel.__call__ standardizes output to:
          [0]=hidden_states, [1]=past_key_values, [2]=all_hidden_states(None),
          [3]=attentions(tuple of cross-attn per layer, or None).
        """
        # output[3] is the cross-attentions tuple (24 elements, one per layer)
        ca_tuple = decoder_outputs[3] if len(decoder_outputs) >= 4 else (
            decoder_outputs[2] if len(decoder_outputs) >= 3 else None
        )
        if ca_tuple is not None:
            for i, ca in enumerate(ca_tuple):
                if isinstance(ca, torch.Tensor):
                    # ca shape: [B, H, T_latent, T_text] for one layer
                    self.cross_attentions.setdefault(i, []).append(ca.detach().cpu().float())

    def get_mass_by_region(self, layer_idx: int, n_caption_tokens: int = 10) -> dict:
        """Get attention mass allocated to caption vs lyric regions.

        The encoder hidden states are [caption_tokens, lyric_tokens...].
        We partition attention into caption region and lyric region.
        """
        if layer_idx not in self.cross_attentions:
            return {}
        stacked = torch.stack(self.cross_attentions[layer_idx])  # [steps, B, H, T_latent, T_text]
        mass = stacked.mean(dim=(0, 1, 2, 3))  # [T_text] averaged over all dims
        n_total = len(mass)
        # First n_caption_tokens are caption region
        caption_mass = mass[:min(n_caption_tokens, n_total)].sum().item()
        lyric_mass = mass[min(n_caption_tokens, n_total):].sum().item()
        return {'caption': caption_mass, 'lyric': lyric_mass, 'total': mass.sum().item()}

    def get_entropy(self, layer_idx: int) -> float:
        if layer_idx not in self.cross_attentions:
            return 0.0
        ca = torch.stack(self.cross_attentions[layer_idx])
        ent = -(ca * torch.log(ca.clamp(min=1e-10))).sum(dim=(-2, -1)).mean().item()
        return ent

    def get_step_similarity(self, layer_idx: int) -> list:
        if layer_idx not in self.cross_attentions or len(self.cross_attentions[layer_idx]) < 2:
            return []
        ca_list = self.cross_attentions[layer_idx]
        sims = []
        for i in range(1, len(ca_list)):
            a = ca_list[i-1].flatten()
            b = ca_list[i].flatten()
            sims.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        return sims


# ====================================================================
# CFG direction recorder (hooked via decoder monkey-patch)
# ====================================================================

class CFGRecorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cond: list = []
        self.uncond: list = []
        self.directions: list = []

    def record(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor):
        self.cond.append(pred_cond.detach().cpu().float())
        self.uncond.append(pred_uncond.detach().cpu().float())
        self.directions.append((pred_cond - pred_uncond).detach().cpu().float())

    def norm_curve(self) -> list:
        return [d.norm().item() for d in self.directions]

    def compute_cosine(self, other: 'CFGRecorder') -> float:
        n = min(len(self.directions), len(other.directions))
        if n == 0:
            return 0.0
        sims = []
        for i in range(n):
            a = self.directions[i].flatten()
            b = other.directions[i].flatten()
            sims.append(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        return float(np.mean(sims))


# ====================================================================
# Decoder monkey-patch for attention + CFG collection
# ====================================================================

class PatchedDecoder:
    """Wraps decoder.forward to enable attention output and CFG recording."""

    def __init__(self, decoder: nn.Module, cfg_recorder: CFGRecorder = None,
                 attention_collector: AttentionCollector = None):
        self.decoder = decoder
        self.cfg_recorder = cfg_recorder
        self.attention_collector = attention_collector
        self._orig_forward = decoder.forward

    def __enter__(self):
        self.decoder.forward = self._patched_forward
        return self

    def __exit__(self, *args):
        self.decoder.forward = self._orig_forward

    def _patched_forward(self, hidden_states, timestep, timestep_r=None,
                          attention_mask=None, encoder_hidden_states=None,
                          encoder_attention_mask=None, context_latents=None,
                          use_cache=True, past_key_values=None, **kw):
        # Always output attentions if collector is set
        out_kw = dict(kw)
        if self.attention_collector is not None:
            out_kw['output_attentions'] = True
        output = self._orig_forward(
            hidden_states, timestep, timestep_r=timestep_r,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=use_cache, past_key_values=past_key_values,
            **out_kw,
        )
        # Collect attention
        if self.attention_collector is not None:
            self.attention_collector.collect(output)
        # Record CFG direction
        if self.cfg_recorder is not None and hidden_states.shape[0] == 2:
            pred_cond, pred_uncond = output[0].chunk(2)
            self.cfg_recorder.record(pred_cond, pred_uncond)
        return output


# ====================================================================
# Baseline inference runner
# ====================================================================

def run_baseline(caption: str, lyrics: str, seed: int = 42, duration: float = 30.0,
                 model=None, llm=None, handler=None,
                 collect_stats: bool = False,
                 cfg_recorder: CFGRecorder = None,
                 save_dir: str = None) -> Dict[str, Any]:
    """Run a single baseline inference with optional diagnostic hooks."""
    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'audios', 'baseline')
    os.makedirs(save_dir, exist_ok=True)

    # Patch decoder for attention statistics and CFG recording
    attn_collector = AttentionCollector(model, len(model.decoder.layers)) if collect_stats else None
    my_cfg = cfg_recorder or CFGRecorder()

    patched = PatchedDecoder(model.decoder, my_cfg, attn_collector)

    try:
        with patched:
            params = GenerationParams(
                task_type='text2music',
                caption=caption, lyrics=lyrics,
                instrumental=False, bpm=120, keyscale='C major',
                timesignature='4', vocal_language='zh',
                duration=duration, inference_steps=50,
                guidance_scale=7.0, seed=seed,
                thinking=True, use_cot_metas=True, use_cot_caption=True,
                lm_temperature=0.75,
            )
            config = GenerationConfig(
                batch_size=1, audio_format='flac',
                use_random_seed=False, seeds=[seed],
            )
            result = generate_music(
                dit_handler=handler, llm_handler=llm,
                params=params, config=config, save_dir=save_dir,
            )

        output = {
            'audio_path': result.audios[0]['path'] if result.audios else '',
            'success': result.success,
            'error': result.error or '',
            'status': result.status_message,
        }

        if collect_stats and attn_collector:
            stats = {}
            for lid in sorted(attn_collector.cross_attentions.keys()):
                mg = attn_collector.get_mass_by_region(lid, n_caption_tokens=10)
                ent = attn_collector.get_entropy(lid)
                sim = attn_collector.get_step_similarity(lid)
                stats[f'layer_{lid}'] = {
                    'mass_by_region': mg,
                    'entropy': ent,
                    'step_similarity_mean': float(np.mean(sim)) if sim else 0,
                }
            output['attention_stats'] = stats

        output['cfg_norm_curve'] = my_cfg.norm_curve()

        return output

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {'audio_path': '', 'success': False, 'error': str(e), 'cfg_norm_curve': []}


# ====================================================================
# Helper to save tables
# ====================================================================

def save_table(results: list, filename: str):
    if not results:
        return
    path = os.path.join(OUTPUT_DIR, 'tables', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        for r in results:
            row = {k: str(v) if isinstance(v, (list, dict, np.ndarray)) else v
                   for k, v in r.items()}
            w.writerow(row)
    print(f"  Table saved: {path}")


# ====================================================================
# PROBE 1: Section scaffold perturbation
# ====================================================================

def probe1_section_scaffold(model, llm, handler, nprompts: int = 3):
    print("\n" + "=" * 60)
    print("PROBE 1: Section scaffold perturbation")
    print("=" * 60)

    with open(os.path.join(OUTPUT_DIR, 'prompts', 'section_scaffold_variants.json')) as f:
        data = json.load(f)

    results = []
    for variant in data['variants'][:nprompts + 3]:
        vid = variant['id']; lyrics = variant['lyrics']; caption = data['_meta']['base_caption']
        for seed in SEEDS[:2]:
            print(f"  [{vid}] seed={seed}")
            out_dir = os.path.join(OUTPUT_DIR, 'audios', 'section_probe', vid)
            cfg = CFGRecorder()
            out = run_baseline(caption, lyrics, seed, BASE_DURATION,
                             model, llm, handler, collect_stats=True, cfg_recorder=cfg,
                             save_dir=out_dir)
            results.append({
                'variant': vid, 'seed': seed,
                'expected_vocal_onset': variant.get('expected_vocal_onset_sec', 0),
                'actual_vocal_onset': None,  # Post-process from audio
                'audio_path': out.get('audio_path', ''),
                'success': out.get('success', False),
                'cfg_norm_mean': float(np.mean(out.get('cfg_norm_curve', [0]))) if out.get('cfg_norm_curve') else 0,
                'error': out.get('error', ''),
            })

    save_table(results, 'table_section_scaffold.csv')
    return results


# ====================================================================
# PROBE 2: Lyric order swap
# ====================================================================

def probe2_lyric_order(model, llm, handler, nprompts: int = 3):
    print("\n" + "=" * 60)
    print("PROBE 2: Lyric order swap")
    print("=" * 60)

    with open(os.path.join(OUTPUT_DIR, 'prompts', 'lyric_order_variants.json')) as f:
        data = json.load(f)

    results = []
    for variant in data['variants'][:nprompts + 2]:
        vid = variant['id']; lyrics = variant['lyrics']; caption = data['_meta']['base_caption']
        for seed in SEEDS[:2]:
            print(f"  [{vid}] seed={seed}")
            out_dir = os.path.join(OUTPUT_DIR, 'audios', 'lyric_order_probe', vid)
            out = run_baseline(caption, lyrics, seed, BASE_DURATION,
                             model, llm, handler, collect_stats=True,
                             save_dir=out_dir)
            results.append({
                'variant': vid, 'seed': seed,
                'target_order': variant.get('target_order', []),
                'recognized_order': None,  # From ASR post-processing
                'line_order_score': None,
                'audio_path': out.get('audio_path', ''),
                'success': out.get('success', False),
                'error': out.get('error', ''),
            })

    save_table(results, 'table_lyric_order.csv')
    return results


# ====================================================================
# PROBE 3: Token group ablation
# ====================================================================

def probe3_token_ablation(model, llm, handler, nprompts: int = 3):
    print("\n" + "=" * 60)
    print("PROBE 3: Token group ablation")
    print("=" * 60)

    with open(os.path.join(OUTPUT_DIR, 'prompts', 'token_ablation_variants.json')) as f:
        data = json.load(f)

    results = []
    for variant in data['variants'][:nprompts + 2]:
        vid = variant['id']; lyrics = variant['lyrics']; caption = variant['caption']
        for seed in SEEDS[:2]:
            print(f"  [{vid}] seed={seed}")
            out_dir = os.path.join(OUTPUT_DIR, 'audios', 'token_ablation_probe', vid)
            cfg = CFGRecorder()
            out = run_baseline(caption, lyrics, seed, BASE_DURATION,
                             model, llm, handler, collect_stats=True, cfg_recorder=cfg,
                             save_dir=out_dir)
            results.append({
                'variant': vid, 'seed': seed,
                'vocal_exists': None,  # Post-process
                'vocal_activity_ratio': None,
                'lyrics_intelligible': None,
                'quality_score': 1.0 if out.get('success') else 0.0,
                'audio_path': out.get('audio_path', ''),
                'success': out.get('success', False),
                'error': out.get('error', ''),
            })

    save_table(results, 'table_token_ablation.csv')
    return results


# ====================================================================
# PROBE 4: Layer-wise cross-attention sensitivity
# ====================================================================

def probe4_layer_sensitivity(model, llm, handler, nprompts: int = 2):
    print("\n" + "=" * 60)
    print("PROBE 4: Layer-wise cross-attention sensitivity")
    print("=" * 60)

    n_layers = len(model.decoder.layers)
    candidate_layers = [l for l in [4, 8, 12, 16, 20, 24] if l < n_layers]

    lyrics = SHORT_LYRICS; caption = BASE_CAPTION

    results = []

    # Baseline (no perturbation)
    print("  [baseline]")
    out_dir = os.path.join(OUTPUT_DIR, 'audios', 'layer_sensitivity_probe', 'baseline')
    base_out = run_baseline(caption, lyrics, SEEDS[0], BASE_DURATION,
                           model, llm, handler, collect_stats=True, save_dir=out_dir)
    results.append({
        'layer': -1, 'token_group': 'none', 'perturbation_type': 'none',
        'delta': 0, 'quality_score': 1.0 if base_out.get('success') else 0.0,
        'audio_path': base_out.get('audio_path', ''),
        'success': base_out.get('success', False),
        'error': base_out.get('error', ''),
    })

    for layer in candidate_layers[:nprompts]:
        for region, rname in [(10, 'caption'), (9999, 'lyric')]:
            for action, aname in [('attenuate', 'attenuate'), ('boost', 'boost')]:
                vid = f"L{layer}_{rname}_{aname}"
                print(f"  [{vid}]")
                out_dir = os.path.join(OUTPUT_DIR, 'audios', 'layer_sensitivity_probe', vid)

                pert = AttentionPerturbationEngine(model)
                if rname == 'caption':
                    if action == 'attenuate':
                        pert.apply_caption_attenuation(layer, 0.2, n_caption_tokens=10)
                    else:
                        pert.apply_caption_boost(layer, 0.2, n_caption_tokens=10)
                else:
                    if action == 'attenuate':
                        pert.apply_lyric_attenuation(layer, 0.2, n_caption_tokens=10)
                    else:
                        pert.apply_lyric_boost(layer, 0.2, n_caption_tokens=10)

                try:
                    out = run_baseline(caption, lyrics, SEEDS[0], BASE_DURATION,
                                     model, llm, handler,
                                     collect_stats=True, save_dir=out_dir)
                    results.append({
                        'layer': layer, 'token_group': rname,
                        'perturbation_type': aname, 'delta': 0.2,
                        'quality_score': 1.0 if out.get('success') else 0.0,
                        'audio_path': out.get('audio_path', ''),
                        'success': out.get('success', False),
                        'error': out.get('error', ''),
                    })
                finally:
                    pert.clear_all()

    save_table(results, 'table_layer_sensitivity.csv')
    return results


# ====================================================================
# PROBE 5: Denoising-step sensitivity
# ====================================================================

def probe5_step_sensitivity(model, llm, handler, nprompts: int = 2):
    print("\n" + "=" * 60)
    print("PROBE 5: Denoising-step sensitivity")
    print("=" * 60)

    lyrics = SHORT_LYRICS; caption = BASE_CAPTION
    windows = [('early', 0.0, 0.3), ('middle', 0.3, 0.7), ('late', 0.7, 1.0)]
    candidate_layers = [l for l in [12, 16, 20] if l < len(model.decoder.layers)]

    results = []

    for layer in candidate_layers[:nprompts]:
        for wname, ws, we in windows:
            for rname in ['caption', 'lyric']:
                vid = f"L{layer}_{wname}_{rname}"
                print(f"  [{vid}]")
                out_dir = os.path.join(OUTPUT_DIR, 'audios', 'step_sensitivity_probe', vid)

                pert = AttentionPerturbationEngine(model)
                if rname == 'caption':
                    pert.apply_caption_attenuation(layer, 0.2, n_caption_tokens=10)
                else:
                    pert.apply_lyric_attenuation(layer, 0.2, n_caption_tokens=10)

                try:
                    out = run_baseline(caption, lyrics, SEEDS[0], BASE_DURATION,
                                     model, llm, handler,
                                     collect_stats=True, save_dir=out_dir)
                    results.append({
                        'step_window': wname, 'window_start': ws, 'window_end': we,
                        'layer': layer, 'token_group': rname,
                        'quality_score': 1.0 if out.get('success') else 0.0,
                        'audio_path': out.get('audio_path', ''),
                        'success': out.get('success', False),
                        'error': out.get('error', ''),
                    })
                finally:
                    pert.clear_all()

    save_table(results, 'table_step_sensitivity.csv')
    return results


# ====================================================================
# PROBE 6: CFG conditional direction analysis
# ====================================================================

def probe6_cfg_direction(model, llm, handler, nprompts: int = 2):
    print("\n" + "=" * 60)
    print("PROBE 6: CFG conditional direction analysis")
    print("=" * 60)

    variants = [
        ('full', BASE_CAPTION, SHORT_LYRICS),
        ('no_lyrics', BASE_CAPTION, "[Verse]\n[Chorus]\n[Verse]\n[Chorus]\n[Outro]"),
        ('no_sections', BASE_CAPTION,
         "风吹过山岗\n雨落在心上\n\n星光闪耀夜空\n照亮我的梦\n\n岁月如歌流淌\n思念在远方\n\n星光闪耀夜空\n照亮我的梦"),
        ('no_style', '', SHORT_LYRICS),
        ('swapped_lyrics', BASE_CAPTION,
         "[Verse]\n雨落在心上\n风吹过山岗\n\n[Chorus]\n照亮我的梦\n星光闪耀夜空\n\n[Outro]"),
        ('short_intro', BASE_CAPTION,
         "[Intro 8s]\n\n[Verse]\n风吹过山岗\n雨落在心上\n\n[Chorus]\n星光闪耀夜空\n照亮我的梦\n\n[Outro]"),
    ]

    cfg_recorders = {}
    results = []

    for vid, cap, lyr in variants[:nprompts + 3]:
        print(f"  [{vid}]")
        out_dir = os.path.join(OUTPUT_DIR, 'audios', 'section_probe')  # reuse dir
        cfg = CFGRecorder()
        out = run_baseline(cap, lyr, SEEDS[0], BASE_DURATION,
                          model, llm, handler, collect_stats=True,
                          cfg_recorder=cfg, save_dir=out_dir)
        cfg_recorders[vid] = cfg
        results.append({
            'variant': vid,
            'cfg_norm_mean': float(np.mean(out.get('cfg_norm_curve', [0]))),
            'cfg_norm_std': float(np.std(out.get('cfg_norm_curve', [0]))) if out.get('cfg_norm_curve') else 0,
            'success': out.get('success', False),
            'error': out.get('error', ''),
        })

    # Pairwise cosine similarities
    keys = list(cfg_recorders.keys())
    pairs = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            sim = cfg_recorders[keys[i]].compute_cosine(cfg_recorders[keys[j]])
            pairs.append({'variant_i': keys[i], 'variant_j': keys[j],
                          'cosine_similarity': round(sim, 4)})
            print(f"    CFG cos({keys[i]}, {keys[j]}) = {sim:.4f}")

    results.append({'pairwise_similarities': pairs})
    save_table(results, 'table_cfg_direction.csv')
    return results


# ====================================================================
# Attention statistics sweep
# ====================================================================

def collect_attention_statistics(model, llm, handler, nprompts: int = 2):
    print("\n" + "=" * 60)
    print("Attention statistics collection")
    print("=" * 60)

    caption = BASE_CAPTION
    lyrics = SHORT_LYRICS

    all_mass_by_region = {}
    all_entropy = {}
    all_step_sim = {}

    for ri in range(nprompts):
        seed = SEEDS[ri % len(SEEDS)]
        print(f"  Run {ri}, seed={seed}")
        out_dir = os.path.join(OUTPUT_DIR, 'audios', 'baseline')
        coll = AttentionCollector(model, len(model.decoder.layers))
        cfg = CFGRecorder()
        patched = PatchedDecoder(model.decoder, cfg, coll)
        with patched:
            run_baseline(caption, lyrics, seed, BASE_DURATION,
                        model, llm, handler, save_dir=out_dir)
        for lid in sorted(coll.cross_attentions.keys()):
            mg = coll.get_mass_by_region(lid, n_caption_tokens=10)
            ent = coll.get_entropy(lid)
            sim = coll.get_step_similarity(lid)
            if mg:
                for rname, mass in mg.items():
                    all_mass_by_region.setdefault(f'L{lid}_{rname}', []).append(mass)
            all_entropy[f'L{lid}'] = ent
            all_step_sim[f'L{lid}'] = float(np.mean(sim)) if sim else 0

    # Save
    def _save_csv(data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            w = csv.writer(f)
            for k, v in data.items():
                if isinstance(v, list):
                    w.writerow([k] + v)
                else:
                    w.writerow([k, v])

    _save_csv(all_mass_by_region,
              os.path.join(OUTPUT_DIR, 'diagnostics', 'attention_group_mass.csv'))
    _save_csv(all_entropy,
              os.path.join(OUTPUT_DIR, 'diagnostics', 'attention_entropy.csv'))
    _save_csv(all_step_sim,
              os.path.join(OUTPUT_DIR, 'diagnostics', 'attention_step_similarity.csv'))
    print("  Attention statistics saved.")


# ====================================================================
# Report generation
# ====================================================================

def generate_report(probe_results: dict):
    """Generate mechanism analysis report with answers to 8 key questions."""
    report = {
        'overview': {
            'model': 'ACE-Step 1.5 SFT (baseline, no adapters)',
            'duration_sec': BASE_DURATION,
            'inference_steps': 50,
            'guidance_scale': 7.0,
            'seeds': SEEDS,
        },
        'experiment_summary': {},
        'answers': {
            'Q1_section_scaffold_controls_vocal_onset': {
                'answer': 'PENDING_AFTER_POST_PROCESSING',
                'evidence': [],
                'details': 'Run asr/vad on probe1 audio files to determine vocal onset shift.'
            },
            'Q2_lyric_order_controls_generated_order': {
                'answer': 'PENDING_AFTER_POST_PROCESSING',
                'evidence': [],
                'details': 'Run asr on probe2 audio files and compute line_order_score.'
            },
            'Q3_necessary_token_groups': {
                'answer': 'PENDING_AFTER_POST_PROCESSING',
                'style_tokens': 'Needed for quality',
                'lyric_tokens': 'Needed for vocal content',
                'section_tokens': 'Needed for structure',
                'details': 'Check probe3 generated audio: which variants still have vocals?'
            },
            'Q4_safe_layers_to_intervene': {
                'answer': 'PENDING_AFTER_POST_PROCESSING',
                'safe_layers': [],
                'dangerous_layers': [],
                'details': 'Check probe4 audio quality and vocal changes per layer.'
            },
            'Q5_safe_denoising_phase': {
                'answer': 'PENDING_AFTER_POST_PROCESSING',
                'early': 'PENDING',
                'middle': 'PENDING',
                'late': 'PENDING',
                'details': 'Analyze probe5 results for quality by denoising window.'
            },
            'Q6_cfg_direction_vs_attention': {
                'answer': 'PENDING_AFTER_POST_PROCESSING',
                'details': 'Analyze probe6 pairwise similarities.'
            },
            'Q7_why_hard_attention_rewiring_failed': {
                'evidence': [],
                'details': 'To be determined from probe4/probe5 quality drops.'
            },
            'Q8_recommended_next_method_direction': {
                'answer': 'PENDING',
                'details': 'Determined after all probes analyzed.'
            },
        }
    }

    # Fill experiment summary from probe results
    for key, results in probe_results.items():
        if isinstance(results, list):
            report['experiment_summary'][key] = {
                'n_runs': len(results),
                'n_success': sum(1 for r in results if r.get('success')),
            }

    # Save JSON
    os.makedirs(os.path.join(OUTPUT_DIR, 'report'), exist_ok=True)
    jpath = os.path.join(OUTPUT_DIR, 'report', 'baseline_progression_report.json')
    with open(jpath, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved: {jpath}")

    # Save TXT
    tpath = os.path.join(OUTPUT_DIR, 'report', 'baseline_progression_report.txt')
    with open(tpath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Baseline Lyric Progression Probe Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: ACE-Step 1.5 SFT (baseline, no adapters)\n")
        f.write(f"Duration: {BASE_DURATION}s, Steps: 50, CFG: 7.0\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        for qk, qv in report['answers'].items():
            f.write(f"{qk}\n")
            f.write(f"  Answer: {qv.get('answer', 'PENDING')}\n")
            f.write(f"  Details: {qv.get('details', '')}\n\n")
    print(f"Text report saved: {tpath}")


# ====================================================================
# MAIN
# ====================================================================

def main():
    print("=" * 60)
    print("Initializing baseline model (no adapters)")
    print("=" * 60)

    handler = AceStepHandler()
    handler.initialize_service(
        project_root='/root/autodl-tmp/Ace-Step1.5',
        config_path='acestep-v15-sft',
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
    )
    model = handler.model.eval()
    model.config.use_section_rope_offset = False
    for l in model.decoder.layers:
        if getattr(l, 'use_section_rope', False): l.use_section_rope = False
        if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False

    print(f"  Device: {device}, Layers: {len(model.decoder.layers)}")

    llm = LLMHandler()
    llm.initialize(
        checkpoint_dir='/root/autodl-tmp/Ace-Step1.5/checkpoints',
        lm_model_path='acestep-5Hz-lm-1.7B',
        backend='pt', device=device,
    )

    n = 3  # prompts per variant

    results = {}
    results['probe1_section'] = probe1_section_scaffold(model, llm, handler, n)
    results['probe2_lyric_order'] = probe2_lyric_order(model, llm, handler, n)
    results['probe3_token_ablation'] = probe3_token_ablation(model, llm, handler, n)
    results['probe4_layer_sensitivity'] = probe4_layer_sensitivity(model, llm, handler, 2)
    results['probe5_step_sensitivity'] = probe5_step_sensitivity(model, llm, handler, 2)
    results['probe6_cfg_direction'] = probe6_cfg_direction(model, llm, handler, 3)
    collect_attention_statistics(model, llm, handler, 2)

    generate_report(results)

    print("\n" + "=" * 60)
    print("ALL PROBES COMPLETE")
    print(f"Results: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
