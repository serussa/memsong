#!/usr/bin/env python3
"""
PD-SBAR 完整评估：10 prompts × 3 方法 → 生成音频 → SongEval → structured metrics
"""
import os, sys, json, time, argparse, re, glob
from pathlib import Path
from copy import deepcopy

ACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACE_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"
os.environ["SIDESTEP_SAFE_ROOT"] = "/root/autodl-tmp"

import torch
import numpy as np

# ===========================================================================
# PROMPTS: 10 diverse prompts with structured lyrics
# ===========================================================================
PROMPTS = [
    ("pop_girl", "pop, female vocal, catchy melody, piano, drums, 120 bpm, C major",
     """[INTRO]

[VERSE]
Walking through the city lights
Every shadow comes alive
I can feel the rhythm grow
Letting all my feelings show

[CHORUS]
Dancing in the neon glow
Where the music takes control
Tonight we're gonna lose control
Let the melody unfold

[VERSE]
Stars are shining from above
Every heartbeat sings of love
Moving to the endless beat
Feel the world beneath my feet

[CHORUS]
Dancing in the neon glow
Where the music takes control
Tonight we're gonna lose control
Let the melody unfold

[BRIDGE]
Higher and higher we go
Letting the music flow

[CHORUS]
Dancing in the neon glow
Where the music takes control
Tonight we're gonna lose control
Let the melody unfold

[OUTRO]"""),
    ("rock_ballad", "rock, electric guitar, drums, bass, powerful male vocal, 140 bpm, G minor",
     """[INTRO]

[VERSE]
The walls are closing in again
I've been fighting long my friend
Every scar upon my skin
Tells a story deep within

[CHORUS]
I will rise up from the ashes
Breaking through these iron chains
Nothing's ever gonna stop me
I will stand up in the rain

[VERSE]
Darkest hour comes before the dawn
Everything I love is gone
But I'm still standing here tonight
Ready for the final fight

[CHORUS]
I will rise up from the ashes
Breaking through these iron chains
Nothing's ever gonna stop me
I will stand up in the rain

[BRIDGE]
This is my redemption song
Nothing here can go wrong

[CHORUS]
I will rise up from the ashes
Breaking through these iron chains
Nothing's ever gonna stop me
I will stand up in the rain

[OUTRO]"""),
    ("jazz_blues", "jazz, blues, saxophone, double bass, piano, slow, 80 bpm, F major",
     """[INTRO]

[VERSE]
Smoke filled room at half past two
Playing songs I never knew
The whiskey's warm and the lights are low
Let the melancholy flow

[CHORUS]
Blue moon shining on my face
Time moves slow in this old place
Sing me one more song tonight
Till the morning brings the light

[VERSE]
Saxophone cries in the dark
Hitting notes that leave their mark
Every chord a memory
Playing songs just for me

[CHORUS]
Blue moon shining on my face
Time moves slow in this old place
Sing me one more song tonight
Till the morning brings the light

[OUTRO]"""),
    ("electronic_dance", "electronic, dance, synth, heavy bass, 4x4 beat, energetic, 128 bpm, A minor",
     """[INTRO]

[VERSE]
Feel the bass drop through the floor
Can't take it anymore
The rhythm's taking over me
Setting every fiber free

[CHORUS]
Pumping through the night
Everything feels right
Lose yourself in sound
Let the beat go round

[VERSE]
Laser lights paint the sky
No more reasons to be shy
Hands up, feel the energy
This is where we're meant to be

[CHORUS]
Pumping through the night
Everything feels right
Lose yourself in sound
Let the beat go round

[BRIDGE]
Drop it now

[CHORUS]
Pumping through the night
Everything feels right
Lose yourself in sound
Let the beat go round

[OUTRO]"""),
    ("acoustic_folk", "acoustic, folk, gentle guitar, soft male vocal, nature sounds, 100 bpm, G major",
     """[INTRO]

[VERSE]
Morning dew upon the grass
Watching all the clouds drift past
Simple life beneath the sun
Every day a new begun

[CHORUS]
Take me to the open road
Let me carry my own load
With the wind upon my face
I have found my happy place

[VERSE]
River flowing deep and wide
Nothing left for me to hide
Walking through the forest green
Feeling peaceful and serene

[CHORUS]
Take me to the open road
Let me carry my own load
With the wind upon my face
I have found my happy place

[BRIDGE]
Just me and the sky
No need to ask why

[CHORUS]
Take me to the open road
Let me carry my own load
With the wind upon my face
I have found my happy place

[OUTRO]"""),
    ("r_and_b", "r&b, soulful, male vocal, smooth, synth pad, 808 drums, 90 bpm, D minor",
     """[INTRO]

[VERSE]
Late night calls and empty streets
Thinking 'bout the heartbeats
That we shared under the moon
Wishing I could see you soon

[CHORUS]
Baby you're the only one
Shining brighter than the sun
Every moment feels so right
When I hold you through the night

[VERSE]
Silk sheets and candlelight
Everything feels so right
Your body moving close to mine
Losing track of space and time

[CHORUS]
Baby you're the only one
Shining brighter than the sun
Every moment feels so right
When I hold you through the night

[BRIDGE]
Let's stay here forever
We'll be young together

[CHORUS]
Baby you're the only one
Shining brighter than the sun
Every moment feels so right
When I hold you through the night

[OUTRO]"""),
    ("cinematic_orchestral", "cinematic, orchestral, strings, brass, epic, dramatic, 85 bpm, C minor",
     """[INTRO]

[VERSE]
Across the mountains high and low
Where the winter winds still blow
A hero stands against the tide
With nowhere left to hide

[CHORUS]
Rise up, rise up, hear the call
Standing tall before the fall
With the fire in our hearts
We will never be apart

[VERSE]
The armies march across the plain
Through the thunder and the rain
Together we will make a stand
Fighting for this sacred land

[CHORUS]
Rise up, rise up, hear the call
Standing tall before the fall
With the fire in our hearts
We will never be apart

[BRIDGE]
For honor and for glory
This is our story

[CHORUS]
Rise up, rise up, hear the call
Standing tall before the fall
With the fire in our hearts
We will never be apart

[OUTRO]"""),
    ("lofi_study", "lofi hip hop, chill, study beats, relaxed, vinyl crackle, 85 bpm, A major",
     """[INTRO]

[VERSE]
Raindrops on my window pane
Thoughts drifting like a train
Coffee warm between my hands
Making all my future plans

[CHORUS]
Let the hours drift away
In this peaceful state I stay
Nothing rushing, nothing due
Just the quiet and the view

[VERSE]
Books are stacked up on the desk
Mind is calm and sense refreshed
Pencil moving on the page
Slowly turning a new page

[CHORUS]
Let the hours drift away
In this peaceful state I stay
Nothing rushing, nothing due
Just the quiet and the view

[BRIDGE]
Time moves slow
Let it go

[CHORUS]
Let the hours drift away
In this peaceful state I stay
Nothing rushing, nothing due
Just the quiet and the view

[OUTRO]"""),
    ("synthwave", "synthwave, retro, 80s inspired, arpeggiator, reverb, driving beat, 130 bpm, D major",
     """[INTRO]

[VERSE]
Night drive on the endless road
Past the city lights that glow
Chrome and leather, neon signs
Leaving all the past behind

[CHORUS]
Running through the sunset strip
With the future on my lips
Every mile a memory
Setting all our spirits free

[VERSE]
Digital world before my eyes
Stars reflected in the sky
Retro dreams of what's to come
Beating like a synthwave drum

[CHORUS]
Running through the sunset strip
With the future on my lips
Every mile a memory
Setting all our spirits free

[BRIDGE]
Into the night we ride
With nothing left to hide

[CHORUS]
Running through the sunset strip
With the future on my lips
Every mile a memory
Setting all our spirits free

[OUTRO]"""),
    ("chinese_pop", "pop, mandopop, female vocal, piano, strings, emotional, 110 bpm, C major",
     """[INTRO]

[VERSE]
月光洒在窗台前
想起你的笑脸
时光匆匆不停歇
回忆在心中盘旋

[CHORUS]
就让这首歌 带走我的思念
在每个夜里 轻轻唱一遍
不管多远 你都在我心间
这份爱永远不会改变

[VERSE]
风吹过的地方
都有你的芳香
漫步在熟悉的路
仿佛你还在身旁

[CHORUS]
就让这首歌 带走我的思念
在每个夜里 轻轻唱一遍
不管多远 你都在我心间
这份爱永远不会改变

[BRIDGE]
想你的时候
抬头看天空

[CHORUS]
就让这首歌 带走我的思念
在每个夜里 轻轻唱一遍
不管多远 你都在我心间
这份爱永远不会改变

[OUTRO]"""),
]


# ===========================================================================
#  SongEval helper
# ===========================================================================
def run_songeval(audio_dir: str, output_path: str) -> dict:
    """Run MuQ-based SongEval on all audio files in a directory."""
    import librosa
    from muq import MuQ
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(str(ACE_ROOT / "SongEval" / "config.yaml"))
    model = instantiate(config.generator).to(device).eval()
    model.load_state_dict(load_file(str(ACE_ROOT / "SongEval" / "ckpt" / "model.safetensors"), device="cpu"), strict=False)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()

    METRICS = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness']
    files = sorted(glob.glob(f"{audio_dir}/*.wav") + glob.glob(f"{audio_dir}/*.flac") + glob.glob(f"{audio_dir}/*.mp3"))
    print(f"  SongEval: {len(files)} files from {audio_dir}")

    if not files:
        return {"error": "no_audio_files", "count": 0}

    all_scores = {m: [] for m in METRICS}
    per_file = {}
    for fpath in files:
        name = Path(fpath).stem
        try:
            wav, sr = librosa.load(fpath, sr=24000)
            wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                emb = muq(wav_t)
                scores = model(emb)[0]
            scores = scores.float().cpu().numpy().flatten()
            entry = {m: round(float(scores[i]), 4) for i, m in enumerate(METRICS)}
            per_file[name] = entry
            for i, m in enumerate(METRICS):
                all_scores[m].append(float(scores[i]))
        except Exception as e:
            print(f"    SongEval error {name}: {e}")
            per_file[name] = {"error": str(e)}

    avg = {m: round(float(np.mean(all_scores[m])), 4) for m in METRICS if all_scores[m]}
    std_ = {m: round(float(np.std(all_scores[m])), 4) for m in METRICS if all_scores[m]}
    result = {"average": avg, "std": std_, "per_file": per_file, "count": len(files)}
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"  SongEval results -> {output_path}")
    return result


# ===========================================================================
#  Audio generation helper (single prompt in subprocess to reset CUDA)
# ===========================================================================
def gen_one_prompt(name, caption, lyrics, duration, seed, method_dir, method,
                   pd_sbar_ckpt=None):
    """Generate one audio file and save to method_dir/name.flac"""
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    dt = AceStepHandler()
    dt.initialize_service(
        project_root="/root/autodl-tmp/Ace-Step1.5",
        config_path="acestep-v15-sft", device="cuda",
        use_flash_attention=False, compile_model=False, offload_to_cpu=False,
    )
    model = dt.model.eval()
    device = next(model.parameters()).device
    model.config.use_section_rope_offset = False
    for l in model.decoder.layers:
        if getattr(l, "use_section_rope", False): l.use_section_rope = False
        if getattr(l, "use_phase_memory", False): l.use_phase_memory = False

    # Force eager attention
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            if m.config._attn_implementation != "eager":
                m.config._attn_implementation = "eager"
    if hasattr(model.config, "_attn_implementation_compiled"):
        model.config._attn_implementation_compiled = None

    T_eff = int(duration * 25)
    T_audio = T_eff // max(getattr(model.config, "patch_size", 2), 1)
    L_enc = 768

    # Build planner + reparameterizer for PD-SBAR
    if method != "baseline":
        from acestep.phase_memory import (
            PDSBARPlanner, SinkhornBregmanReparameterizer,
            parse_lyrics_to_units, build_duration_scaffold,
        )
        section_ids = torch.zeros(L_enc, dtype=torch.long, device=device)
        units, _, debug = parse_lyrics_to_units(
            lyrics, section_ids[0].cpu(),
            auto_transition_ratios=dict(intro=0., outro=0., chorus_to_verse=0.,
                                        chorus_to_bridge=0., bridge_to_chorus=0.),
        )
        tcm = debug.get("tag_control_mask", None)
        scaffold = build_duration_scaffold(units, text_len=L_enc, tag_control_mask=tcm)
        scaffold = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in scaffold.items()}

        planner = PDSBARPlanner(
            sigma=0.18, leak_cost=4.0, epsilon=0.05, sinkhorn_iters=30,
            slack_ratio=0.08, budget_smoothing=0.1, eta_dual=1.0, lambda_max=3.0,
        ).to(device).float()
        planner.build_from_scaffold(scaffold, T_audio, batch_size=1, device=device)

        reparam = SinkhornBregmanReparameterizer(
            planner=planner, rewired_layers=[8, 12, 16, 20],
            alpha=0.6, gamma_leak=3.0, beta_dual=1.0,
        )

        # Load LoRA if specified
        if pd_sbar_ckpt:
            from acestep.training_v2.fixed_lora_module import _inject_lora_on_rewired_layers
            from safetensors.torch import load_file as sf_load
            _inject_lora_on_rewired_layers(
                model, rewired_layers=[8, 12, 16, 20],
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                rank=16, alpha=32,
            )
            state_dict = sf_load(pd_sbar_ckpt)
            sd = model.state_dict()
            for k, v in state_dict.items():
                if k in sd and sd[k].shape == v.shape:
                    sd[k].copy_(v)
            model.load_state_dict(sd, strict=False)

        # Install reparameterizer hook on generate_audio's decoder forward
        # We patch eager_attention_forward globally (it's already set by install)
        reparam.install(model)

    llm = LLMHandler()
    llm.initialize(
        checkpoint_dir="/root/autodl-tmp/Ace-Step1.5",
        lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda",
    )

    params = GenerationParams(
        task_type="text2music", caption=caption, lyrics=lyrics,
        instrumental=False, bpm=120, keyscale="C major", timesignature="4",
        vocal_language="en", duration=duration, inference_steps=50,
        guidance_scale=7.0, seed=seed, thinking=True,
        use_cot_metas=True, use_cot_caption=True, lm_temperature=0.75,
    )
    config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False, seeds=[seed])

    result = generate_music(
        dit_handler=dt, llm_handler=llm,
        params=params, config=config, save_dir=str(method_dir),
    )
    if result.success and result.audios:
        for a in result.audios:
            print(f"  -> {a['path']}")
    else:
        print(f"  FAIL: {result.error}")

    if method != "baseline":
        reparam.remove()


# ===========================================================================
#  Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                        default="/root/autodl-tmp/pd_sbar_eval_output")
    parser.add_argument("--duration", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pd-sbar-ckpt", type=str, default=None,
                        help="Path to PD-SBAR LoRA checkpoint. If not set, use inference-only.")
    parser.add_argument("--skip-gen", action="store_true", default=False,
                        help="Skip generation, only run SongEval on existing files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods_list = ["baseline", "pd_sbar"]
    if args.pd_sbar_ckpt or not args.skip_gen:
        methods_list.append("pd_sbar_trained")

    # Create method subdirectories
    for m in methods_list:
        if m in ("baseline", "pd_sbar", "pd_sbar_trained"):
            (output_dir / m / "audios").mkdir(parents=True, exist_ok=True)

    # ---- Generate audio ---------------------------------------------------
    if not args.skip_gen:
        print("=" * 70)
        print(f"  Generating {len(PROMPTS)} prompts × {len(methods_list)} methods")
        print("=" * 70)
        for name, caption, lyrics in PROMPTS:
            for method in methods_list:
                aud_dir = output_dir / method / "audios"
                existing = list(aud_dir.glob(f"{name}.*"))
                if existing:
                    print(f"  [skip] {method}/{name} — already exists: {existing[0].name}")
                    continue
                print(f"\n  [{method}] {name} ...")
                ckpt_path = args.pd_sbar_ckpt if method == "pd_sbar_trained" else None
                gen_one_prompt(
                    name, caption, lyrics, args.duration, args.seed,
                    str(aud_dir), method, pd_sbar_ckpt=ckpt_path,
                )
        print("\n  Generation complete!")
    else:
        print("  Skipping generation (--skip-gen)")

    # ---- SongEval ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Running SongEval")
    print("=" * 70)
    songeval_results = {}
    for method in methods_list:
        aud_dir = output_dir / method / "audios"
        out_path = str(output_dir / method / "songeval_results.json")
        if list(aud_dir.glob("*.*")):
            r = run_songeval(str(aud_dir), out_path)
            songeval_results[method] = r
        else:
            print(f"  [skip] {method} — no audio files")
            songeval_results[method] = {"error": "no_audio"}

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SongEval Comparison Summary")
    print("=" * 70)
    METRICS = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness']
    header = f"{'Method':<25}" + "".join(f" {m:>12}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for method in methods_list:
        r = songeval_results.get(method, {})
        avg = r.get("average", {})
        row = f"{method:<25}"
        for m in METRICS:
            row += f" {avg.get(m, 0):>12.4f}"
        print(row)
    print("=" * 70)

    # ---- Collect diagnostics from all methods -----------------------------
    all_results = {"methods": methods_list, "songeval": songeval_results}
    (output_dir / "all_results.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  Full results -> {output_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
