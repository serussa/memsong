#!/usr/bin/env python3
"""
Batch: 10 prompts × 3 methods × same seed → SongEval
"""
import os, sys, json, time, argparse
from pathlib import Path
from copy import deepcopy

ACE_STEP_ROOT = Path("/root/ACE-Step-1.5")
sys.path.insert(0, str(ACE_STEP_ROOT))
os.environ["ACESTEP_OFFLINE"] = "1"
os.environ["ACESTEP_MINIMAL_COMPONENTS"] = "1"

import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
from acestep.phase_memory import (
    TransportRetrievalAdapter, PMRetrievalPhaseMemory,
    parse_lyrics_to_units, build_duration_scaffold,
)
from acestep.tgca.lyrics_parser import LyricsStructureParser

CHECKPOINT = "/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt"
MODEL_ROOT = Path("/root/autodl-tmp/Ace-Step1.5")
OUTPUT_ROOT = Path("/root/autodl-tmp/batch_eval_30")
SEED = 42
DURATION = 150

PROMPTS = [
    ("pop_girl", "pop, female vocal, catchy melody, piano, drums, 120 bpm, C major",
     """[INTRO]\n\n[VERSE]\nWalking through the city lights\nEvery shadow comes alive\nI can feel the rhythm grow\nLetting all my feelings show\n\n[CHORUS]\nDancing in the neon glow\nWhere the music takes control\nTonight we're gonna lose control\nLet the melody unfold\n\n[VERSE]\nStars are shining from above\nEvery heartbeat sings of love\nMoving to the endless beat\nFeel the world beneath my feet\n\n[CHORUS]\nDancing in the neon glow\nWhere the music takes control\nTonight we're gonna lose control\nLet the melody unfold\n\n[BRIDGE]\nHigher and higher we go\nLetting the music flow\n\n[CHORUS]\nDancing in the neon glow\nWhere the music takes control\nTonight we're gonna lose control\nLet the melody unfold\n\n[OUTRO]"""),
    ("rock_ballad", "rock, electric guitar, drums, bass, powerful male vocal, 140 bpm, G minor",
     """[INTRO]\n\n[VERSE]\nThe walls are closing in again\nI've been fighting long my friend\nEvery scar upon my skin\nTells a story deep within\n\n[CHORUS]\nI will rise up from the ashes\nBreaking through these iron chains\nNothing's ever gonna stop me\nI will stand up in the rain\n\n[VERSE]\nDarkest hour comes before the dawn\nEverything I love is gone\nBut I'm still standing here tonight\nReady for the final fight\n\n[CHORUS]\nI will rise up from the ashes\nBreaking through these iron chains\nNothing's ever gonna stop me\nI will stand up in the rain\n\n[BRIDGE]\nThis is my redemption song\nNothing here can go wrong\n\n[CHORUS]\nI will rise up from the ashes\nBreaking through these iron chains\nNothing's ever gonna stop me\nI will stand up in the rain\n\n[OUTRO]"""),
    ("jazz_blues", "jazz, blues, saxophone, double bass, piano, slow, 80 bpm, F major",
     """[INTRO]\n\n[VERSE]\nSmoke filled room at half past two\nPlaying songs I never knew\nThe whiskey's warm and the lights are low\nLet the melancholy flow\n\n[CHORUS]\nBlue moon shining on my face\nTime moves slow in this old place\nSing me one more song tonight\nTill the morning brings the light\n\n[VERSE]\nSaxophone cries in the dark\nHitting notes that leave their mark\nEvery chord a memory\nPlaying songs just for me\n\n[CHORUS]\nBlue moon shining on my face\nTime moves slow in this old place\nSing me one more song tonight\nTill the morning brings the light\n\n[OUTRO]"""),
    ("electronic_dance", "electronic, dance, synth, heavy bass, 4x4 beat, energetic, 128 bpm, A minor",
     """[INTRO]\n\n[VERSE]\nFeel the bass drop through the floor\nCan't take it anymore\nThe rhythm's taking over me\nSetting every fiber free\n\n[CHORUS]\nPumping through the night\nEverything feels right\nLose yourself in sound\nLet the beat go round\n\n[VERSE]\nLaser lights paint the sky\nNo more reasons to be shy\nHands up, feel the energy\nThis is where we're meant to be\n\n[CHORUS]\nPumping through the night\nEverything feels right\nLose yourself in sound\nLet the beat go round\n\n[BRIDGE]\nDrop it now\n\n[CHORUS]\nPumping through the night\nEverything feels right\nLose yourself in sound\nLet the beat go round\n\n[OUTRO]"""),
    ("acoustic_folk", "acoustic, folk, gentle guitar, soft male vocal, nature sounds, 100 bpm, G major",
     """[INTRO]\n\n[VERSE]\nMorning dew upon the grass\nWatching all the clouds drift past\nSimple life beneath the sun\nEvery day a new begun\n\n[CHORUS]\nTake me to the open road\nLet me carry my own load\nWith the wind upon my face\nI have found my happy place\n\n[VERSE]\nRiver flowing deep and wide\nNothing left for me to hide\nWalking through the forest green\nFeeling peaceful and serene\n\n[CHORUS]\nTake me to the open road\nLet me carry my own load\nWith the wind upon my face\nI have found my happy place\n\n[BRIDGE]\nJust me and the sky\nNo need to ask why\n\n[CHORUS]\nTake me to the open road\nLet me carry my own load\nWith the wind upon my face\nI have found my happy place\n\n[OUTRO]"""),
    ("r_and_b", "r&b, soulful, male vocal, smooth, synth pad, 808 drums, 90 bpm, D minor",
     """[INTRO]\n\n[VERSE]\nLate night calls and empty streets\nThinking 'bout the heartbeats\nThat we shared under the moon\nWishing I could see you soon\n\n[CHORUS]\nBaby you're the only one\nShining brighter than the sun\nEvery moment feels so right\nWhen I hold you through the night\n\n[VERSE]\nSilk sheets and candlelight\nEverything feels so right\nYour body moving close to mine\nLosing track of space and time\n\n[CHORUS]\nBaby you're the only one\nShining brighter than the sun\nEvery moment feels so right\nWhen I hold you through the night\n\n[BRIDGE]\nLet's stay here forever\nWe'll be young together\n\n[CHORUS]\nBaby you're the only one\nShining brighter than the sun\nEvery moment feels so right\nWhen I hold you through the night\n\n[OUTRO]"""),
    ("cinematic_orchestral", "cinematic, orchestral, strings, brass, epic, dramatic, 85 bpm, C minor",
     """[INTRO]\n\n[VERSE]\nAcross the mountains high and low\nWhere the winter winds still blow\nA hero stands against the tide\nWith nowhere left to hide\n\n[CHORUS]\nRise up, rise up, hear the call\nStanding tall before the fall\nWith the fire in our hearts\nWe will never be apart\n\n[VERSE]\nThe armies march across the plain\nThrough the thunder and the rain\nTogether we will make a stand\nFighting for this sacred land\n\n[CHORUS]\nRise up, rise up, hear the call\nStanding tall before the fall\nWith the fire in our hearts\nWe will never be apart\n\n[BRIDGE]\nFor honor and for glory\nThis is our story\n\n[CHORUS]\nRise up, rise up, hear the call\nStanding tall before the fall\nWith the fire in our hearts\nWe will never be apart\n\n[OUTRO]"""),
    ("lofi_study", "lofi hip hop, chill, study beats, relaxed, vinyl crackle, 85 bpm, A major",
     """[INTRO]\n\n[VERSE]\nRaindrops on my window pane\nThoughts drifting like a train\nCoffee warm between my hands\nMaking all my future plans\n\n[CHORUS]\nLet the hours drift away\nIn this peaceful state I stay\nNothing rushing, nothing due\nJust the quiet and the view\n\n[VERSE]\nBooks are stacked up on the desk\nMind is calm and sense refreshed\nPencil moving on the page\nSlowly turning a new page\n\n[CHORUS]\nLet the hours drift away\nIn this peaceful state I stay\nNothing rushing, nothing due\nJust the quiet and the view\n\n[BRIDGE]\nTime moves slow\nLet it go\n\n[CHORUS]\nLet the hours drift away\nIn this peaceful state I stay\nNothing rushing, nothing due\nJust the quiet and the view\n\n[OUTRO]"""),
    ("synthwave", "synthwave, retro, 80s inspired, arpeggiator, reverb, driving beat, 130 bpm, D major",
     """[INTRO]\n\n[VERSE]\nNight drive on the endless road\nPast the city lights that glow\nChrome and leather, neon signs\nLeaving all the past behind\n\n[CHORUS]\nRunning through the sunset strip\nWith the future on my lips\nEvery mile a memory\nSetting all our spirits free\n\n[VERSE]\nDigital world before my eyes\nStars reflected in the sky\nRetro dreams of what's to come\nBeating like a synthwave drum\n\n[CHORUS]\nRunning through the sunset strip\nWith the future on my lips\nEvery mile a memory\nSetting all our spirits free\n\n[BRIDGE]\nInto the night we ride\nWith nothing left to hide\n\n[CHORUS]\nRunning through the sunset strip\nWith the future on my lips\nEvery mile a memory\nSetting all our spirits free\n\n[OUTRO]"""),
    ("chinese_pop", "pop, mandopop, female vocal, piano, strings, emotional, 110 bpm, C major",
     """[INTRO]\n\n[VERSE]\n月光洒在窗台前\n想起你的笑脸\n时光匆匆不停歇\n回忆在心中盘旋\n\n[CHORUS]\n就让这首歌 带走我的思念\n在每个夜里 轻轻唱一遍\n不管多远 你都在我心间\n这份爱永远不会改变\n\n[VERSE]\n风吹过的地方\n都有你的芳香\n漫步在熟悉的路\n仿佛你还在身旁\n\n[CHORUS]\n就让这首歌 带走我的思念\n在每个夜里 轻轻唱一遍\n不管多远 你都在我心间\n这份爱永远不会改变\n\n[BRIDGE]\n想你的时候\n抬头看天空\n\n[CHORUS]\n就让这首歌 带走我的思念\n在每个夜里 轻轻唱一遍\n不管多远 你都在我心间\n这份爱永远不会改变\n\n[OUTRO]"""),
]


def setup_retrieval(model, lyrics_text, T_eff, use_pm_gate, device):
    """Inject retrieval hook. Returns hook handle for cleanup."""
    D = model.config.hidden_size
    pm = PMRetrievalPhaseMemory(
        dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True,
    ).to(device).float()
    adapt = TransportRetrievalAdapter(
        hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
        transport_mode="sinkhorn", sinkhorn_iters=10,
        transport_sigma=0.18, scoring_mode="position_only",
        use_pm_gate=use_pm_gate, gate_hidden_dim=128,
        write_alpha_init=0.005, write_alpha_max=0.01,
        out_proj_init_std=0.01,
    ).to(device).float()

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    pm.load_state_dict(ckpt["phase_memory"])
    adapt.load_state_dict(ckpt["retrieval_adapter"])
    adapt.eval(); pm.eval()

    orig_prepare = model.prepare_condition
    hook_handle = [None]

    def patched_prepare(*args, **kwargs):
        result = orig_prepare(*args, **kwargs)
        if result[0] is not None and hook_handle[0] is None:
            enc_hs = result[0]; L_enc = enc_hs.shape[1]; B_e, D_h = enc_hs.shape[0], enc_hs.shape[-1]
            parser = LyricsStructureParser()
            parsed = parser.parse(lyrics_text, num_chunks=L_enc)
            units, _, debug = parse_lyrics_to_units(lyrics_text, parsed.section_type_ids, auto_transition_ratios={})
            sc = build_duration_scaffold(units, text_len=L_enc, tag_control_mask=debug.get("tag_control_mask"))
            sc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
            U = len(sc["unit_boundaries"]) - 1
            c_all = (sc["unit_boundaries"][:-1] + sc["unit_boundaries"][1:]) / 2
            mu_all = sc["unit_duration"] / sc["unit_duration"].sum()
            usid = sc["unit_section_ids"]; uil = sc["lyric_unit_mask"]
            t2u = sc["token_to_unit"]; lym = sc["lyric_mask"]
            eh_f = enc_hs.float()
            uth_list = []
            for uid in range(U):
                tmask = (t2u == uid) & lym if uil[uid].item() else (t2u == uid)
                tmb = tmask.unsqueeze(0).expand(B_e, -1)
                uth_list.append(eh_f[tmb].view(B_e, -1, D_h).mean(dim=1) if tmb.any() else torch.zeros(B_e, D_h, device=device, dtype=torch.float32))
            unit_text_hidden = torch.stack(uth_list, dim=1)
            p_audio = torch.linspace(0, 1, T_eff, device=device, dtype=torch.float32).unsqueeze(0)

            def _hook(_mod, _inp, out):
                H = out[0]; B, T_cur = H.shape[0], H.shape[1]
                pa = p_audio[:, :T_cur]
                if B > pa.shape[0]: pa = pa.expand(B, -1).contiguous()
                with torch.no_grad():
                    ps = pm(H.float(), torch.zeros(B, device=device, dtype=torch.float32))
                    dh, Pi, diag = adapt(H.float(), None, ps, pa.float(), unit_text_hidden.float(),
                                         c_all.unsqueeze(0).float(), mu_all.unsqueeze(0).float(),
                                         usid.unsqueeze(0), uil.unsqueeze(0))
                return ((H + dh.to(dtype=H.dtype)), *out[1:])
            hook_handle[0] = model.decoder.layers[12].register_forward_hook(_hook)
        return result

    model.prepare_condition = patched_prepare
    return hook_handle, orig_prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["baseline", "sinkhorn_only", "sinkhorn_pmgate"])
    parser.add_argument("--prompts", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=150)
    parser.add_argument("--eval-only", action="store_true", help="Only run SongEval on existing files")
    args = parser.parse_args()

    seed = args.seed
    duration = args.duration
    methods = args.methods
    prompt_indices = args.prompts if args.prompts else list(range(len(PROMPTS)))

    total = len(prompt_indices) * len(methods)
    OUTPUT_ROOT.mkdir(exist_ok=True)

    if not args.eval_only:
        # Init DiT
        print("=" * 60)
        print("Initializing DiT...")
        dit_handler = AceStepHandler()
        dit_status, dit_success = dit_handler.initialize_service(
            project_root=str(MODEL_ROOT), config_path="acestep-v15-sft",
            device="cuda", use_flash_attention=False, compile_model=False, offload_to_cpu=False,
        )
        if not dit_success: print(f"DiT init failed: {dit_status}"); sys.exit(1)
        model = dit_handler.model
        device = next(model.parameters()).device
        model.config.use_section_rope_offset = False
        for layer in model.decoder.layers:
            if getattr(layer, "use_section_rope", False): layer.use_section_rope = False
            if getattr(layer, "use_phase_memory", False): layer.use_phase_memory = False

        # Init LLM
        print("Initializing LLM...")
        llm_handler = LLMHandler()
        llm_success = llm_handler.initialize(
            checkpoint_dir=str(MODEL_ROOT), lm_model_path="acestep-5Hz-lm-1.7B",
            backend="pt", device="cuda",
        )
        if not llm_success: print("LLM init failed"); sys.exit(1)
        print("All models ready\n")

        count = 0
        for pi in prompt_indices:
            pname, caption, lyrics = PROMPTS[pi]

            for method in methods:
                count += 1
                out_dir = OUTPUT_ROOT / f"{pname}_{method}"
                out_dir.mkdir(parents=True, exist_ok=True)

                existing = list(out_dir.glob("*.flac"))
                if existing:
                    print(f"[{count}/{total}] SKIP {pname}/{method} — already exists")
                    continue

                print(f"\n[{count}/{total}] {pname} / {method} (seed={seed})")

                # Restore original prepare_condition
                model.prepare_condition = model.__class__.prepare_condition.__get__(model, model.__class__)

                if method != "baseline":
                    setup_retrieval(model, lyrics, int(duration * 25), method == "sinkhorn_pmgate", device)

                params = GenerationParams(
                    task_type="text2music", caption=caption, lyrics=lyrics,
                    instrumental=False, bpm=120, keyscale="C major",
                    timesignature="4", vocal_language="en", duration=duration,
                    inference_steps=50, guidance_scale=7.0, seed=seed,
                    thinking=True, use_cot_metas=True, use_cot_caption=True,
                    lm_temperature=0.75,
                )
                gen_config = GenerationConfig(
                    batch_size=1, audio_format="flac", use_random_seed=False, seeds=[seed],
                )

                t0 = time.time()
                result = generate_music(dit_handler=dit_handler, llm_handler=llm_handler,
                                         params=params, config=gen_config, save_dir=str(out_dir))
                elapsed = time.time() - t0
                if result.success:
                    print(f"  ✓ {elapsed:.0f}s → {result.audios[0]['path']}")
                else:
                    print(f"  ✗ FAILED: {result.error}")

    # ===== SongEval =====
    print("\n" + "=" * 60)
    print("Running SongEval on all outputs...")
    songeval_dir = ACE_STEP_ROOT / "SongEval"

    all_results = {}
    for pi in prompt_indices:
        pname, _, _ = PROMPTS[pi]
        for method in methods:
            key = f"{pname}_{method}"
            out_dir = OUTPUT_ROOT / key
            flacs = list(out_dir.glob("*.flac"))
            if not flacs:
                print(f"  No audio for {key}, skipping")
                continue
            audio = flacs[0]
            eval_out = out_dir / "songeval"
            eval_out.mkdir(exist_ok=True)

            ret = os.system(f"cd {songeval_dir} && python eval.py -i {audio} -o {eval_out} 2>/dev/null")
            res_file = eval_out / "result.json"
            if res_file.exists():
                with open(res_file) as f:
                    all_results[key] = json.load(f)
                fid = list(all_results[key].keys())[0]
                s = all_results[key][fid]
                print(f"  {key}: Coherence={s['Coherence']:.4f} Musicality={s['Musicality']:.4f} "
                      f"Memorability={s['Memorability']:.4f} Clarity={s['Clarity']:.4f} Naturalness={s['Naturalness']:.4f}")
            else:
                print(f"  {key}: SongEval failed")

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    header = f"{'Name':<28} {'Coherence':>10} {'Musicality':>10} {'Memorability':>12} {'Clarity':>10} {'Naturalness':>12}"
    print(header)
    print("-" * 90)
    for key in sorted(all_results.keys()):
        data = all_results[key]
        for fid, scores in data.items():
            print(f"{key:<28} {scores['Coherence']:>10.4f} {scores['Musicality']:>10.4f} "
                  f"{scores['Memorability']:>12.4f} {scores['Clarity']:>10.4f} {scores['Naturalness']:>12.4f}")

    # Average per method
    print("\n" + "-" * 90)
    print("AVERAGE PER METHOD")
    print("-" * 90)
    for method in methods:
        method_keys = [k for k in all_results if k.endswith(f"_{method}")]
        if not method_keys:
            continue
        avg = {"Coherence": [], "Musicality": [], "Memorability": [], "Clarity": [], "Naturalness": []}
        for k in method_keys:
            for fid, s in all_results[k].items():
                for m in avg:
                    avg[m].append(s[m])
        avg = {m: sum(v)/len(v) if v else 0 for m, v in avg.items()}
        print(f"{method:<28} {avg['Coherence']:>10.4f} {avg['Musicality']:>10.4f} "
              f"{avg['Memorability']:>12.4f} {avg['Clarity']:>10.4f} {avg['Naturalness']:>12.4f}")


if __name__ == "__main__":
    main()
