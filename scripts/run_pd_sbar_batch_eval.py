#!/usr/bin/env python3
"""
PD-SBAR Batch Evaluation: 10 prompts × baseline + PD-SBAR → SongEval
Each generation in subprocess to avoid CUDA state leakage.
"""
import subprocess, sys, json, os, time, glob, argparse
from pathlib import Path

ACE_ROOT = Path(__file__).resolve().parent.parent

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

METHODS = [
    ("baseline", None),
    ("pd_sbar", None),                       # inference-only
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                        default="/root/autodl-tmp/pd_sbar_eval_output")
    parser.add_argument("--duration", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pd-sbar-ckpt", type=str, default=None)
    parser.add_argument("--skip-gen", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # ---- Phase 1: Generate ------------------------------------------------
    if not args.skip_gen:
        print("=" * 70)
        print(f"  Generating {len(PROMPTS)} prompts × {len(METHODS)} methods")
        print(f"  Duration: {args.duration}s, Seed: {args.seed}")
        print("=" * 70)
        for method_name, ckpt in METHODS:
            aud_dir = output_dir / method_name / "audios"
            aud_dir.mkdir(parents=True, exist_ok=True)
            for name, caption, lyrics in PROMPTS:
                out_path = aud_dir / f"{name}.flac"
                if out_path.exists():
                    print(f"  [SKIP] {method_name}/{name}.flac exists")
                    continue
                print(f"\n  [{method_name}] {name} ...", end=" ", flush=True)
                t0 = time.time()
                cmd = [
                    sys.executable, str(ACE_ROOT / "gen_one_pd_sbar.py"),
                    "--method", method_name,
                    "--caption", caption,
                    "--lyrics", lyrics,
                    "--duration", str(args.duration),
                    "--seed", str(args.seed),
                    "--output-dir", str(aud_dir),
                    "--name", name,
                ]
                if method_name == "pd_sbar_trained" and args.pd_sbar_ckpt:
                    cmd += ["--pd-sbar-ckpt", args.pd_sbar_ckpt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                elapsed = time.time() - t0
                if result.returncode == 0 and ("SUCCESS" in result.stdout or out_path.exists()):
                    print(f"OK ({elapsed:.0f}s)")
                else:
                    print(f"FAIL ({elapsed:.0f}s)")
                    print(f"  stderr: {result.stderr[-300:]}" if result.stderr else "  no stderr")
        print("\n  Generation complete!")

    # ---- Phase 2: SongEval ------------------------------------------------
    print("\n" + "=" * 70)
    print("  SongEval Evaluation")
    print("=" * 70)
    import librosa
    import numpy as np
    from muq import MuQ
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    METRICS = ['Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness']

    config = OmegaConf.load(str(ACE_ROOT / "SongEval" / "config.yaml"))
    model = instantiate(config.generator).to(torch_device).eval()
    model.load_state_dict(load_file(str(ACE_ROOT / "SongEval" / "ckpt" / "model.safetensors"), device="cpu"), strict=False)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(torch_device).eval()

    songeval_results = {}
    for method_name, _ in METHODS:
        aud_dir = output_dir / method_name / "audios"
        files = sorted(glob.glob(f"{aud_dir}/*.flac") + glob.glob(f"{aud_dir}/*.wav"))
        if not files:
            print(f"  [SKIP] {method_name} — no audio files")
            continue
        all_scores = {m: [] for m in METRICS}
        per_file = {}
        for fpath in files:
            fname = Path(fpath).stem
            try:
                wav, sr = librosa.load(fpath, sr=24000)
                wav_t = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(torch_device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    emb = muq(wav_t)
                    scores = model(emb)[0]
                scores = scores.float().cpu().numpy().flatten()
                entry = {m: round(float(scores[i]), 4) for i, m in enumerate(METRICS)}
                per_file[fname] = entry
                for i, m in enumerate(METRICS):
                    all_scores[m].append(float(scores[i]))
            except Exception as e:
                print(f"    SongEval error {fname}: {e}")
        avg = {m: round(float(np.mean(all_scores[m])), 4) for m in METRICS if all_scores[m]}
        songeval_results[method_name] = {"average": avg, "per_file": per_file, "count": len(files)}
        out_path = output_dir / method_name / "songeval_results.json"
        out_path.write_text(json.dumps(songeval_results[method_name], indent=2))
        print(f"  {method_name}: " + " | ".join(f"{m}={avg.get(m, 0):.4f}" for m in METRICS))

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SongEval Summary")
    print("=" * 70)
    header = f"{'Method':<25}" + "".join(f" {m:>12}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for method_name, _ in METHODS:
        r = songeval_results.get(method_name, {})
        avg = r.get("average", {})
        row = f"{method_name:<25}"
        for m in METRICS:
            row += f" {avg.get(m, 0):>12.4f}"
        print(row)
    print("=" * 70)

    # Save all
    (output_dir / "all_results.json").write_text(json.dumps(songeval_results, indent=2, default=str))
    print(f"\n  All results: {output_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
