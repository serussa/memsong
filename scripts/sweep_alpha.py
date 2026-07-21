#!/usr/bin/env python3
"""Quick α sweep: 3 prompts × 4 values → generate + SongEval."""
import subprocess, sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALPHAS = [0.05, 0.1, 0.2, 0.3]
PROMPTS = [
    ("pop_girl", "pop, female vocal, catchy melody, piano, drums, 120 bpm, C major",
     "[INTRO]\n\n[VERSE]\nWalking through the city lights\nEvery shadow comes alive\nI can feel the rhythm grow\nLetting all my feelings show\n\n[CHORUS]\nDancing in the neon glow\nWhere the music takes control\nTonight we're gonna lose control\nLet the melody unfold\n\n[VERSE]\nStars are shining from above\nEvery heartbeat sings of love\nMoving to the endless beat\nFeel the world beneath my feet\n\n[CHORUS]\nDancing in the neon glow\nWhere the music takes control\nTonight we're gonna lose control\nLet the melody unfold\n\n[BRIDGE]\nHigher and higher we go\nLetting the music flow\n\n[CHORUS]\nDancing in the neon glow\nWhere the music takes control\nTonight we're gonna lose control\nLet the melody unfold\n\n[OUTRO]"),
    ("rock_ballad", "rock, electric guitar, drums, bass, powerful male vocal, 140 bpm, G minor",
     "[INTRO]\n\n[VERSE]\nThe walls are closing in again\nI've been fighting long my friend\nEvery scar upon my skin\nTells a story deep within\n\n[CHORUS]\nI will rise up from the ashes\nBreaking through these iron chains\nNothing's ever gonna stop me\nI will stand up in the rain\n\n[VERSE]\nDarkest hour comes before the dawn\nEverything I love is gone\nBut I'm still standing here tonight\nReady for the final fight\n\n[CHORUS]\nI will rise up from the ashes\nBreaking through these iron chains\nNothing's ever gonna stop me\nI will stand up in the rain\n\n[BRIDGE]\nThis is my redemption song\nNothing here can go wrong\n\n[CHORUS]\nI will rise up from the ashes\nBreaking through these iron chains\nNothing's ever gonna stop me\nI will stand up in the rain\n\n[OUTRO]"),
    ("r_and_b", "r&b, soulful, male vocal, smooth, synth pad, 808 drums, 90 bpm, D minor",
     "[INTRO]\n\n[VERSE]\nLate night calls and empty streets\nThinking 'bout the heartbeats\nThat we shared under the moon\nWishing I could see you soon\n\n[CHORUS]\nBaby you're the only one\nShining brighter than the sun\nEvery moment feels so right\nWhen I hold you through the night\n\n[VERSE]\nSilk sheets and candlelight\nEverything feels so right\nYour body moving close to mine\nLosing track of space and time\n\n[CHORUS]\nBaby you're the only one\nShining brighter than the sun\nEvery moment feels so right\nWhen I hold you through the night\n\n[BRIDGE]\nLet's stay here forever\nWe'll be young together\n\n[CHORUS]\nBaby you're the only one\nShining brighter than the sun\nEvery moment feels so right\nWhen I hold you through the night\n\n[OUTRO]"),
]

OUT = Path("/root/autodl-tmp/pd_sbar_sweep")
SEED = 42; DURATION = 150
GEN = "gen_one_pd_sbar.py"

# Generate
for name, caption, lyrics in PROMPTS:
    for alpha in ALPHAS:
        aud_dir = OUT / f"a{alpha:.2f}" / "audios"
        aud_dir.mkdir(parents=True, exist_ok=True)
        out_path = aud_dir / f"{name}.flac"
        if out_path.exists():
            print(f"[SKIP] α={alpha:.2f} {name}")
            continue
        print(f"[GEN] α={alpha:.2f} {name} ...", end=" ", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, GEN, "--method", "pd_sbar",
             "--caption", caption, "--lyrics", lyrics,
             "--duration", str(DURATION), "--seed", str(SEED),
             "--output-dir", str(aud_dir), "--name", name,
             "--alpha", str(alpha)],
            capture_output=True, text=True, timeout=600,
        )
        el = time.time() - t0
        if r.returncode == 0 and "SUCCESS" in r.stdout:
            print(f"OK ({el:.0f}s)")
        else:
            print(f"FAIL ({el:.0f}s)")
            print(f"  {r.stderr[-200:]}" if r.stderr else "  no stderr")
        sys.stdout.flush()

# SongEval
print("\n=== SongEval ===")
METRICS = ["Coherence","Musicality","Memorability","Clarity","Naturalness"]
results = {}
for alpha in ALPHAS:
    aud_dir = OUT / f"a{alpha:.2f}" / "audios"
    eval_out = OUT / f"a{alpha:.2f}" / "songeval"
    eval_out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["/root/miniconda3/envs/musicgen/bin/python",
         "/root/ACE-Step-1.5/SongEval/eval.py",
         "-i", str(aud_dir), "-o", str(eval_out)],
        cwd="/root/ACE-Step-1.5/SongEval", capture_output=True, timeout=120,
    )
    if (eval_out / "result.json").exists():
        scores = json.loads((eval_out / "result.json").read_text())
        avg = {}
        for m in METRICS:
            vals = [s[m] for s in scores.values()]
            avg[m] = sum(vals)/len(vals)
        results[f"α={alpha:.2f}"] = avg

# Table
print(f"\n{'α':<10}" + "".join(f"{m:>12}" for m in METRICS))
print("-" * 70)
for k, v in results.items():
    print(f"{k:<10}" + "".join(f"{v.get(m,0):>12.4f}" for m in METRICS))
