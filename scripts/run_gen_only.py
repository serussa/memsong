#!/usr/bin/env python3
"""Generate 10 prompts × baseline + PD-SBAR, one-by-one in subprocess."""
import subprocess, sys, time, os, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

OUTPUT_DIR = Path("/root/autodl-tmp/pd_sbar_eval_output")
SEED = 42
DURATION = 150

def gen(method):
    aud_dir = OUTPUT_DIR / method / "audios"
    aud_dir.mkdir(parents=True, exist_ok=True)
    for name, caption, lyrics in PROMPTS:
        out_path = aud_dir / f"{name}.flac"
        if out_path.exists():
            print(f"  [SKIP] {method}/{name}")
            continue
        print(f"  [{method}] {name} ...", end=" ", flush=True)
        t0 = time.time()
        cmd = [
            sys.executable, "gen_one_pd_sbar.py",
            "--method", method,
            "--caption", caption,
            "--lyrics", lyrics,
            "--duration", str(DURATION),
            "--seed", str(SEED),
            "--output-dir", str(aud_dir),
            "--name", name,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0
        if r.returncode == 0 and ("SUCCESS" in r.stdout or out_path.exists()):
            print(f"OK ({elapsed:.0f}s)")
        else:
            print(f"FAIL ({elapsed:.0f}s)")
            err = r.stderr[-500:] if r.stderr else "(no stderr)"
            print(f"  {err}")
        sys.stdout.flush()

if __name__ == "__main__":
    print("=" * 70)
    print("  Generating 10 prompts × 2 methods")
    print(f"  Duration: {DURATION}s, Seed: {SEED}")
    print("=" * 70)

    for method in ["baseline", "pd_sbar"]:
        print(f"\n--- Method: {method} ---")
        gen(method)

    print("\nDone!")
    for method in ["baseline", "pd_sbar"]:
        aud_dir = OUTPUT_DIR / method / "audios"
        count = len(list(aud_dir.glob("*.flac")))
        print(f"  {method}: {count} files")
