#!/root/miniconda3/envs/musicgen/bin/python
"""Run eval: each generation in a fresh subprocess."""
import subprocess, sys, json, glob, os, shutil
from pathlib import Path

OUTPUT_ROOT = Path("/root/autodl-tmp/eval_after_train")
CKPT = "/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt"
GEN_SCRIPT = "/root/ACE-Step-1.5/gen_one.py"

# Generate script for a single run
GENERATOR = r"""#!/root/miniconda3/envs/musicgen/bin/python
import sys, os
sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ['ACESTEP_OFFLINE'] = '1'
os.environ['ACESTEP_MINIMAL_COMPONENTS'] = '1'
os.environ['SIDESTEP_SAFE_ROOT'] = '/root/autodl-tmp'
import torch, pickle
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

METHOD, CKPT, CAPTION, LYRICS, OUT_DIR, SEED, DURATION = pickle.load(sys.stdin.buffer)

dt = AceStepHandler()
dt.initialize_service(project_root='/root/autodl-tmp/Ace-Step1.5', config_path='acestep-v15-sft',
    device='cuda', use_flash_attention=False, compile_model=False, offload_to_cpu=False)
model = dt.model.eval(); device = next(model.parameters()).device
model.config.use_section_rope_offset = False
for l in model.decoder.layers:
    if getattr(l, 'use_section_rope', False): l.use_section_rope = False
    if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False
llm = LLMHandler()
llm.initialize(checkpoint_dir='/root/autodl-tmp/Ace-Step1.5', lm_model_path='acestep-5Hz-lm-1.7B', backend='pt', device='cuda')

if METHOD != 'baseline':
    from acestep.phase_memory import TransportRetrievalAdapter, PMRetrievalPhaseMemory, parse_lyrics_to_units, build_duration_scaffold
    from acestep.tgca.lyrics_parser import LyricsStructureParser
    D = model.config.hidden_size
    pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True).to(device).float()
    adapt = TransportRetrievalAdapter(hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
        transport_mode='sinkhorn', sinkhorn_iters=10, transport_sigma=0.18, scoring_mode='position_only',
        use_pm_gate=False, gate_hidden_dim=128, write_alpha_init=0.005, write_alpha_max=0.01, out_proj_init_std=0.01).to(device).float()
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
    pm.load_state_dict(ckpt['phase_memory']); adapt.load_state_dict(ckpt['retrieval_adapter']); adapt.eval(); pm.eval()
    T_eff = int(DURATION * 25); hook_holder = [None]; orig = model.prepare_condition

    def patched(*a, **kw):
        r = orig(*a, **kw)
        if r[0] is not None and hook_holder[0] is None:
            eh = r[0]; L = eh.shape[1]; B, D_h = eh.shape[0], eh.shape[-1]
            parser = LyricsStructureParser(); ps = parser.parse(LYRICS, num_chunks=L)
            units, _, debug = parse_lyrics_to_units(LYRICS, ps.section_type_ids, auto_transition_ratios={})
            sc = build_duration_scaffold(units, text_len=L, tag_control_mask=debug.get('tag_control_mask'))
            sc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sc.items()}
            U = len(sc['unit_boundaries']) - 1
            ca = (sc['unit_boundaries'][:-1] + sc['unit_boundaries'][1:]) / 2
            mu = sc['unit_duration'] / sc['unit_duration'].sum()
            usid = sc['unit_section_ids']; uil = sc['lyric_unit_mask']; t2u = sc['token_to_unit']; lym = sc['lyric_mask']
            eh_f = eh.float(); uth = []
            for uid in range(U):
                tm = (t2u == uid) & lym if uil[uid].item() else (t2u == uid)
                tmb = tm.unsqueeze(0).expand(B, -1)
                uth.append(eh_f[tmb].view(B, -1, D_h).mean(dim=1) if tmb.any() else torch.zeros(B, D_h, device=device))
            uth = torch.stack(uth, dim=1); pa = torch.linspace(0, 1, T_eff, device=device).unsqueeze(0)
            def hook(_m, _i, o):
                Ho = o[0]; B, T = Ho.shape[0], Ho.shape[1]; pp = pa[:, :T]
                if B > pp.shape[0]: pp = pp.expand(B, -1).contiguous()
                with torch.no_grad():
                    ps2 = pm(Ho.float(), torch.zeros(B, device=device))
                    dh, Pi, diag = adapt(Ho.float(), None, ps2, pp.float(), uth.float(),
                        ca.unsqueeze(0).float(), mu.unsqueeze(0).float(), usid.unsqueeze(0), uil.unsqueeze(0))
                return (Ho + dh.to(dtype=Ho.dtype), *o[1:])
            hook_holder[0] = model.decoder.layers[12].register_forward_hook(hook)
        return r
    model.prepare_condition = patched

params = GenerationParams(task_type='text2music', caption=CAPTION, lyrics=LYRICS,
    instrumental=False, bpm=120, keyscale='C major', timesignature='4',
    vocal_language='zh', duration=DURATION, inference_steps=50, guidance_scale=7.0,
    seed=SEED, thinking=True, use_cot_metas=True, use_cot_caption=True, lm_temperature=0.75)
config = GenerationConfig(batch_size=1, audio_format='flac', use_random_seed=False, seeds=[SEED])
result = generate_music(dit_handler=dt, llm_handler=llm, params=params, config=config, save_dir=OUT_DIR)
print('OK:' + (result.audios[0]['path'] if result.audios else 'no_audio'))
"""

with open(GEN_SCRIPT, 'w') as f:
    f.write(GENERATOR)
os.chmod(GEN_SCRIPT, 0o755)

PROMPTS = [
    ("love_story", "pop, female vocal, piano, guitar, drums, romantic, 120 bpm, C major",
     "[INTRO]\n\n[VERSE]\n在晨曦中我看到你\n你的笑容如花般绽放\n时光静止在那一刻\n心跳轻轻回响\n\n[CHORUS]\n你就是我心中最美的旋律\n在每一个瞬间陪伴我\n我们的故事如歌般动人\n在永恒时光里绽放\n\n[VERSE]\n当夜幕降临星空闪烁\n我在梦中依然与你相拥\n每个瞬间都是无价\n你的眼神是唯一的灯塔\n\n[CHORUS]\n你就是我心中最美的旋律\n在每一个瞬间陪伴我\n我们的故事如歌般动人\n在永恒时光里绽放\n\n[BRIDGE]\n岁月如歌声声入耳\n只愿与你一同走过\n\n[CHORUS]\n你就是我心中最美的旋律\n在每一个瞬间陪伴我\n我们的故事如歌般动人\n在永恒时光里绽放\n\n[OUTRO]"),
    ("dream_journey", "pop, dreamy, female vocal, synth, pad, gentle drums, 100 bpm, D major",
     "[INTRO]\n\n[VERSE]\n在晨光中我醒来\n回忆昨夜的梦境\n星河带我去未知的世界\n寻找心中的声音\n\n[CHORUS]\n在世界的每个角落\n找回失去的光辉\n跟随梦想飞翔在天际\n让爱指引方向\n\n[VERSE]\n在繁华中我独自漫步\n人潮涌动如浮云散去\n我在寻找一个出口\n通往内心深处的宁静\n\n[CHORUS]\n在世界的每个角落\n找回失去的光辉\n跟随梦想飞翔在天际\n让爱指引方向\n\n[BRIDGE]\n无论黑暗如何吞噬\n我心中的光永不熄灭\n\n[CHORUS]\n在世界的每个角落\n找回失去的光辉\n跟随梦想飞翔在天际\n让爱指引方向\n\n[OUTRO]"),
    ("city_night", "pop, electronic, urban, female vocal, synth, beat, 128 bpm, A minor",
     "[INTRO]\n\n[VERSE]\n夜幕降临城市心跳\n灯光闪烁如梦如幻\n走在这铁和水的交响\n耳边低语历史的回响\n\n[CHORUS]\n我们在虚空遨游追寻声音\n遗忘的真相在呼唤\n每个灵魂都有它的节奏\n在这平行宇宙中飞翔\n\n[VERSE]\n一路狂奔追逐时间\n碎片拼出未来的蓝图\n黎明来临燃起希望\n无惧风暴继续航行\n\n[CHORUS]\n我们在虚空遨游追寻声音\n遗忘的真相在呼唤\n每个灵魂都有它的节奏\n在这平行宇宙中飞翔\n\n[BRIDGE]\n每一次挣扎让我更坚强\n心中的火焰永不熄灭\n\n[CHORUS]\n我们在虚空遨游追寻声音\n遗忘的真相在呼唤\n每个灵魂都有它的节奏\n在这平行宇宙中飞翔\n\n[OUTRO]"),
    ("sunshine", "pop, cheerful, female vocal, ukulele, guitar, happy, 130 bpm, G major",
     "[INTRO]\n\n[VERSE]\n阳光洒满整个午後\n微风吹过你的笑容\n每一天都像新的开始\n世界因你而不同\n\n[CHORUS]\n你是我的阳光照亮每一天\n把所有的阴霾都驱散\n牵着手一起走不管多远\n有你在身边就是晴天\n\n[VERSE]\n彩虹出现在雨後\n鸟儿在枝头唱歌\n生活本就如此简单\n快乐就在你我心间\n\n[CHORUS]\n你是我的阳光照亮每一天\n把所有的阴霾都驱散\n牵着手一起走不管多远\n有你在身边就是晴天\n\n[BRIDGE]\n一起笑一起闹\n这就是最好的时光\n\n[CHORUS]\n你是我的阳光照亮每一天\n把所有的阴霾都驱散\n牵着手一起走不管多远\n有你在身边就是晴天\n\n[OUTRO]"),
    ("parting", "pop, ballad, sad, female vocal, piano, strings, slow, 75 bpm, E minor",
     "[INTRO]\n\n[VERSE]\n爱总忽然退潮心慌乱触礁\n沉没在深海里看海面闪耀\n回忆像水草紧紧的缠绕\n梦才温热眼角就冰冷掉\n\n[CHORUS]\n你手心的太阳只轻放在我背上\n委屈就能笑着落泪被释放\n在手心的太阳黑暗里特别明亮\n让远路好像是一种分享而不是漫长\n\n[VERSE]\n努力越过风暴向着未来飘\n我们才会遇到感动的拥抱\n你总是能知道我的坚强剩多少\n给我最刚好的依靠\n\n[CHORUS]\n你手心的太阳只轻放在我背上\n委屈就能笑着落泪被释放\n在手心的太阳黑暗里特别明亮\n让远路好像是一种分享而不是漫长\n\n[BRIDGE]\n就算世界再乱我也不心慌\n我手心的太阳或许只像个月亮\n\n[CHORUS]\n你手心的太阳有种安定的力量\n就算世界再乱我也不心慌\n我手心的太阳或许只像个月亮\n却用所有爱为你投射我最暖的光芒\n\n[OUTRO]"),
    ("youth", "pop, rock, energetic, male vocal, electric guitar, drums, 140 bpm, C major",
     "[INTRO]\n\n[VERSE]\n年少轻狂的我们\n追逐着各自的梦\n不怕跌倒不怕痛\n因为青春就是资本\n\n[CHORUS]\n燃烧吧青春像烈火一样\n让梦想在天空中翱翔\n不管前方有多少风浪\n我们都要勇敢去闯\n\n[VERSE]\n时光匆匆不停留\n但我们不会回头\n用热血写下的歌\n会一直唱到最后\n\n[CHORUS]\n燃烧吧青春像烈火一样\n让梦想在天空中翱翔\n不管前方有多少风浪\n我们都要勇敢去闯\n\n[BRIDGE]\n这就是我们的时代\n没有什么能阻挡\n\n[CHORUS]\n燃烧吧青春像烈火一样\n让梦想在天空中翱翔\n不管前方有多少风浪\n我们都要勇敢去闯\n\n[OUTRO]"),
    ("moonlight", "chinese traditional, guzheng, erhu, gentle, poetic, 90 bpm, G major",
     "[INTRO]\n\n[VERSE]\n明月几时有把酒问青天\n不知天上宫阙今夕是何年\n我欲乘风归去又恐琼楼玉宇\n高处不胜寒起舞弄清影\n\n[CHORUS]\n人有悲欢离合月有阴晴圆缺\n此事古难全但愿人长久\n千里共婵娟\n\n[VERSE]\n转朱阁低绮户照无眠\n不应有恨何事长向别时圆\n人有悲欢离合月有阴晴圆缺\n\n[CHORUS]\n人有悲欢离合月有阴晴圆缺\n此事古难全但愿人长久\n千里共婵娟\n\n[BRIDGE]\n但愿人长久\n千里共婵娟\n\n[CHORUS]\n人有悲欢离合月有阴晴圆缺\n此事古难全但愿人长久\n千里共婵娟\n\n[OUTRO]"),
    ("rain", "pop, sad, male vocal, piano, acoustic guitar, melancholic, 85 bpm, D minor",
     "[INTRO]\n\n[VERSE]\n下雨的夜晚\n想起你的脸\n窗外的雨滴\n敲打着思念\n\n[CHORUS]\ni miss you every day\n你不在我身边\n雨中的城市\n模糊了视线\ni miss you every night\n回忆在蔓延\n就让这场雨\n带走我的思念\n\n[VERSE]\n伞下的空间\n只剩下孤单\n走过的街道\n都是你影子\n\n[CHORUS]\ni miss you every day\n你不在我身边\n雨中的城市\n模糊了视线\ni miss you every night\n回忆在蔓延\n就让这场雨\n带走我的思念\n\n[BRIDGE]\n雨过天晴后\n你会不会回来\n\n[CHORUS]\ni miss you every day\n你不在我身边\n雨中的城市\n模糊了视线\ni miss you every night\n回忆在蔓延\n就让这场雨\n带走我的思念\n\n[OUTRO]"),
    ("hero", "pop, rock, cinematic, male vocal, orchestra, drums, epic, 120 bpm, C minor",
     "[INTRO]\n\n[VERSE]\n逆着风向前走\n不回头不低头\n就算世界都沉默\n我也要坚持到最后\n\n[CHORUS]\n我就是我不一样的烟火\n天空海阔做最坚强的泡沫\n也许有一天我会倒下\n但我的歌会一直唱下去\n\n[VERSE]\n跌倒了爬起来\n擦干泪继续走\n梦想就在前方\n我不能就这样放弃\n\n[CHORUS]\n我就是我不一样的烟火\n天空海阔做最坚强的泡沫\n也许有一天我会倒下\n但我的歌会一直唱下去\n\n[BRIDGE]\n这是属于我的舞台\n我要活出自己的精彩\n\n[CHORUS]\n我就是我不一样的烟火\n天空海阔做最坚强的泡沫\n也许有一天我会倒下\n但我的歌会一直唱下去\n\n[OUTRO]"),
    ("spring", "pop, folk, female vocal, guitar, flute, cheerful, 115 bpm, A major",
     "[INTRO]\n\n[VERSE]\n春天来了花儿开了\n鸟儿在枝头唱歌\n微风吹过田野\n带来了泥土的芬芳\n\n[CHORUS]\n春天在哪里呀春天在哪里\n春天就在小朋友的眼睛里\n这里有红花呀这里有绿草\n还有那会唱歌的小黄鹂\n\n[VERSE]\n冰雪融化小溪流淌\n大地换上了新装\n蝴蝶在花丛中飞舞\n一切都是那么美好\n\n[CHORUS]\n春天在哪里呀春天在哪里\n春天就在小朋友的眼睛里\n这里有红花呀这里有绿草\n还有那会唱歌的小黄鹂\n\n[BRIDGE]\n啦...\n春天在每一个人的心里\n\n[CHORUS]\n春天在哪里呀春天在哪里\n春天就在小朋友的眼睛里\n这里有红花呀这里有绿草\n还有那会唱歌的小黄鹂\n\n[OUTRO]"),
]

SEED = 42
DURATION = 150

if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)
OUTPUT_ROOT.mkdir(parents=True)

total = len(PROMPTS) * 2
count = 0

for pname, caption, lyrics in PROMPTS:
    for method in ("baseline", "transport_only"):
        count += 1
        out_dir = str(OUTPUT_ROOT / f"{pname}_{method}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[{count}/{total}] {pname}/{method} ...", end=" ", flush=True)

        import pickle as pk
        payload = pk.dumps((method, CKPT, caption, lyrics, out_dir, SEED, DURATION))
        result = subprocess.run(
            [GEN_SCRIPT],
            input=payload,
            capture_output=True,
            timeout=600,
        )
        last_line = result.stdout.decode().strip().split("\n")[-1] if result.stdout else ""
        if last_line.startswith("OK:"):
            print("✓")
        else:
            err = result.stderr.decode().strip()[:80] if result.stderr else last_line[:80]
            print(f"✗ {err}")

# SongEval
print("\n=== SongEval ===")
for d in sorted(OUTPUT_ROOT.iterdir()):
    if not d.is_dir(): continue
    audio = list(d.glob("*.flac"))
    if not audio: continue
    eval_out = d / "songeval"
    eval_out.mkdir(exist_ok=True)
    subprocess.run(
        ["/root/miniconda3/envs/musicgen/bin/python", "/root/ACE-Step-1.5/SongEval/eval.py",
         "-i", str(audio[0]), "-o", str(eval_out)],
        capture_output=True, timeout=120,
    )
    if (eval_out / "result.json").exists():
        with open(eval_out / "result.json") as f:
            s = list(json.load(f).values())[0]
        print(f"  {d.name:<30} Co={s['Coherence']:.4f} Mu={s['Musicality']:.4f} Me={s['Memorability']:.4f} Cl={s['Clarity']:.4f} Na={s['Naturalness']:.4f}")

# Summary
print("\n" + "=" * 74)
print("AVERAGES")
print("=" * 74)
metrics = ["Coherence","Musicality","Memorability","Clarity","Naturalness"]
bl = {m:[] for m in metrics}; tr = {m:[] for m in metrics}
for d in OUTPUT_ROOT.iterdir():
    if not d.is_dir(): continue
    rp = d / "songeval" / "result.json"
    if not rp.exists(): continue
    with open(rp) as f: s = list(json.load(f).values())[0]
    n = d.name
    if n.endswith("_baseline"):
        for m in metrics: bl[m].append(s[m])
    elif n.endswith("_transport_only"):
        for m in metrics: tr[m].append(s[m])

for name, data in [("baseline", bl), ("transport_only", tr)]:
    avg = {m: sum(v)/len(v) if v else 0 for m, v in data.items()}
    print(f"{name:<20} {avg['Coherence']:>10.4f} {avg['Musicality']:>10.4f} {avg['Memorability']:>12.4f} {avg['Clarity']:>10.4f} {avg['Naturalness']:>12.4f}")
