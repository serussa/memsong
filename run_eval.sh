#!/bin/bash
set -e
OUTPUT_ROOT="/root/autodl-tmp/eval_after_train"
CKPT="/root/autodl-tmp/exp_sinkhorn_pmgate/checkpoints/best_loss/pm_retrieval.pt"
DURATION=150
SEED=42

rm -rf "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

# 10 prompts with caption, lyrics
PROMPTS=(
  "love_story|pop, female vocal, piano, guitar, drums, romantic, 120 bpm, C major|$(cat << 'LYRICS'
[INTRO]

[VERSE]
在晨曦中我看到你
你的笑容如花般绽放
时光静止在那一刻
心跳轻轻回响

[CHORUS]
你就是我心中最美的旋律
在每一个瞬间陪伴我
我们的故事如歌般动人
在永恒时光里绽放

[VERSE]
当夜幕降临星空闪烁
我在梦中依然与你相拥
每个瞬间都是无价
你的眼神是唯一的灯塔

[CHORUS]
你就是我心中最美的旋律
在每一个瞬间陪伴我
我们的故事如歌般动人
在永恒时光里绽放

[BRIDGE]
岁月如歌声声入耳
只愿与你一同走过

[CHORUS]
你就是我心中最美的旋律
在每一个瞬间陪伴我
我们的故事如歌般动人
在永恒时光里绽放

[OUTRO]
LYRICS
)"

  "dream_journey|pop, dreamy, female vocal, synth, pad, gentle drums, 100 bpm, D major|$(cat << 'LYRICS'
[INTRO]

[VERSE]
在晨光中我醒来
回忆昨夜的梦境
星河带我去未知的世界
寻找心中的声音

[CHORUS]
在世界的每个角落
找回失去的光辉
跟随梦想飞翔在天际
让爱指引方向

[VERSE]
在繁华中我独自漫步
人潮涌动如浮云散去
我在寻找一个出口
通往内心深处的宁静

[CHORUS]
在世界的每个角落
找回失去的光辉
跟随梦想飞翔在天际
让爱指引方向

[BRIDGE]
无论黑暗如何吞噬
我心中的光永不熄灭

[CHORUS]
在世界的每个角落
找回失去的光辉
跟随梦想飞翔在天际
让爱指引方向

[OUTRO]
LYRICS
)"

  "city_night|pop, electronic, urban, female vocal, synth, beat, 128 bpm, A minor|$(cat << 'LYRICS'
[INTRO]

[VERSE]
夜幕降临城市心跳
灯光闪烁如梦如幻
走在这铁和水的交响
耳边低语历史的回响

[CHORUS]
我们在虚空遨游追寻声音
遗忘的真相在呼唤
每个灵魂都有它的节奏
在这平行宇宙中飞翔

[VERSE]
一路狂奔追逐时间
碎片拼出未来的蓝图
黎明来临燃起希望
无惧风暴继续航行

[CHORUS]
我们在虚空遨游追寻声音
遗忘的真相在呼唤
每个灵魂都有它的节奏
在这平行宇宙中飞翔

[BRIDGE]
每一次挣扎让我更坚强
心中的火焰永不熄灭

[CHORUS]
我们在虚空遨游追寻声音
遗忘的真相在呼唤
每个灵魂都有它的节奏
在这平行宇宙中飞翔

[OUTRO]
LYRICS
)"

  "sunshine|pop, cheerful, female vocal, ukulele, guitar, happy, 130 bpm, G major|$(cat << 'LYRICS'
[INTRO]

[VERSE]
阳光洒满整个午後
微风吹过你的笑容
每一天都像新的开始
世界因你而不同

[CHORUS]
你是我的阳光照亮每一天
把所有的阴霾都驱散
牵着手一起走不管多远
有你在身边就是晴天

[VERSE]
彩虹出现在雨後
鸟儿在枝头唱歌
生活本就如此简单
快乐就在你我心间

[CHORUS]
你是我的阳光照亮每一天
把所有的阴霾都驱散
牵着手一起走不管多远
有你在身边就是晴天

[BRIDGE]
一起笑一起闹
这就是最好的时光

[CHORUS]
你是我的阳光照亮每一天
把所有的阴霾都驱散
牵着手一起走不管多远
有你在身边就是晴天

[OUTRO]
LYRICS
)"

  "parting|pop, ballad, sad, female vocal, piano, strings, slow, 75 bpm, E minor|$(cat << 'LYRICS'
[INTRO]

[VERSE]
爱总忽然退潮心慌乱触礁
沉没在深海里看海面闪耀
回忆像水草紧紧的缠绕
梦才温热眼角就冰冷掉

[CHORUS]
你手心的太阳只轻放在我背上
委屈就能笑着落泪被释放
在手心的太阳黑暗里特别明亮
让远路好像是一种分享而不是漫长

[VERSE]
努力越过风暴向着未来飘
我们才会遇到感动的拥抱
你总是能知道我的坚强剩多少
给我最刚好的依靠

[CHORUS]
你手心的太阳只轻放在我背上
委屈就能笑着落泪被释放
在手心的太阳黑暗里特别明亮
让远路好像是一种分享而不是漫长

[BRIDGE]
就算世界再乱我也不心慌
我手心的太阳或许只像个月亮

[CHORUS]
你手心的太阳有种安定的力量
就算世界再乱我也不心慌
我手心的太阳或许只像个月亮
却用所有爱为你投射我最暖的光芒

[OUTRO]
LYRICS
)"

  "youth|pop, rock, energetic, male vocal, electric guitar, drums, 140 bpm, C major|$(cat << 'LYRICS'
[INTRO]

[VERSE]
年少轻狂的我们
追逐着各自的梦
不怕跌倒不怕痛
因为青春就是资本

[CHORUS]
燃烧吧青春像烈火一样
让梦想在天空中翱翔
不管前方有多少风浪
我们都要勇敢去闯

[VERSE]
时光匆匆不停留
但我们不会回头
用热血写下的歌
会一直唱到最后

[CHORUS]
燃烧吧青春像烈火一样
让梦想在天空中翱翔
不管前方有多少风浪
我们都要勇敢去闯

[BRIDGE]
这就是我们的时代
没有什么能阻挡

[CHORUS]
燃烧吧青春像烈火一样
让梦想在天空中翱翔
不管前方有多少风浪
我们都要勇敢去闯

[OUTRO]
LYRICS
)"

  "moonlight|chinese traditional, guzheng, erhu, gentle, poetic, 90 bpm, G major|$(cat << 'LYRICS'
[INTRO]

[VERSE]
明月几时有把酒问青天
不知天上宫阙今夕是何年
我欲乘风归去又恐琼楼玉宇
高处不胜寒起舞弄清影

[CHORUS]
人有悲欢离合月有阴晴圆缺
此事古难全但愿人长久
千里共婵娟

[VERSE]
转朱阁低绮户照无眠
不应有恨何事长向别时圆
人有悲欢离合月有阴晴圆缺

[CHORUS]
人有悲欢离合月有阴晴圆缺
此事古难全但愿人长久
千里共婵娟

[BRIDGE]
但愿人长久
千里共婵娟

[CHORUS]
人有悲欢离合月有阴晴圆缺
此事古难全但愿人长久
千里共婵娟

[OUTRO]
LYRICS
)"

  "rain|pop, sad, male vocal, piano, acoustic guitar, melancholic, 85 bpm, D minor|$(cat << 'LYRICS'
[INTRO]

[VERSE]
下雨的夜晚
想起你的脸
窗外的雨滴
敲打着思念

[CHORUS]
i miss you every day
你不在我身边
雨中的城市
模糊了视线
i miss you every night
回忆在蔓延
就让这场雨
带走我的思念

[VERSE]
伞下的空间
只剩下孤单
走过的街道
都是你影子

[CHORUS]
i miss you every day
你不在我身边
雨中的城市
模糊了视线
i miss you every night
回忆在蔓延
就让这场雨
带走我的思念

[BRIDGE]
雨过天晴后
你会不会回来

[CHORUS]
i miss you every day
你不在我身边
雨中的城市
模糊了视线
i miss you every night
回忆在蔓延
就让这场雨
带走我的思念

[OUTRO]
LYRICS
)"

  "hero|pop, rock, cinematic, male vocal, orchestra, drums, epic, 120 bpm, C minor|$(cat << 'LYRICS'
[INTRO]

[VERSE]
逆着风向前走
不回头不低头
就算世界都沉默
我也要坚持到最后

[CHORUS]
我就是我不一样的烟火
天空海阔做最坚强的泡沫
也许有一天我会倒下
但我的歌会一直唱下去

[VERSE]
跌倒了爬起来
擦干泪继续走
梦想就在前方
我不能就这样放弃

[CHORUS]
我就是我不一样的烟火
天空海阔做最坚强的泡沫
也许有一天我会倒下
但我的歌会一直唱下去

[BRIDGE]
这是属于我的舞台
我要活出自己的精彩

[CHORUS]
我就是我不一样的烟火
天空海阔做最坚强的泡沫
也许有一天我会倒下
但我的歌会一直唱下去

[OUTRO]
LYRICS
)"

  "spring|pop, folk, female vocal, guitar, flute, cheerful, 115 bpm, A major|$(cat << 'LYRICS'
[INTRO]

[VERSE]
春天来了花儿开了
鸟儿在枝头唱歌
微风吹过田野
带来了泥土的芬芳

[CHORUS]
春天在哪里呀春天在哪里
春天就在小朋友的眼睛里
这里有红花呀这里有绿草
还有那会唱歌的小黄鹂

[VERSE]
冰雪融化小溪流淌
大地换上了新装
蝴蝶在花丛中飞舞
一切都是那么美好

[CHORUS]
春天在哪里呀春天在哪里
春天就在小朋友的眼睛里
这里有红花呀这里有绿草
还有那会唱歌的小黄鹂

[BRIDGE]
啦...
春天在每一个人的心里

[CHORUS]
春天在哪里呀春天在哪里
春天就在小朋友的眼睛里
这里有红花呀这里有绿草
还有那会唱歌的小黄鹂

[OUTRO]
LYRICS
)"
)

TOTAL=$(( ${#PROMPTS[@]} * 2 ))
COUNT=0

for ENTRY in "${PROMPTS[@]}"; do
  PNAME="${ENTRY%%|*}"
  REST="${ENTRY#*|}"
  CAPTION="${REST%%|*}"
  LYRICS="${REST#*|}"
  LYRICS="${LYRICS#*|}"

  for METHOD in "baseline" "transport_only"; do
    COUNT=$((COUNT+1))
    OUT_DIR="$OUTPUT_ROOT/${PNAME}_${METHOD}"
    mkdir -p "$OUT_DIR"

    if ls "$OUT_DIR"/*.flac 2>/dev/null >/dev/null; then
      echo "[$COUNT/$TOTAL] SKIP $PNAME/$METHOD"
      continue
    fi

    echo "[$COUNT/$TOTAL] $PNAME/$METHOD"
    PYTHON_SCRIPT=$(cat << PYSCRIPT
import sys, os
sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ['ACESTEP_OFFLINE'] = '1'
os.environ['ACESTEP_MINIMAL_COMPONENTS'] = '1'
os.environ['SIDESTEP_SAFE_ROOT'] = '/root/autodl-tmp'

import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

MODEL_ROOT = '/root/autodl-tmp/Ace-Step1.5'
CKPT = '$CKPT'

dt = AceStepHandler()
dt.initialize_service(project_root=MODEL_ROOT, config_path='acestep-v15-sft',
    device='cuda', use_flash_attention=False, compile_model=False, offload_to_cpu=False)
model = dt.model.eval(); device = next(model.parameters()).device
model.config.use_section_rope_offset = False
for l in model.decoder.layers:
    if getattr(l, 'use_section_rope', False): l.use_section_rope = False
    if getattr(l, 'use_phase_memory', False): l.use_phase_memory = False

llm = LLMHandler()
llm.initialize(checkpoint_dir=MODEL_ROOT, lm_model_path='acestep-5Hz-lm-1.7B', backend='pt', device='cuda')

lyrics = """$LYRICS"""
D = model.config.hidden_size
PYSCRIPT

    if [ "$METHOD" != "baseline" ]; then
      PYTHON_SCRIPT+=$(cat << 'PYSCRIPT'
from acestep.phase_memory import TransportRetrievalAdapter, PMRetrievalPhaseMemory, parse_lyrics_to_units, build_duration_scaffold
from acestep.tgca.lyrics_parser import LyricsStructureParser
import numpy as np

pm = PMRetrievalPhaseMemory(dim=D, mem_dim=128, hidden_dim=256, normalize_internal_state=True).to(device).float()
adapt = TransportRetrievalAdapter(hidden_dim=D, text_dim=D, pm_dim=256, d_r=256,
    transport_mode='sinkhorn', sinkhorn_iters=10, transport_sigma=0.18, scoring_mode='position_only',
    use_pm_gate=False, gate_hidden_dim=128, write_alpha_init=0.005, write_alpha_max=0.01, out_proj_init_std=0.01).to(device).float()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=True)
pm.load_state_dict(ckpt['phase_memory']); adapt.load_state_dict(ckpt['retrieval_adapter'])
adapt.eval(); pm.eval()

T_eff = int($DURATION * 25)
hook_holder = [None]
orig_prepare = model.prepare_condition

def patched_prepare(*a, **kw):
    r = orig_prepare(*a, **kw)
    if r[0] is not None and hook_holder[0] is None:
        eh = r[0]; L = eh.shape[1]; B, D_h = eh.shape[0], eh.shape[-1]
        parser = LyricsStructureParser()
        ps = parser.parse(lyrics, num_chunks=L)
        units, _, debug = parse_lyrics_to_units(lyrics, ps.section_type_ids, auto_transition_ratios={})
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
        uth = torch.stack(uth, dim=1)
        pa = torch.linspace(0, 1, T_eff, device=device).unsqueeze(0)

        def layer_hook(_m, _i, o):
            Ho = o[0]; B, T = Ho.shape[0], Ho.shape[1]
            pp = pa[:, :T]
            if B > pp.shape[0]: pp = pp.expand(B, -1).contiguous()
            with torch.no_grad():
                ps2 = pm(Ho.float(), torch.zeros(B, device=device))
                dh, Pi, diag = adapt(Ho.float(), None, ps2, pp.float(), uth.float(),
                    ca.unsqueeze(0).float(), mu.unsqueeze(0).float(),
                    usid.unsqueeze(0), uil.unsqueeze(0))
            return (Ho + dh.to(dtype=Ho.dtype), *o[1:])

        hook_holder[0] = model.decoder.layers[12].register_forward_hook(layer_hook)
    return r

model.prepare_condition = patched_prepare
PYSCRIPT
    fi

    PYTHON_SCRIPT+=$(cat << PYSCRIPT

params = GenerationParams(task_type='text2music', caption='$CAPTION', lyrics=lyrics,
    instrumental=False, bpm=120, keyscale='C major', timesignature='4',
    vocal_language='zh', duration=$DURATION, inference_steps=50, guidance_scale=7.0,
    seed=$SEED, thinking=True, use_cot_metas=True, use_cot_caption=True, lm_temperature=0.75)
config = GenerationConfig(batch_size=1, audio_format='flac', use_random_seed=False, seeds=[$SEED])
result = generate_music(dit_handler=dt, llm_handler=llm, params=params, config=config, save_dir='$OUT_DIR')
print('OK:' + (result.audios[0]['path'] if result.audios else 'no_audio'))
if not result.success: print('ERROR:' + str(result.error), file=sys.stderr)
PYSCRIPT
)

    OUTPUT=$(/root/miniconda3/envs/musicgen/bin/python -c "$PYTHON_SCRIPT" 2>&1 | tail -1)
    if echo "$OUTPUT" | grep -q "^OK:"; then
      echo "  ✓"
    else
      echo "  ✗ $OUTPUT"
    fi
  done
done

echo ""
echo "=== SongEval ==="
for d in "$OUTPUT_ROOT"/*/; do
  audio=$(ls "$d"/*.flac 2>/dev/null | head -1)
  [ -z "$audio" ] && continue
  key=$(basename "$d")
  eval_out="$d/songeval"
  mkdir -p "$eval_out"
  /root/miniconda3/envs/musicgen/bin/python /root/ACE-Step-1.5/SongEval/eval.py -i "$audio" -o "$eval_out" 2>/dev/null
  res="$eval_out/result.json"
  if [ -f "$res" ]; then
    scores=$(/root/miniconda3/envs/musicgen/bin/python -c "import json; d=json.load(open('$res')); k=list(d.keys())[0]; s=d[k]; print(f'{s[\"Coherence\"]:.4f} {s[\"Musicality\"]:.4f} {s[\"Memorability\"]:.4f} {s[\"Clarity\"]:.4f} {s[\"Naturalness\"]:.4f}')" 2>/dev/null)
    printf "%-35s %s\n" "$key" "$scores"
  fi
done

echo ""
echo "=== AVERAGES ==="
/root/miniconda3/envs/musicgen/bin/python -c "
import json, glob, os
metrics = ['Coherence','Musicality','Memorability','Clarity','Naturalness']
bl = {m:[] for m in metrics}; tr = {m:[] for m in metrics}
for fp in glob.glob('$OUTPUT_ROOT/*/songeval/result.json'):
    d = os.path.basename(os.path.dirname(os.path.dirname(fp)))
    with open(fp) as f: s = list(json.load(f).values())[0]
    if d.endswith('_baseline'):
        for m in metrics: bl[m].append(s[m])
    elif d.endswith('_transport_only'):
        for m in metrics: tr[m].append(s[m])
print(f\"{'Method':<20} {'Coherence':>10} {'Musicality':>10} {'Memorability':>12} {'Clarity':>10} {'Naturalness':>12}\")
print('-'*74)
for name, data in [('baseline',bl),('transport_only',tr)]:
    avg = {m: sum(v)/len(v) if v else 0 for m,v in data.items()}
    print(f'{name:<20} {avg[\"Coherence\"]:>10.4f} {avg[\"Musicality\"]:>10.4f} {avg[\"Memorability\"]:>12.4f} {avg[\"Clarity\"]:>10.4f} {avg[\"Naturalness\"]:>12.4f}')
"
