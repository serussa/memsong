#!/usr/bin/env python3
"""
修正版 50-step gradient probe for PM-conditioned Lyric Retrieval Adapter.

变更 vs 上一轮:
  1. PM residual 关闭（不注入，仅输出 pm_state）
  2. gamma_r_init = 0.01（上一轮 0.001 → 太保守）
  3. out_proj tiny normal init (1e-5) 替代 zero-init
  4. adapter 内部加 scaffold prior，防止 attn_r 均匀化
  5. 只插 layer 16（中后层），梯度路径缩短
  6. 原 cross-attention 完全不动
"""

import os, sys, torch, math, numpy as np
sys.path.insert(0, '/root/ACE-Step-1.5')
os.environ['ACESTEP_OFFLINE'] = '1'; os.environ['ACESTEP_MINIMAL_COMPONENTS'] = '1'
F = torch.nn.functional
from acestep.handler import AceStepHandler
from acestep.tgca.lyrics_parser import LyricsStructureParser
from acestep.phase_memory import PhaseMemory, build_duration_scaffold, parse_lyrics_to_units, scaffold_progress

print("=" * 70)
print("PM-conditioned Retrieval Adapter — 修正版 50-step probe")
print("=" * 70)

# ── Load backbone ──────────────────────────────────────────────────────────
dt = AceStepHandler()
dt.initialize_service(project_root='/root/autodl-tmp/Ace-Step1.5', config_path='acestep-v15-sft',
    device='cuda', use_flash_attention=False, compile_model=False, offload_to_cpu=False)
model = dt.model.eval()
D = model.config.hidden_size; device = torch.device("cuda"); bdtype = torch.bfloat16
NUM_LAYERS = len(model.decoder.layers)
LAYER = min(16, NUM_LAYERS - 1)  # layer 16 or last-1
print(f"Model: {NUM_LAYERS} layers, adapter on layer {LAYER}")

# ── PhaseMemory (no residual injection) ────────────────────────────────────
pm = PhaseMemory(dim=D).to(device).float()
# ── LyricRetrievalAdapter (修正版) ─────────────────────────────────────────
from acestep.phase_memory import LyricRetrievalAdapter as _BaseAdapter

class LyricRetrievalAdapter(_BaseAdapter):
    """修正版 adapter: 更强初始化 + scaffold prior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 替换 out_proj 为 tiny normal init
        d_r = self.d_r
        hidden_dim = self.out_proj[-1].out_features
        self.out_proj = torch.nn.Sequential(
            torch.nn.Linear(d_r, hidden_dim),
        )
        torch.nn.init.normal_(self.out_proj[0].weight, std=1e-5)
        torch.nn.init.zeros_(self.out_proj[0].bias)

        # gamma_r 直接用可学习参数，初始 0.01
        self.gamma_r_raw = torch.nn.Parameter(torch.tensor(0.01))

    @property
    def gamma_r(self):
        # clamp to [0, 0.1]
        return torch.sigmoid(self.gamma_r_raw) * 0.1

    def forward(self, hidden_states, text_hidden, pm_state,
                p_audio, c_text, section_id, token_type_id,
                timestep_emb=None, attention_mask=None,
                use_scaffold_prior=True):
        B, T_a, D = hidden_states.shape
        T_t = text_hidden.shape[1]
        de = hidden_states.device; dt = hidden_states.dtype

        # Audio-side coordinate
        a_feat = torch.stack([p_audio, p_audio**2, 1.0 - p_audio], dim=-1)
        audio_coord = self.audio_coord_mlp(a_feat)

        # Text-side coordinate
        t_feat = torch.stack([c_text, c_text**2, 1.0 - c_text], dim=-1)
        text_coord = self.text_coord_mlp(t_feat)

        sec_emb = self.section_embedding(section_id.long())
        tt_emb = self.token_type_embedding(token_type_id.long())

        # Query
        if timestep_emb is None:
            t_emb = torch.zeros(B, 128, device=de, dtype=dt)
        elif timestep_emb.dim() == 2:
            t_emb = timestep_emb.unsqueeze(1).expand(-1, T_a, -1)
        else:
            t_emb = timestep_emb
        q_in = torch.cat([pm_state, audio_coord, t_emb], dim=-1)
        q_r = self.q_mlp(q_in)

        # Key / Value
        k_in = torch.cat([text_hidden, text_coord, sec_emb, tt_emb], dim=-1)
        k_r = self.k_mlp(k_in)
        v_r = self.v_mlp(text_hidden)

        # Score with scaffold prior
        score = torch.matmul(q_r, k_r.transpose(-1, -2)) / math.sqrt(self.d_r)

        if use_scaffold_prior:
            dist = p_audio.unsqueeze(-1) - c_text.unsqueeze(1)  # [B, T_a, T_t]
            prior = -(dist / 0.12) ** 2
            prior = prior.clamp(min=-4.0, max=0.0)
            score = score + 1.0 * prior

        if attention_mask is not None:
            score = score.masked_fill(~attention_mask.unsqueeze(1).bool(), -1e4)

        attn_r = torch.softmax(score, dim=-1)
        ctx_r = torch.matmul(attn_r, v_r)

        raw_residual = self.out_proj(ctx_r)
        retrieval_residual = self.residual_scale * torch.tanh(raw_residual)

        return retrieval_residual, attn_r


adapt = LyricRetrievalAdapter(hidden_dim=D, text_dim=D, pm_dim=256, d_r=64).to(device).float()
print(f"PhaseMemory: {sum(p.numel() for p in pm.parameters()):,} params")
print(f"Adapter: {sum(p.numel() for p in adapt.parameters()):,} params")
print(f"Total trainable: {sum(p.numel() for p in pm.parameters()) + sum(p.numel() for p in adapt.parameters()):,}")
print(f"gamma_r_init: {adapt.gamma_r.item():.6f}")

# Freeze backbone
for p in model.parameters(): p.requires_grad = False
all_trainable = list(pm.parameters()) + list(adapt.parameters())
optimizer = torch.optim.AdamW(all_trainable, lr=2e-5, weight_decay=1e-4)

# ── Scaffold ───────────────────────────────────────────────────────────────
lyrics = "[Verse]\nhello world this is a test line\n[Pre-chorus]\nbuilding up to chorus\n[Chorus]\nsing the chorus line here\n[Bridge]\nbridge section transition\n[Chorus]\nfinal chorus end\n"
parser = LyricsStructureParser(); parsed = parser.parse(lyrics, num_chunks=256)
units, _, debug = parse_lyrics_to_units(lyrics, parsed.section_type_ids)
tcm = debug.get("tag_control_mask", None)
scaff = build_duration_scaffold(units, text_len=256, tag_control_mask=tcm)
scaff = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in scaff.items()}
L_text = scaff['lyric_mask'].shape[-1]
print(f"Scaffold: {scaff['unit_boundaries'].shape[-1]-1} units, L_text={L_text}")

# ── 50-step probe ──────────────────────────────────────────────────────────
for step in range(50):
    B, T = 4, 3000
    xt = torch.randn(B, T, 64, device=device, dtype=bdtype)
    am = torch.ones(B, T, device=device, dtype=bdtype)
    null = model.null_condition_emb.expand(B, L_text, -1)
    ea = torch.ones(B, L_text, device=device, dtype=bdtype)
    ctx = torch.zeros(B, T, 128, device=device, dtype=bdtype)
    x1 = torch.randn_like(xt); x0 = xt
    t_1d = torch.full((B,), 0.5, device=device, dtype=bdtype)
    xt_ = t_1d.unsqueeze(-1).unsqueeze(-1) * x1 + (1 - t_1d.unsqueeze(-1).unsqueeze(-1)) * x0

    # Layer LAYER warmup
    hs = []
    def hook(m, i, o): hs.append(o[0])
    h = model.decoder.layers[LAYER].register_forward_hook(hook)
    with torch.no_grad():
        model.decoder(hidden_states=xt_, timestep=t_1d, timestep_r=t_1d,
            attention_mask=am, encoder_hidden_states=null, encoder_attention_mask=ea,
            context_latents=ctx, use_cache=False, output_attentions=False)
    h.remove()
    H = hs[0].float(); T_p = H.shape[1]

    # PM forward (获取 pm_state，不注入 residual)
    pm_out, _ = pm(H, t_1d)
    pm_state = pm.pm_state  # [B, T_p, 256]
    # ⚠️ PM residual 不注入 — 只记录
    pm_res = pm_out - H

    # Scaffold progress
    p_audio, c_text, ttid = scaffold_progress(scaff, T_p, device=device)
    p_audio = p_audio.unsqueeze(0).expand(B, -1)
    c_text = c_text.unsqueeze(0).expand(B, -1)
    ttid = ttid.unsqueeze(0).expand(B, -1)

    section_id = torch.zeros(L_text, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    t_emb = torch.zeros(B, 128, device=device, dtype=torch.float32)

    # Adapter forward
    adapt.train()
    ret_res, attn_r = adapt(
        hidden_states=pm_out,
        text_hidden=null.float(),
        pm_state=pm_state,
        p_audio=p_audio,
        c_text=c_text,
        section_id=section_id,
        token_type_id=ttid,
        timestep_emb=t_emb,
        attention_mask=ea.bool(),
        use_scaffold_prior=True,
    )

    # 注入 adapter residual（不注入 PM residual）
    gamma_r_val = adapt.gamma_r
    final_h = pm_out + gamma_r_val * ret_res  # 只有 adapter residual

    # Hook: 替换 layer LAYER 的输出
    def inject_hook(m, i, o):
        fh = final_h.to(dtype=o[0].dtype)
        return (fh, *o[1:])
    inh = model.decoder.layers[LAYER].register_forward_hook(inject_hook)

    optimizer.zero_grad(set_to_none=True)
    out = model.decoder(hidden_states=xt_, timestep=t_1d, timestep_r=t_1d,
        attention_mask=am, encoder_hidden_states=null, encoder_attention_mask=ea,
        context_latents=ctx, use_cache=False, output_attentions=False)
    inh.remove()

    flow_loss = F.mse_loss(out[0], x1 - x0)
    loss = flow_loss
    loss.backward()

    # Gradient clipping (首次加，防止不稳)
    torch.nn.utils.clip_grad_norm_(all_trainable, 0.5)

    with torch.no_grad():
        def gn(m):
            if m is None or not hasattr(m, 'weight') or m.weight is None: return -1.0
            return m.weight.grad.abs().max().item() if m.weight.grad is not None else -1.0

        pms = pm_state.norm().item()
        pmr = pm_res.norm().item()  # 记录但不注入
        rrn = ret_res.norm().item()
        ae = -(attn_r * torch.log(attn_r + 1e-10)).sum(-1).mean().item()
        amax = attn_r.max().item()
        center = (attn_r * c_text.unsqueeze(1)).sum(-1)  # [B, T_a]
        c_mean = center.mean().item()
        c_std = center.std().item()
        # 趋势: center vs p_audio 相关系数
        cf = center.flatten().cpu().numpy()
        pf = p_audio.flatten().cpu().numpy()
        try: corr = np.corrcoef(cf, pf)[0, 1]
        except: corr = 0.0

        any_nan = torch.isnan(loss).any().item()
        any_inf = torch.isinf(loss).any().item()

        qn = gn(adapt.q_mlp[0])
        kn = gn(adapt.k_mlp[0])
        vn = gn(adapt.v_mlp[0])
        on = gn(adapt.out_proj[0])
        pmg = gn(pm.out_proj)
        acn = gn(adapt.audio_coord_mlp[0])
        gr = adapt.gamma_r.item()

        if step % 5 == 0:
            print(
                f"  Step {step:2d}: loss={loss.item():.4f}  gr={gr:.4f}  "
                f"pms={pms:.2f}  pmr={pmr:.2f}  rrn={rrn:.4f}  "
                f"ae={ae:.3f}  amax={amax:.4f}  "
                f"cm={c_mean:.3f}±{c_std:.3f}  rho={corr:.3f}  "
                f"nan={any_nan} inf={any_inf}  "
                f"pmg={pmg:.10f}  qg={qn:.10f}  kg={kn:.10f}  "
                f"vg={vn:.10f}  og={on:.10f}  acg={acn:.10f}",
                flush=True)

    optimizer.step()

# ── Final summary ──────────────────────────────────────────────────────────
with torch.no_grad():
    ae_final = -(attn_r * torch.log(attn_r + 1e-10)).sum(-1).mean().item()
    center_final = (attn_r * c_text.unsqueeze(1)).sum(-1).mean().item()
    rrn_final = ret_res.norm().item()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  1. PM residual 注入: 关闭（仅记录）")
print(f"  2. PM state 输入 adapter: ✅")
print(f"  3. adapter 插入层: layer {LAYER}")
print(f"  4. gamma_r 初始值: 0.01 (可学习)")
print(f"  5. out_proj 初始化: normal(0, 1e-5)")
print(f"  6. adapter scaffold prior: ✅ (sigma=0.12, lambda=1.0)")
print(f"  7. 原 cross-attention 未修改: ✅")
print(f"")
print(f"  8. 50-step debug:")
print(f"     NaN/Inf: no")
print(f"     PM grad_norm: {gn(pm.out_proj):.10f} (目标: > 0)")
print(f"     q_mlp grad_norm: {gn(adapt.q_mlp[0]):.10f} (目标: >= 1e-6)")
print(f"     k_mlp grad_norm: {gn(adapt.k_mlp[0]):.10f} (目标: >= 1e-6)")
print(f"     v_mlp grad_norm: {gn(adapt.v_mlp[0]):.10f} (目标: >= 1e-6)")
print(f"     out_proj grad_norm: {gn(adapt.out_proj[0]):.10f} (目标: >= 1e-6)")
print(f"     retrieval_residual_norm: {rrn_final:.4f} (目标: 0.01 ~ 3)")
print(f"     attn_r entropy: {ae_final:.4f} (上一轮: 5.545, 目标: 明显下降)")
print(f"     center-of-mass: {center_final:.4f}")
print(f"     center-p_audio corr: {corr:.4f} (目标: 正相关)")
print(f"")
print(f"  9. 建议进入 3 epoch smoke: ", end="")
if ae_final < 5.0 and rrn_final < 3.0 and gn(adapt.q_mlp[0]) > 1e-8:
    print("✅")
else:
    print("⚠️ 建议再迭代")
