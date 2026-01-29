# ===================== train.py =====================
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.nn import ModuleList
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.head import Detect

# =========================================================
# =============== 0. 实验模式控制中心 ======================
# =========================================================
EXPERIMENT_MODE = "baseline" 

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_CKPT = "/data1/saq/Fish/ultralytics/open_clip_model.safetensors"
TSV_PATH = "/data1/saq/Fish/ultralytics/datasets/ImageSets/tokens_train.tsv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_filtered_text(raw_text, mode):
    if mode == "baseline": return ""
    if mode == "random": return "apple banana orange"
    parts = raw_text.split(" ")
    if mode == "count":
        return [p for p in parts if "count=" in p][0] if any("count=" in p for p in parts) else ""
    if mode == "size":
        return [p for p in parts if "size=" in p][0] if any("size=" in p for p in parts) else ""
    if mode == "region":
        return [p for p in parts if "region=" in p][0] if any("region=" in p for p in parts) else ""
    return raw_text 

# =========================================================
# =============== 1. CLIP & LoRA 初始化 ===================
# =========================================================
class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        self.weight.requires_grad = False
        if self.bias is not None: self.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(original_layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    def forward(self, x):
        return self.original_layer(x) + (x @ self.lora_A.to(x.device) @ self.lora_B.to(x.device)) * self.scaling

def inject_lora(model):
    for name, module in model.named_modules():
        if "transformer" in name and isinstance(module, nn.Linear):
            parts = name.split('.')
            parent = model
            for part in parts[:-1]: parent = getattr(parent, part)
            setattr(parent, parts[-1], LoRALinear(module))
    return model

clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_CKPT)
clip_model = inject_lora(clip_model.to(DEVICE))
tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

TEXT_MAP = {}
if os.path.exists(TSV_PATH):
    with open(TSV_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(maxsplit=1) 
            if len(parts) == 2: TEXT_MAP[parts[0]] = get_filtered_text(parts[1], EXPERIMENT_MODE)

# =========================================================
# =============== 2. SESI 核心架构 =========================
# =========================================================
class SmartTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x, data_dict, *args, **kwargs):
        return super(SmartTensor, cls).__new__(cls, x, *args, **kwargs)
    def __init__(self, x, data_dict, *args, **kwargs):
        self.data_dict = data_dict
        for k, v in data_dict.items(): setattr(self, k, v)
    def __getitem__(self, key):
        if isinstance(key, str): return self.data_dict[key]
        return super().__getitem__(key)

class SESIShuffleFusion(nn.Module):
    def __init__(self, text_dim, feat_dim, groups=4):
        super().__init__()
        self.groups = groups
        self.spat_predictor = nn.Sequential(nn.Linear(text_dim, 128), nn.GELU(), nn.Linear(128, 3))
        self.text_to_feat = nn.Linear(text_dim, feat_dim)
        self.interaction = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim * 2, kernel_size=3, padding=1, groups=feat_dim * 2),
            nn.BatchNorm2d(feat_dim * 2), nn.SiLU(),
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1), nn.BatchNorm2d(feat_dim)
        )
        self.se_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(feat_dim, feat_dim // 8, 1), nn.ReLU(), nn.Conv2d(feat_dim // 8, feat_dim, 1), nn.Sigmoid())
        self.influence = nn.Parameter(torch.tensor([0.2]))

    def forward(self, feat, text_feat):
        B, C, H, W = feat.shape
        p = self.spat_predictor(text_feat)
        mu_x, mu_y = torch.sigmoid(p[:, 0]).view(B,1,1,1), torch.sigmoid(p[:, 1]).view(B,1,1,1)
        sigma = (torch.sigmoid(p[:, 2]) * 0.3 + 0.15).view(B,1,1,1)
        gy, gx = torch.meshgrid(torch.linspace(0, 1, H, device=feat.device), torch.linspace(0, 1, W, device=feat.device), indexing='ij')
        s_mask = torch.exp(-((gx - mu_x)**2 + (gy - mu_y)**2) / (2 * sigma**2))
        t_feat = self.text_to_feat(text_feat).view(B, C, 1, 1).expand(-1, -1, H, W)
        combined = torch.cat([feat, t_feat], dim=1).view(B, 2, self.groups, C // self.groups, H, W)
        combined = combined.transpose(1, 2).contiguous().view(B, 2 * C, H, W)
        res = self.interaction(combined)
        return feat + self.influence * (res * self.se_gate(res) * s_mask)

def detect_forward(self, x):
    text_feats = getattr(self, "_text_feats", None)
    if text_feats is not None and hasattr(self, "text_fusions"):
        x = [self.text_fusions[i](xi, text_feats) for i, xi in enumerate(x)]
    preds = self.forward_head(x, **getattr(self, 'one2many', {}))
    if self.training: return preds
    if getattr(self, 'end2end', False):
        x_detach = [xi.detach() for xi in x]
        one2one = self.forward_head(x_detach, **getattr(self, 'one2one', {}))
        res_dict = {"one2many": preds, "one2one": one2one}
        st = SmartTensor(one2one[0] if isinstance(one2one, (list, tuple)) else one2one, res_dict)
        return (self._inference(st), res_dict)
    return (self._inference(preds), preds)

Detect.forward = detect_forward

# =========================================================
# =============== 3. Trainer & 可视化函数 ==================
# =========================================================
class SESITrainer(DetectionTrainer):
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        img_files = batch.get("im_file")
        if not img_files: return batch
        img_ids = [Path(p).stem for p in img_files]
        texts = [TEXT_MAP.get(idx, "fish") for idx in img_ids]
        tokens = tokenizer(texts).to(DEVICE)
        clip_model.to(DEVICE)
        with torch.no_grad():
            text_feats = F.normalize(clip_model.encode_text(tokens), dim=-1).float()
        for m in self.model.modules():
            if isinstance(m, Detect): m._text_feats = text_feats
        return batch

    def build_optimizer(self, model, **kwargs):
        optimizer = super().build_optimizer(model, **kwargs)
        lora_params = [p for p in clip_model.parameters() if p.requires_grad]
        if lora_params: optimizer.add_param_group({'params': lora_params, 'lr': kwargs.get('lr', 0.01) * 0.1})
        return optimizer

def train():
    trainer = SESITrainer(cfg="/data1/saq/Fish/ultralytics/cfg/default.yaml", overrides=dict(
        device="0", model="/data1/saq/Fish/ultralytics/yolov8n.pt", 
        project="./runs", name=f"experiment_{EXPERIMENT_MODE}",
        epochs=100, batch=16, imgsz=640, lr0=0.005, workers=0, conf=0.001
    ))
    if isinstance(trainer.model, str): trainer.setup_model()
    for n, p in trainer.model.named_parameters():
        if "backbone" in n: p.requires_grad = False
    for m in trainer.model.modules():
        if isinstance(m, Detect):
            ch = getattr(m, 'ch', [c2[0].conv.in_channels for c2 in m.cv2])
            m.text_fusions = ModuleList([SESIShuffleFusion(512, c).to(DEVICE) for c in ch])
            m.add_module("text_fusions", m.text_fusions)
            for param in m.text_fusions.parameters(): param.requires_grad = True
    print(f"🚀 [EXPERIMENT] Mode: {EXPERIMENT_MODE.upper()} | Starting...")
    trainer.train()
    return trainer

SAVE_DIR = Path("./paper_figures")
SAVE_DIR.mkdir(exist_ok=True)

def get_gaussian_mask(mu_x, mu_y, sigma, h, w):
    y, x = np.ogrid[:h, :w]
    center_x, center_y = mu_x * w, mu_y * h
    s = sigma * max(h, w)
    mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * s**2))
    return (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

@torch.no_grad()
@torch.no_grad()
def generate_scientific_plot(trainer, img_path, text_query):
    model = trainer.model
    model.eval()
    
    # 1. 图像加载与预处理 (关键修正)
    img_bgr = cv2.imread(str(img_path))
    h0, w0 = img_bgr.shape[:2]
    # YOLO 默认输入尺寸是 640x640
    img_resized = cv2.resize(img_bgr, (640, 640))
    # BGR 转 RGB -> 转 Tensor -> 归一化 -> 增加 Batch 维度 (1, 3, 640, 640)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().to(DEVICE) / 255.0
    img_tensor = img_tensor.unsqueeze(0) 

    # 2. 准备文本特征
    tokens = tokenizer([text_query]).to(DEVICE)
    t_feat = F.normalize(clip_model.encode_text(tokens), dim=-1).float()
    
    # 3. 提取 SESI 参数
    detect_layer = next(m for m in model.modules() if isinstance(m, Detect))
    fusion_layer = detect_layer.text_fusions[0]
    p = fusion_layer.spat_predictor(t_feat)
    mu_x, mu_y = torch.sigmoid(p[0, 0]).item(), torch.sigmoid(p[0, 1]).item()
    sigma = torch.sigmoid(p[0, 2]).item() * 0.3 + 0.15

    # 4. 注入特征并推理
    for m in model.modules():
        if isinstance(m, Detect): m._text_feats = t_feat
    
    # 传入 Tensor 而非 Path
    preds = model(img_tensor)
    
    # 后处理：由于直接调用 model(tensor) 返回的是原始输出
    # 我们需要使用 Detect 模块的推理接口来转换结果
    # 简便起见，我们直接调用 YOLO 封装好的 post-process
    results = detect_layer.predict_by_decode(preds) if hasattr(detect_layer, 'predict_by_decode') else preds
    
    # 如果上面解析复杂，最稳妥的办法是使用内部封装的推理流，但手动处理 tensor
    # 这里我们采用一种最直接的“论文级”画法：
    # 直接使用非极大值抑制（NMS）获取框
    from ultralytics.utils.ops import non_max_suppression
    # preds 通常是一个 list 或者 tensor，我们需要过滤
    if isinstance(preds, (list, tuple)): preds = preds[0]
    output = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)[0]

    # 5. 绘图逻辑
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
    
    # 左图
    axes[0].imshow(img_rgb)
    if output is not None and len(output) > 0:
        for det in output:
            # 将 640x640 的坐标映射回原图尺寸
            x1, y1, x2, y2 = det[:4]
            x1, x2 = x1 * w0 / 640, x2 * w0 / 640
            y1, y2 = y1 * h0 / 640, y2 * h0 / 640
            axes[0].add_patch(plt.Rectangle((x1.item(), y1.item()), (x2-x1).item(), (y2-y1).item(), 
                                            fill=False, edgecolor='#00FF00', linewidth=2))
    axes[0].set_title("Standard Detection", fontsize=14)
    axes[0].axis('off')

    # 右图 (热力图)
    mask = get_gaussian_mask(mu_x, mu_y, sigma, h0, w0)
    axes[1].imshow(img_rgb)
    heatmap = axes[1].imshow(mask, cmap='jet', alpha=0.45)
    if output is not None and len(output) > 0:
        for det in output:
            x1, y1, x2, y2 = det[:4]
            x1, x2 = x1 * w0 / 640, x2 * w0 / 640
            y1, y2 = y1 * h0 / 640, y2 * h0 / 640
            axes[1].add_patch(plt.Rectangle((x1.item(), y1.item()), (x2-x1).item(), (y2-y1).item(), 
                                            fill=False, edgecolor='white', linewidth=1, linestyle='--'))
    axes[1].set_title(f"Ours: Semantic-Guided Mask\n{text_query}", fontsize=12)
    axes[1].axis('off')
    
    plt.savefig(SAVE_DIR / f"Paper_Vis_{Path(img_path).stem}.png", bbox_inches='tight')
    plt.close()

# =========================================================
# =============== 4. 程序入口 =============================
# =========================================================
if __name__ == "__main__":
    train()