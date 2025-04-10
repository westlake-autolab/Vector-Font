import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import DDPMScheduler
from torch import Downblock, UPblock

class FontConditionEncoder(nn.Mod):
    """字体条件编码：支持名称或示例图像条件"""
    def __init__(self, font_num=1000, img_channels=3):
        super().__init__()
        # 名称条件分支
        self.font_embedding = nn.Embedding(font_num, 512)
        # 示例图像条件分支 
        self.img_encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 256, 3, stride=2, padding=1),
            nn.GroupNorm(16, 256), nn.SiLU()
        )
        
    def forward(self, font_ids=None, example_imgs=None):
        if font_ids is not None:
            return self.font_embedding(font_ids)
        else:
            return self.img_encoder(example_imgs)

# 去噪UNet
class RasterDenoiser(nn.Module):
    """
    input 4通道噪声图像
    """
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024)
        )
        
        # UNet主干（简化）
        self.down_blocks = nn.ModuleList([
            DownBlock(4, 64),
            DownBlock(64, 128),
            DownBlock(128, 256)
        ])
        self.up_blocks = nn.ModuleList([
            UpBlock(256, 128),
            UpBlock(128, 64),
            UpBlock(64, 64)
        ])
        self.final = nn.Conv2d(64, 4, 3, padding=1)
        
    def forward(self, x, t, char_emb, font_emb):
        '''
        source: 
        '''
        t_emb = self.time_embed(timestep_embedding(t, 512)) # 时间
        
        t_emb = t_emb + char_emb.unsqueeze(1) # 字符特征
        
        # 字体条件交叉注意力
        if font_emb.dim() == 4:  # 图
            b, c, h, w = font_emb.shape
            font_emb = font_emb.view(b, c, h*w).permute(0, 2, 1)
            for blk in self.down_blocks + self.up_blocks:
                if hasattr(blk, 'attn'):
                    x = blk.attn(x, context=font_emb) 
        else: # 名称
            t_emb = t_emb + font_emb.unsqueeze(1)
        
        skips = []
        for blk in self.down_blocks:
            x = blk(x, t_emb)
            skips.append(x)
        for blk in self.up_blocks:
            x = blk(x, skips.pop(), t_emb)
        return self.final(x)

class RasterDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.char_embed = nn.Embedding(10000, 896) # Unicode
        self.font_encoder = FontConditionEncoder()
        self.denoiser = RasterDenoiser()
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='cosine')
        
    def forward(self, char_ids, font_cond, clean_x0):
        # 生成扩散噪声
        noise = torch.randn_like(clean_x0)
        timesteps = torch.randint(0, 1000, (clean_x0.size(0),), device=clean_x0.device)
        noisy_x = self.scheduler.add_noise(clean_x0, noise, timesteps)
        
        # 特征编码
        char_emb = self.char_embed(char_ids)  # (B, 896)
        font_emb = self.font_encoder(font_cond)
        
        # 噪声预测
        pred_noise = self.denoiser(noisy_x, timesteps, char_emb, font_emb)
        return F.mse_loss(pred_noise, noise)


# init
model = RasterDM().cuda()
opt = AdamW(model.parameters(), lr=3.24e-5)

# char_ids Unicode index
# font_data 字体ID tensor or [B,3,H,W]
# x0 4通道 ground truth [B,4,64,64]

for batch in dataloader:
    x0, char_ids, font_cond = batch
    loss = model(char_ids, font_cond, x0) 
    opt.zero_grad()
    loss.backward()
    opt.step()

# 采样推理
def generate_raster(model, char_id, font_cond, num_steps=1000):
    model.eval()
    with torch.no_grad():
        # 初始噪声
        x = torch.randn(1, 4, 64, 64).cuda()
        
        # 条件编码
        char_emb = model.char_embed(char_id)
        font_emb = model.font_encoder(font_cond)
        
        # 时序去噪
        for t in reversed(range(num_steps)):
            timestep = torch.tensor([t], dtype=torch.long).cuda()
            noise_pred = model.denoiser(x, timestep, char_emb, font_emb)
            x = model.scheduler.step(noise_pred, t, x).prev_sample
    return x
