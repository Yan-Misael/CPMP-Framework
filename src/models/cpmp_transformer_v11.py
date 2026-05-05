import torch
import torch.nn as nn
from models.base.transformer import Transformer

class CPMPTransformer(Transformer):
    def __init__(self, H, C_dim, X_dim, d_model=64, nhead=8, num_layers=2, ff_dim_multiplier=4, dropout=0.1):
        super().__init__(
            H=H, C_dim=C_dim, X_dim=X_dim, d_model=d_model,
            nhead=nhead, num_layers=num_layers,
            ff_dim_multiplier=ff_dim_multiplier, dropout=dropout
        )
        self.d_model = d_model
        self.H = H
        
        # --- COMPONENTES PARA POLÍTICA ---
        self.policy_input_proj = nn.Linear(C_dim, d_model)
        self.policy_empty_embed = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.policy_intra_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers, enable_nested_tensor=False
        )
        self.policy_summary = nn.Linear(H * d_model, d_model)
        self.policy_x_proj = nn.Linear(X_dim, d_model)
        self.policy_fusion = nn.Linear(d_model * 2, d_model)
        self.policy_fusion_norm = nn.LayerNorm(d_model)
        
        self.policy_inter_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers
        )
        self.origin_proj = nn.Linear(d_model, d_model)
        self.dest_proj = nn.Linear(d_model, d_model)

        # --- COMPONENTES PARA VALOR (COSTO) ---
        self.value_input_proj = nn.Linear(C_dim, d_model)
        self.value_empty_embed = nn.Parameter(torch.randn(1, 1, 1, d_model))
        self.value_intra_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers, enable_nested_tensor=False
        )
        self.value_summary = nn.Linear(H * d_model, d_model)
        self.value_x_proj = nn.Linear(X_dim, d_model)
        self.value_fusion = nn.Linear(d_model * 2, d_model)
        self.value_fusion_norm = nn.LayerNorm(d_model)

        self.value_inter_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * ff_dim_multiplier, dropout, batch_first=True),
            num_layers=num_layers 
        )
        self.cost_attention = nn.Linear(d_model, 1)
        self.cost_head = nn.Sequential(
            nn.Linear(d_model, d_model * ff_dim_multiplier),
            nn.GELU(),
            nn.Linear(d_model * ff_dim_multiplier, d_model),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, S, X):
        batch_size, S_len, H, C_dim = S.shape
        device = S.device
        S_float = S.float()
        padding_mask = (S == -1).all(dim=-1).unsqueeze(-1)

        # --- FLUJO DE POLÍTICA ---
        xp = self.policy_input_proj(S_float)
        xp = torch.where(padding_mask, self.policy_empty_embed, xp)
        xp = xp.view(batch_size * S_len, H, self.d_model)
        xp = self.policy_intra_attn(xp)
        xp = xp.view(batch_size, S_len, H * self.d_model)
        
        sp = self.policy_summary(xp)
        xp_ext = self.policy_x_proj(X)
        p_emb = self.policy_fusion_norm(self.policy_fusion(torch.cat([sp, xp_ext], dim=-1)))
        
        x_policy = self.policy_inter_stack(p_emb)
        q = self.origin_proj(x_policy)
        k = self.dest_proj(x_policy)
        
        # Logits y Máscaras
        logits_matrix = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model**0.5)
        
        # (Máscaras simplificadas para mantener el foco en la arquitectura)
        mask_diag = torch.eye(S_len, device=device).bool().unsqueeze(0)
        is_origin_empty = (S == -1).all(dim=-1).all(dim=2)
        is_dest_full = ~(S == -1).all(dim=-1).any(dim=2)
        mask_origin = is_origin_empty.unsqueeze(2).expand(-1, -1, S_len)
        mask_dest = is_dest_full.unsqueeze(1).expand(-1, S_len, -1)
        logits_matrix = logits_matrix.masked_fill(mask_diag | mask_origin | mask_dest, -1e4)
        
        indices = torch.arange(S_len, device=device)
        mask_flat = indices.view(-1, 1) != indices.view(1, -1)
        logits = logits_matrix[:, mask_flat]

        # --- FLUJO DE VALOR (Totalmente separado + Detach inicial) ---
        # Detachamos S y X por seguridad, aunque al no haber capas compartidas 
        # el gradiente no cruzaría, pero es una buena práctica aquí.
        xv = self.value_input_proj(S_float)
        xv = torch.where(padding_mask, self.value_empty_embed, xv)
        xv = xv.view(batch_size * S_len, H, self.d_model)
        
        # Si quieres que el gradiente de valor NUNCA toque la entrada:
        xv = xv.detach() 
        
        xv = self.value_intra_attn(xv)
        xv = xv.view(batch_size, S_len, H * self.d_model)
        
        sv = self.value_summary(xv)
        xv_ext = self.value_x_proj(X)
        v_emb = self.value_fusion_norm(self.value_fusion(torch.cat([sv, xv_ext], dim=-1)))
        
        x_value = self.value_inter_stack(v_emb)
        attn_w = torch.softmax(self.cost_attention(x_value), dim=1)
        x_for_cost = torch.sum(x_value * attn_w, dim=1)
        
        cost = self.cost_head(x_for_cost).squeeze(-1)

        return logits, cost