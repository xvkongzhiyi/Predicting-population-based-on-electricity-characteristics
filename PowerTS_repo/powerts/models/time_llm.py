# powerts/models/time_llm.py
"""
Simple Time-LLM demo module.

Design:
- Uses a pretrained causal LLM (default: gpt2) from HuggingFace.
- Freezes LLM parameters by default; trains only a small regression head.
- Uses a lightweight numeric embedding (linear -> mean pool -> projector) as conditioning
  and injects it into the LLM's last hidden state before a prediction head.

Usage:
    model = TimeLLM(model_name="gpt2", input_dim=2, seq_len=96, pred_len=24, freeze_llm=True)
    yhat = model(x, user_idx)  # x: [B, seq_len, C], returns [B, pred_len]
"""
from typing import Optional
import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise ImportError("Please install transformers to use TimeLLM: pip install transformers") from e


class TimeLLM(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        input_dim: int = 2,
        seq_len: int = 96,
        pred_len: int = 24,
        numeric_embed_dim: int = 128,
        head_hidden: int = 256,
        freeze_llm: bool = True,
    ):
        """
        Args:
            model_name: HF model id (gpt2, distilgpt2, ...)
            input_dim: number of numeric channels (e.g., y, temp)
            seq_len: history length (for reference)
            pred_len: horizon to predict
            numeric_embed_dim: dim of numeric embedding summary
            head_hidden: hidden dim for regression head
            freeze_llm: whether to freeze pretrained LLM weights (recommended)
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim

        # numeric embedding: per-time-step linear -> mean pool -> projector
        self.num_proj = nn.Linear(input_dim, numeric_embed_dim)
        self.num_act = nn.GELU()
        self.num_pool_ln = nn.LayerNorm(numeric_embed_dim)
        self.num_to_cond = nn.Linear(numeric_embed_dim, numeric_embed_dim)

        # token model (LLM)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # ensure pad token exists (gpt2 doesn't define it)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

        # freeze LLM params by default
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        # projection from LLM hidden_size and numeric cond -> prediction
        hidden_size = self.llm.config.hidden_size
        self.cond_proj = nn.Linear(numeric_embed_dim, hidden_size)

        # small MLP head (trainable)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, pred_len)
        )

    def forward(self, x: torch.Tensor, user_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, input_dim] (float tensor, already on correct device)
            user_idx: optional, not used in this demo but kept for compatibility
        Returns:
            y_pred: [B, pred_len] float tensor (same device as input)
        """
        device = x.device
        B = x.shape[0]

        # numeric summary: embed per timestep and mean-pool
        # x: [B, L, C] -> [B, L, numeric_embed_dim]
        num_emb = self.num_act(self.num_proj(x))
        # mean pool over time -> [B, numeric_embed_dim]
        num_summary = num_emb.mean(dim=1)
        num_summary = self.num_pool_ln(num_summary)
        cond_vec = self.num_to_cond(num_summary)  # [B, hidden_size_cond] (numeric_embed_dim)

        # Build a simple textual prompt that includes last few numeric values (lightweight)
        # NOTE: This is a demo; more advanced tokenization/quantization can be used in real Time-LLM.
        # Convert last N numeric values to a short string prompt.
        last_vals = x[:, -min(self.seq_len, 24):, 0]  # take up to last 24 y-values for prompt
        # Build prompt strings in Python (on CPU)
        prompt_texts = []
        last_vals_cpu = last_vals.detach().cpu().numpy()
        for i in range(B):
            arr = last_vals_cpu[i]
            # keep decimal with limited precision
            s = ",".join([f"{float(v):.3f}" for v in arr])
            prompt_texts.append("History: " + s + " -> Predict next values:")

        # Tokenize prompt batch and move tokens to device
        inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # Forward through LLM (we only need hidden states)
        outputs = self.llm(**inputs)
        # outputs.hidden_states may be None depending on model/config; ensure it's present
        if getattr(outputs, "hidden_states", None) is None:
            # try calling with output_hidden_states=True
            outputs = self.llm(**inputs, output_hidden_states=True)

        # take last layer hidden states and last token -> [B, hidden_size]
        hidden_states = outputs.hidden_states[-1]  # [B, seq_toks, hidden_size]
        last_hidden = hidden_states[:, -1, :]  # [B, hidden_size]

        # project numeric cond to hidden_size and add
        cond_projected = self.cond_proj(cond_vec.to(device))  # [B, hidden_size]
        fused = last_hidden + cond_projected

        # prediction head
        y_pred = self.head(fused)  # [B, pred_len]
        # ensure same dtype as input
        y_pred = y_pred.to(x.dtype)

        return y_pred
