

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


class BLA(nn.Module):
    
    
    def __init__(
        self,
        brain_encoder: nn.Module,
        device: torch.device, 
        llm_name: str = "Qwen/Qwen3-0.6B-Base",
        brain_token_dim: int = 128,
        action_dim: int = 3,
        action_bins: int = 3,
    ):
        
        super().__init__()
        
        self.llm_name = llm_name
        self.brain_token_dim = brain_token_dim # 128
        
        self.brain_encoder = brain_encoder
        
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            torch_dtype=torch.bfloat16
        )
        
        self.llm_embedding_dim = self.llm.config.hidden_size # 1024
        
        self.brain_projection = nn.Sequential(
            nn.LayerNorm(self.brain_token_dim),
            nn.Linear(self.brain_token_dim, self.llm_embedding_dim),
            nn.GELU(),
            nn.Linear(self.llm_embedding_dim, self.llm_embedding_dim)
        )
        
        for p in self.brain_encoder.parameters():
            p.requires_grad = True
            
        for p in self.brain_encoder.classifier_module.parameters():
            p.requires_grad = False

        for p in self.brain_projection.parameters():
            p.requires_grad = True
        
        # [longitudinal movement, yaw rotation, vertical movement]
        # ACT_0 - ACT_2
        self.action_dim = action_dim
        self.action_bins = action_bins
        
        self.action_tokens = [
            f"<ACT_{bin_index}>"
            for bin_index in range(self.action_bins)
        ]
        
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": self.action_tokens
        })
        
        self.tokenizer.padding_side = "left"
        
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        self.action_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.action_tokens
        )
        
        self.llm_embedding_layer = self.llm.get_input_embeddings()
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def encode_and_merge(
        self,
        eeg: torch.Tensor, 
        instructions: list[str],
    ):
        
        # eeg is the eeg brain data, with shape (B, 875, 22)
        # instructions is the instruction strings for each window in the batch
        
        batch_size = eeg.shape[0]
        
        if len(instructions) != batch_size:
            raise ValueError(
                "number of instructions must equal batch size"
            )
        
        # ------------------------------------------
        # 1) encode and project the eeg data
        # ------------------------------------------
        
        # (B, 875, 22) -> (B, 5, 128)
        eeg = eeg.to(self.device)
        brain_embeddings = self.brain_encoder.encode(eeg)
        
        # (B, 5, 128) -> (B, 5, 1024)
        brain_tokens = self.brain_projection(brain_embeddings)
        
        brain_tokens = brain_tokens.to(
            dtype=self.llm_embedding_layer.weight.dtype
        )
        
        
        # ------------------------------------------
        # 2) tokenize the language instructions
        # ------------------------------------------
        
        formatted_instructions = [
            (
                "Control the drone according to the brain signal and "
                f"the following instructions:\n{instruction}\n"
                "Drone action:"
            )
            for instruction in instructions
        ]
        
        text_inputs = self.tokenizer(
            formatted_instructions,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt"
        )
        
        text_input_ids = text_inputs["input_ids"].to(self.device) # (B, text_length)
        text_attention_mask = text_inputs['attention_mask'].to(self.device) # (B, text_length)
        
        # (B, text_length) -> (B, text_length, 1024)
        text_tokens = self.llm_embedding_layer(text_input_ids)
        
        
        # ------------------------------------------
        # 3) build the complete multimodal sequence
        # ------------------------------------------
        
        # so now we have the following:
        # brain tokens: (B, 5, 1024)
        # text tokens: (B, text_length, 1024)
        
        # we then want to prepend the brain tokens to the text tokens
        # (B, 5 + text_length, 1024)
        full_tokens = torch.cat((brain_tokens, text_tokens), dim=1)
        
        
        # ------------------------------------------
        # 4) build the complete attention mask
        # ------------------------------------------
        
        # (B, 5)
        brain_attention_mask = torch.ones(
            batch_size,
            brain_tokens.shape[1],
            dtype=torch.long,
            device=self.device
        )
        
        # (B, 5 + text_length)
        full_attention_mask = torch.cat(
            (brain_attention_mask, text_attention_mask),
            dim=1
        )
        
        return full_tokens, full_attention_mask
    
    
    def forward_causal(
        self, 
        eeg: torch.Tensor, 
        instructions: list[str],
        target_action_token_ids: torch.Tensor | None = None,
    ):
        
        # ------------------------------------------
        # 1) merge multi-modal data, and produce attention mask
        # ------------------------------------------
        # (B, 5 + text_length, 1024)
        # (B, 5 + text_length)
        full_tokens, full_attention_mask = self.encode_and_merge(eeg, instructions)
        
        # ------------------------------------------
        # 2) causal mask for training with one forward pass
        # ------------------------------------------
        
        batch_size = eeg.shape[0]
        
        if target_action_token_ids is None:
            raise ValueError("forward_causal requires target_action_token_ids")
        
        if target_action_token_ids.shape != (batch_size, self.action_dim):
            raise ValueError(
                f"expected target_action_token_ids shape "
                f"({batch_size}, {self.action_dim}), "
                f"but received {tuple(target_action_token_ids.shape)}"
            )
        
        target_action_token_ids = target_action_token_ids.to(self.device)
        
        # (B, 3, 1024)
        target_action_token_embeds = self.llm_embedding_layer(
            target_action_token_ids
        )
        
        # (B, 5 + text_length + 3, 1024)
        full_causal_tokens = torch.cat(
            [full_tokens, target_action_token_embeds],
            dim = 1
        )
        
        # (B, 3)
        causal_tokens_attention_mask = torch.ones(
            batch_size,
            self.action_dim,
            dtype=torch.long,
            device=self.device
        )
        
        # (B, 5 + text_length + 3)
        full_causal_attention_mask = torch.cat(
            [full_attention_mask, causal_tokens_attention_mask],
            dim = 1
        )
        
        # 5 + text_length
        prefix_len = full_tokens.shape[1]
        
        # ignore loss on brain + instruction prefix tokens
        # (B, 5 + text_length)
        prefix_labels = torch.full(
            (batch_size, prefix_len),
            -100,
            dtype = torch.long,
            device = self.device
        )
        
        # (B, 5 + text_length + 3)
        full_labels = torch.cat(
            [prefix_labels, target_action_token_ids],
            dim = 1
        )
        
        # outputs.logits: (B, 5 + text_length + 3, vocab_size)
        # token N in outputs.logits predicts token N + 1 in full_causal_tokens
        outputs = self.llm(
            inputs_embeds=full_causal_tokens,
            attention_mask=full_causal_attention_mask,
            labels=full_labels,
            use_cache=False,
            return_dict=True,
        )
        
        # the token id's (index position in vocab logits) for each possible action token
        # (action_bins,)
        action_token_id_tensor = torch.tensor(
            self.action_token_ids,
            dtype=torch.long,
            device=self.device
        )
        
        # (B, 3, vocab_size)
        action_step_logits = outputs.logits[:, prefix_len - 1 : prefix_len - 1 + self.action_dim, :]
        
        # (B, 3, action_bins)
        # these are basically the logits for only the action tokens, for each step of autoregressive generation
        action_logits = action_step_logits.index_select(
            index=action_token_id_tensor,
            dim=2
        )
        
        # (B, 3)
        # for each generated token, which logit id was the highest
        predicted_action_class = action_logits.argmax(dim=2)
        predicted_action_token_ids = action_token_id_tensor[predicted_action_class] # the true token id's of the predicted action tokens
        
        return {
            "loss": outputs.loss,
            "predicted_action_token_ids": predicted_action_token_ids
        }
    
    def forward_generate(
        self, 
        eeg: torch.Tensor, 
        instructions: list[str],
    ):
        
        # (B, 5 + text_length, 1024)
        # (B, 5 + text_length)
        prefix_embeds, prefix_mask = self.encode_and_merge(eeg, instructions)
        
        # outputs.logits: (B, 5 + text_length, vocab_size)
        outputs = self.llm(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            use_cache=True,
            return_dict=True
        )
        
        batch_size = eeg.shape[0]
        
        past = outputs.past_key_values
        pred_ids = []
        attn = prefix_mask
        
        # (action_bins,)
        action_token_id_tensor = torch.tensor(
            self.action_token_ids,
            dtype=torch.long,
            device=self.device
        )
        
        for step in range(self.action_dim):
            # (B, vocab_size)
            step_logits = outputs.logits[:, -1, :]
            
            # (B, action_bins)
            action_logits = step_logits.index_select(index=action_token_id_tensor, dim = 1)
            
            # (B,)
            pred_class = action_logits.argmax(dim = 1)
            
            # (B,)
            next_ids = action_token_id_tensor[pred_class]
            
            pred_ids.append(next_ids)
            
            if step < self.action_dim - 1:
                # (B, 1, 1024)
                next_embeds = self.llm_embedding_layer(next_ids).unsqueeze(1)
                
                # (B, 1)
                next_attn = torch.ones(
                    batch_size,
                    1,
                    dtype=torch.long,
                    device=self.device,
                )
                
                # (B, 5 + text_length + step + 1)
                attn = torch.cat([attn, next_attn], dim = 1)
                
                outputs = self.llm(
                    inputs_embeds=next_embeds,
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True
                )
                past = outputs.past_key_values
        
        # (B, 3)
        return {
            "predicted_action_token_ids": torch.stack(pred_ids, dim = 1)
        }
    
    
    def forward(
        self,
        eeg: torch.Tensor,
        instructions: list[str],
        target_action_token_ids: torch.Tensor | None = None,
    ):
        if target_action_token_ids is not None:
            return self.forward_causal(
                eeg=eeg,
                instructions=instructions,
                target_action_token_ids=target_action_token_ids,
            )
        
        return self.forward_generate(
            eeg=eeg,
            instructions=instructions,
        )
        
        
        