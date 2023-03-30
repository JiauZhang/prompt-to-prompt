import torch

class CrossAttnCtrl:
    def __init__(self, device):
        self.__kv = {}
        self.device = device
        self.record = False
        self.timestep = None

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        self_attn = encoder_hidden_states is None
        if self_attn:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        if not self_attn or self.record:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if self_attn:
            attn_id = id(attn)
            if self.record:
                kv = [key.to(self.device), value.to(self.device)]
                if attn_id in self.__kv:
                    self.__kv[attn_id][self.timestep] = kv
                else:
                    self.__kv[attn_id] = {self.timestep: kv}
            else:
                key, value = self.__kv[attn_id][self.timestep]
                key, value = key.to(query.device), value.to(query.device)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
