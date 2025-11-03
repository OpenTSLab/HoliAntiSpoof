import torch


def insert_spoof_embeddings(audio_features, lengths, spoof_embed):
    """
    audio_features: (T, E)
    lengths: list or 1D tensor of segment lengths [T1, T2, ..., TN]
    spoof_embed: (N, E)
    
    return: (T + N, E)
    """
    E = audio_features.shape[1]
    outputs = []
    start = 0
    for i, l in enumerate(lengths):
        end = start + l
        outputs.append(audio_features[start:end])
        outputs.append(spoof_embed[i].unsqueeze(0))
        start = end
    return torch.cat(outputs, dim=0)


def insert_spoof_embeddings_fast(audio_features, lengths, spoof_embed):
    T, E = audio_features.shape
    N = len(lengths)

    end_idx = torch.cumsum(lengths, dim=0)
    insert_positions = end_idx + torch.arange(N, device=audio_features.device)

    out = torch.zeros((T + N, E), device=audio_features.device, dtype=audio_features.dtype)

    offset = torch.arange(T, device=audio_features.device)
    offset_adjust = torch.repeat_interleave(torch.arange(N, device=audio_features.device), lengths)
    offset += offset_adjust

    out[offset] = audio_features
    out[insert_positions] = spoof_embed
    return out


audio_features = torch.arange(0, 12).reshape(6, 2).float()  # (6, 2)
lengths = torch.tensor([2, 3, 1])
spoof_embed = torch.tensor([[100, 101], [200, 201], [300, 301]]).float()

result1 = insert_spoof_embeddings(audio_features, lengths, spoof_embed)
result2 = insert_spoof_embeddings_fast(audio_features, lengths, spoof_embed)
print(result1)
print(result2)
