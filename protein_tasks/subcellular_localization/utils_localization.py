import torch

def custom_none_collate_fn(batch):
    filtered_batch = [(emb, lbl) for emb, lbl in batch if emb is not None and emb.shape == torch.Size([1280])]
    if not filtered_batch:
        return torch.Tensor(), torch.Tensor()  # Return empty tensors if batch is empty
    embeddings, labels = zip(*filtered_batch)
    return torch.stack(embeddings), torch.stack(labels)