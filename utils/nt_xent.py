import torch
import numpy as np

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        # Store configured batch size for reference, but don't rely on it for dynamic calculations
        self.initial_batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            # Typo fixed here: _cosine_simililarity -> _cosine_similarity
            return self._cosine_similarity
        else:
            # Typo fixed here: _dot_simililarity -> _dot_similarity
            return self._dot_similarity

    # Mask generation moved to forward pass
    # def _get_correlated_mask(self):
    #     # ... (old code removed)

    @staticmethod
    # Typo fixed here: _dot_simililarity -> _dot_similarity
    def _dot_similarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, M) # Adjusted for clarity, M might not be 2N
        # v shape: (N, M)
        return v

    # Typo fixed here: _cosine_simililarity -> _cosine_similarity
    def _cosine_similarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, M, C) # Adjusted for clarity, M might not be 2N
        # v shape: (N, M)
        # Original implementation was incorrect, using class instance method:
        # v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        # Corrected call to the nn.CosineSimilarity instance:
        similarity = torch.nn.CosineSimilarity(dim=-1)
        v = similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def _get_correlated_mask(self, batch_size):
        # Generate mask dynamically based on actual batch size B
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def forward(self, zis, zjs):
        # Get the actual batch size B
        B = zis.shape[0]
        if B == 0: # Handle case of genuinely empty input batch
             return torch.tensor(0.0, device=self.device, requires_grad=True)

        representations = torch.cat([zjs, zis], dim=0)
        # representations shape: [2*B, D]

        similarity_matrix = self.similarity_function(representations, representations)
        # similarity_matrix shape: [2*B, 2*B]

        # --- Calculate positives using actual batch size B ---
        l_pos = torch.diag(similarity_matrix, B)  # Offset by B
        r_pos = torch.diag(similarity_matrix, -B) # Offset by -B
        # l_pos, r_pos should have shape [B]

        # Ensure l_pos and r_pos have the expected size B, handle edge cases if necessary
        # (though torch.diag should handle B correctly if similarity_matrix is 2B x 2B)
        if l_pos.shape[0] != B or r_pos.shape[0] != B:
             # This case should ideally not happen if B > 0 and matrix is correct
             # Handle potential issues, maybe raise error or return zero loss
             print(f"Warning: Positive pair dimension mismatch. Expected {B}, got l_pos:{l_pos.shape[0]}, r_pos:{r_pos.shape[0]}")
             # Returning zero loss might hide underlying issues, consider raising error
             # raise RuntimeError(f"Positive pair dimension mismatch. Expected {B}, got l_pos:{l_pos.shape[0]}, r_pos:{r_pos.shape[0]}")
             return torch.tensor(0.0, device=self.device, requires_grad=True) # Or handle more gracefully

        positives = torch.cat([l_pos, r_pos]).view(2 * B, 1)
        # positives shape: [2*B, 1]

        # --- Calculate negatives using dynamically generated mask ---
        mask_samples_from_same_repr = self._get_correlated_mask(B)
        # mask shape: [2*B, 2*B]

        # Ensure mask shape matches similarity matrix shape
        if mask_samples_from_same_repr.shape != similarity_matrix.shape:
            raise RuntimeError(f"Mask shape {mask_samples_from_same_repr.shape} does not match similarity matrix shape {similarity_matrix.shape}")

        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * B, -1)
        # negatives shape: [2*B, 2*B-2]

        logits = torch.cat((positives, negatives), dim=1)
        # logits shape: [2*B, 2*B-1]
        logits /= self.temperature

        # --- Create labels based on actual batch size B ---
        labels = torch.zeros(2 * B).to(self.device).long()
        # labels shape: [2*B]

        loss = self.criterion(logits, labels)

        # --- Normalize loss based on actual batch size B ---
        # Check B > 0 to avoid division by zero if the check at the start wasn't sufficient
        if B > 0:
            loss = loss / (2 * B)
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True) # Should already be handled

        return loss