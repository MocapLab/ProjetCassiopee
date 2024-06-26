import torch
import torch.nn.functional as TF

def contrastive_loss(encoded_x, encoded_x_same, encoded_x_diff):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html
    # TODO: possibly wrong...
    dist_same = TF.cosine_similarity(encoded_x, encoded_x_same)
    dist_diff = - TF.cosine_similarity(encoded_x, encoded_x_diff)

    # ???
    # dist_same = TF.mse_loss(encoded_x, encoded_x_same)
    # dist_diff = 1 - TF.mse_loss(encoded_x, encoded_x_diff)
    # sum = dist_same - dist_diff

    sum = dist_same + dist_diff
    sum = torch.reshape(sum, [])

    print("Distance similar = ", dist_same)
    print("Distance different = ", dist_diff)

    return sum


def contrastive_classification_loss(encoded_x, encoded_x_same, encoded_x_diff, output_x, target, classification_loss):
    loss = classification_loss(output_x, target)
    loss += contrastive_loss(encoded_x, encoded_x_same, encoded_x_diff)
    return loss


def contrastive_reconstruction_loss(encoded_x, encoded_x_same, encoded_x_diff, decoded_x, x, reconstruction_loss):
    loss = reconstruction_loss(decoded_x, x)

    print("Reconstruction loss = ", loss)
    print("Contrastive loss = ", contrastive_loss(encoded_x, encoded_x_same, encoded_x_diff))

    loss += contrastive_loss(encoded_x, encoded_x_same, encoded_x_diff)

    return loss