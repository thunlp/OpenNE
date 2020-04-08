import torch

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    # print(preds)
    # print(labels)
    # print(mask)
    # print("?")
    cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss = cross_entropy(input=preds, target=labels)
    # print(loss.shape)
    # print(mask.shape)
    # print(loss)
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    loss = loss.t() * mask
    return torch.mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1))
    #    tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = correct_prediction.type(torch.float32) # tf.cast(correct_prediction, tf.float32)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    accuracy_all *= mask
    return torch.mean(accuracy_all)
