import torch


def eval_model(model: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: str):
    '''a function for evaluating any model'''

    model.train(mode=False)  # same as model.eval()
    model.to(device)

    batch_size = test_dataloader.batch_size
    # going through the batches in test_dataloader
    correct_pred = 0
    total_loss = 0
    for X_test, y_test in test_dataloader:
        # X_test = batch of BATCH_SIZE images
        # y_test = corresponding labels
        X_test, y_test = X_test.to(device), y_test.to(device)

        # accuracy
        y_pred = model(X_test)
        correct_pred += torch.eq(y_pred.argmax(dim=1), y_test).sum().item()

        # loss
        loss = loss_fn(y_pred, y_test)
        total_loss += loss

    accuracy = (correct_pred/((len(test_dataloader))*(batch_size)))*100
    total_loss = total_loss / (len(test_dataloader))

    # returning a dictionary
    return {"model_name": model.__class__.__name__,
            "model_acc": accuracy,
            "model_loss": total_loss}
