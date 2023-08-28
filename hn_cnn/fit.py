import torch

from hn_cnn.cnn import HNLR
from hn_cnn.constants import *
from hn_cnn.utils import update_parameters, save_model

DEFAULT_HYPERPARAMETERS = {
    EPOCHS: 3000,
    LEARNING_RATE: 0.05,
    MOMENTUM: 0.9,
    DAMPENING: 0,
    RELU_SLOPE: 0.1,
    WEIGHTS_DECAY: 0.0001,
    OPTIMIZER: torch.optim.SGD,
    CLASS_WEIGHTS: [0.7, 3.7],
}

@torch.no_grad()
def evaluate(model, val_loader, weights):
    model.eval()
    metrics = {}
    for batch in val_loader:
        metrics = model.validation_step(batch, weights)
    return metrics

def fit(model, data_loaders, parameters={}, store_model={}):
    """ Train the model.
    """
    output = []
    best_val_auc = 0
    hyperparameters = update_parameters(parameters, DEFAULT_HYPERPARAMETERS)
    if isinstance(model, HNLR):
        # Train the logistic regression model
        for batch in data_loaders[TRAIN]:
            model.training(batch, class_weights=hyperparameters[CLASS_WEIGHTS])
        # Compute the metrics
        metrics = {}
        for subset, data_loader in data_loaders.items():
            if subset != TRAIN:
                for batch in data_loader:
                    subset_metrics = model.validation_step(batch)
                    metrics[subset] = subset_metrics
                    print(subset)
                    print(subset_metrics)
        output.append(metrics)
    else:
        # Train the neural networks
        optimizer = hyperparameters[OPTIMIZER](
            model.parameters(),
            lr=hyperparameters[LEARNING_RATE],
            momentum=hyperparameters[MOMENTUM],
            dampening=hyperparameters[DAMPENING],
            weight_decay=hyperparameters[WEIGHTS_DECAY],
        )
        # ADAM
        # optimizer = opt_func(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        # Training
        # start_time = time.time()
        for epoch in range(0, hyperparameters[EPOCHS]):
            print(f"Epoch {epoch}/{hyperparameters[EPOCHS]}")
            # Training
            model.train()
            for batch in data_loaders[TRAIN]:
                optimizer.zero_grad()
                loss = model.training_step(batch, hyperparameters[CLASS_WEIGHTS])
                # Gradient Normalization
                #grad_norm = 0
                #grad_params = torch.autograd.grad(outputs=loss,
                #    inputs=model.parameters(),
                #    create_graph=True
                # )
                #for grad in grad_params:
                #    grad_norm += grad.pow(2).sum()
                #grad_norm = grad_norm.sqrt()
                #loss = loss + grad_norm
                loss.backward()
                # Clipping the weights
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                # Clipping the weights
                #with torch.no_grad():
                #    for param in model.parameters():
                #        param.clamp_(-2, 2)
            # Time by epoch
            # print("--- %s seconds ---" % (time.time() - start_time))
            # Validation
            model.eval()
            with torch.no_grad():
                metrics = {}
                for subset, data_loader in data_loaders.items():
                    if subset != TRAIN:
                        subset_metrics = evaluate(model, data_loader, hyperparameters[CLASS_WEIGHTS])
                        metrics[subset] = subset_metrics
                        print(subset)
                        print(subset_metrics)
                output.append(metrics)
                if store_model[MODEL_PATH] and metrics[VALIDATION][ROC][AUC] > best_val_auc \
                    and metrics[VALIDATION][ROC][AUC] > store_model.get(THRESHOLD, 0) \
                        and abs(metrics[VALIDATION][ROC][AUC] - metrics[VALIDATION][ROC][AUC]) < \
                            store_model.get(MAX_DIFFERENCE, 1):
                    save_model(
                        store_model[MODEL_PATH],
                        epoch,
                        model,
                        optimizer,
                        loss,
                        model_id=store_model[MODEL_ID] or str(type(model))
                    )
                    best_val_auc = metrics[VALIDATION][ROC][AUC]
                if metrics[TRAIN_METRICS][ROC][AUC] > store_model.get(STOP_THRESHOLD, 0.95):
                    break 

    return output
