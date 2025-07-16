for values in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), values))
    print("Training with", params)
    wandb_project = 'sar'
    wandb_entity = "erenberkedogan-new-jersey-institute-of-technology"
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=params,
        name="LSTM_" + str(params),
        notes=f"Grid search comparing lstm performance with constant learning rate {params["learning_rate"]} , dropout {params["dropout"]}, and {params["num_layers"]} layers",
    )
    state_dict = train(device, params, **params)       # pass num_pred=params["num_pred"], etc.
    scores = []

    for AR in [11698,11726,13165,13179,13183]:
        score, fig = eval(device, AR, False, state_dict, **params)
        wandb.log({f"Predictions/AR{AR}": wandb.Image(fig)})
        plt.close(fig) 
        scores.append(score)
    val_rmse = float(np.mean(scores))
    print(f"Score: {val_rmse:.8f}")

    if val_rmse < best_score:
        best_score, best_params, best_state = val_rmse, params, state_dict
    print(f"Best Parameters: {best_params}")
    wandb.finish()
