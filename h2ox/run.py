if __name__ == "__main__":
    # make experiment folders 
    expt_dir = create_model_experiment_folder(data_dir, "test_experiment")
    out_filename = create_model_save_str()

    # READ DATA 
    train_ds, test_ds = load_data_for_experiment(data_dir, diffed=DIFFED, rolling=ROLLING, deseas=DESEAS)
    # normalize the data
    norm_ds, (mean_, std_) = normalize_data(train_ds)
    # train the data
    train_ds = norm_ds.sel(time=slice(TRAIN_MIN, TRAIN_MAX))

    # MAKE DATASET for training
    train_dd = FcastDataset(
        ds=train_ds.sel(sample=[SAMPLE]), 
        forecast=fc.sel(sample=[SAMPLE]), 
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        target_var=TARGET_VAR,
        mode="train", 
        encode_doy=ENCODE_DOY, 
        encode_tod=False, 
        encode_dow=False,
        time_freq="D",
        history_variables=HISTORY_VARIABLES,
        forecast_variables=FORECAST_VARIABLES,
        total_time_bounds=TOTAL_TIME_BOUNDS,
        season=SEASON,
    )

    # --- MAKE DATALOADERS (train -- val) ---
    train_dataset, validation_dataset = train_val_split(train_dd, random_val_split=RANDOM_VAL_SPLIT)
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_dl = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- MAKE MODEL ---
    # get hyperparams from data shapes
    data_example = train_dl.__iter__().__next__()

    # get hyperparams from data shapes
    historical_input_size = data_example["x_d"].shape[-1]
    forecast_input_size = data_example["x_f"].shape[-1]
    horizon = data_example["x_f"].shape[1]
    seq_length = data_example["x_d"].shape[1]

    model = S2SModel(
        horizon=horizon,
        historical_input_size=historical_input_size,
        forecast_input_size=forecast_input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=0.4,
    )

    # --- TRAIN MODEL ---
    losses, val_losses = train(model, train_dl, epochs=N_EPOCHS, val_dl=val_dl, validate_every_n=VAL_EVERY)
    
    # --- EVALUATE MODEL ON TEST DATA ---
    norm_test_ds = (test_ds - mean_) / std_
    
    if EVAL_TEST:
        min_date = norm_test_ds.time.min().values
        max_date = norm_test_ds.time.max().values
        samples = norm_test_ds.sample.values
        test_fc = make_fcast_data_of_ones(min_date, max_date, samples=samples)
        test_fc = test_fc.sortby("initialisation_date")    
        
        # make dataset
        test_dd = FcastDataset(
            ds=norm_test_ds.sel(sample=[SAMPLE]), 
            forecast=test_fc.sel(sample=[SAMPLE]), 
            seq_len=SEQ_LEN,
            horizon=HORIZON,
            target_var=TARGET_VAR,
            mode="test", 
            encode_doy=ENCODE_DOY, 
            encode_tod=False, 
            encode_dow=False,
            time_freq="D",
            history_variables=HISTORY_VARIABLES,
            forecast_variables=FORECAST_VARIABLES,
            total_time_bounds=TOTAL_TIME_BOUNDS,
            season=SEASON,
        )

        test_dl = DataLoader(test_dd, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    else:
        test_dl = val_dl

    preds = test(model, test_dl)

    # unnormalize data
    preds = unnormalize_preds(preds, mean_, std_, target=TARGET_VAR, sample=SAMPLE)

    pass