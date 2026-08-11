# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib as _matplotlib

    _matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from torch import nn

    from torchgeo.datasets import PASTIS100

    return PASTIS100, mo, nn, pd, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Marimo showcase: PASTIS100

    This notebook uses PASTIS100, TorchGeo's small PASTIS subset, to demonstrate a
    rendered dataframe, reactive sliders, and opt-in model training.
    """)
    return


@app.cell
def _(PASTIS100, pd):
    pastis100 = PASTIS100(root='tests/data/pastis',download=True)
    pastis100_dataframe = pd.DataFrame(
        {
            'PASTIS100 index': range(len(pastis100)),
            'observations': [
                pastis100[index]['image'].shape[0] for index in range(len(pastis100))
            ],
            'spatial size': [
                f'{pastis100[index]["image"].shape[-2]} × {pastis100[index]["image"].shape[-1]}'
                for index in range(len(pastis100))
            ],
        }
    )
    return pastis100, pastis100_dataframe


@app.cell
def _(mo, pastis100, pastis100_dataframe, pd):
    pastis100_index = mo.ui.slider(
        start=0,
        stop=len(pastis100) - 1,
        step=1,
        value=0,
        label='PASTIS100 sample index',
        show_value=True,
    )
    pastis100_timestamp = mo.ui.slider(
        start=0,
        stop=min(sample['image'].shape[0] for sample in pastis100) - 1,
        step=1,
        value=0,
        label='PASTIS100 timestamp',
        show_value=True,
    )
    pastis100_timestamps = pd.date_range(
        '2024-01-01',
        periods=min(sample['image'].shape[0] for sample in pastis100),
        freq='MS',
    )
    mo.vstack([pastis100_dataframe, mo.hstack([pastis100_index, pastis100_timestamp])])
    return pastis100_index, pastis100_timestamp, pastis100_timestamps


@app.cell
def _(
    pastis100,
    pastis100_index,
    pastis100_timestamp,
    pastis100_timestamps,
    plt,
):
    selected_image = (
        pastis100[pastis100_index.value]['image'][
            pastis100_timestamp.value, [2, 1, 0]
        ].permute(1, 2, 0)
        / 255
    )
    selected_timestamp = pastis100_timestamps[pastis100_timestamp.value]
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(selected_image)
    axis.set_title(
        f'PASTIS100 sample {pastis100_index.value}, {selected_timestamp:%Y-%m-%d}'
    )
    axis.set_axis_off()
    figure.tight_layout()
    figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train PASTIS100
    """)
    return


@app.cell
def _(mo):
    run_training = mo.ui.run_button(label='Train PASTIS100')
    run_training  # noqa: B018
    return (run_training,)


@app.cell
def _(mo, nn, pastis100, run_training, torch):
    mo.stop(
        not run_training.value,
        mo.md('Click **Train PASTIS100** to start the training run.'),
    )
    model = nn.Conv2d(3, 20, kernel_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _epoch in range(10):
        for index in range(len(pastis100)):
            sample = pastis100[index]
            logits = model(sample['image'][None, 0, :3] / 255)
            loss = nn.functional.cross_entropy(logits, sample['mask'][None])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    training_loss = losses[-1]
    return (training_loss,)


@app.cell
def _(mo, training_loss):
    mo.md(f"""
    Finished training. Final batch loss: `{training_loss:.3f}`
    """)
    return


if __name__ == "__main__":
    app.run()
