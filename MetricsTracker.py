import torch
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from pytorch_lightning import Callback
from IPython import display


def convert_tensors_to_arrays(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, list):
        return [convert_tensors_to_arrays(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_tensors_to_arrays(value) for key, value in data.items()}
    else:
        return data


class MetricTracker(Callback):
    def __init__(self):
        self.collection = []
        self.plotlosses = PlotLosses(outputs=[MatplotlibPlot()])

    def on_validation_epoch_end(self, trainer, module):
        if trainer.current_epoch == 0:
            return
        self.plotlosses.update(convert_tensors_to_arrays(trainer.logged_metrics))
        self.plotlosses.send()
