from collections import OrderedDict

from scipy.signal import savgol_filter


class Recorder:
    def __init__(self, seq=[], step=[], decay=0.99):
        self.data = seq
        self.step = step

        self.moving_avg_val = None
        self.decay = decay

    def record(self, value, step=None):
        self.data.append(value)

        if self.step is not None:
            self.step.append(step)

        if self.moving_avg_val is None:
            self.moving_avg_val = value

        else:
            self.moving_avg_val = (
                self.decay * self.moving_avg_val + (1 - self.decay) * value
            )

    def last(self):
        return self.data[-1]

    def moving_avg(self):
        return self.moving_avg_val

    def _get_window(self, window):
        window = min(len(self.data), window)

        if window % 2 == 0:
            window -= 1

        return window

    def savgol(self, window=99, order=1):
        window = self._get_window(window)

        if window < order + 1:
            smoothed = [0]

        else:
            smoothed = savgol_filter(self.data, window, order)

        return smoothed

    def moving_savgol(self, window=99, order=1):
        window = self._get_window(window)

        if window < order + 1:
            smoothed = [0]

        else:
            smoothed = savgol_filter(self.data[-window:], window, order)

        return smoothed[-1]


class RecorderSet:
    def __init__(self, *args, **kwargs):
        self.recorder = OrderedDict()

        for arg in args:
            self.recorder[arg] = Recorder(**kwargs)

    def record(self, step=None, **kwargs):
        for name, value in kwargs.items():
            self.recorder[name].record(value, step)

    def __getattr__(self, name):
        return self.recorder[name]
