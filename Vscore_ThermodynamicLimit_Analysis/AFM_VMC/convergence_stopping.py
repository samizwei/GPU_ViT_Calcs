#modified version of ConvercenceStopping from NetKet

from collections import deque

import numpy as np

from netket.utils import struct


class LateConvergenceStopping(struct.Pytree, mutable=True):
    """A simple callback to stop the optimisation when the monitored quantity gets
    below a certain threshold for at least `patience` steps.
    """

    target: float = struct.field(serialize=False)
    """Target value for the monitored quantity. Training will stop if the driver drops below this value."""
    monitor: str = struct.field(serialize=False)
    """Loss statistic to monitor. Should be one of 'mean', 'variance', 'error_of_mean'."""
    smoothing_window: int = struct.field(serialize=False)
    """The loss is smoothed over the last `smoothing_window` iterations to
    reduce statistical fluctuations"""
    patience: int = struct.field(serialize=False)
    """The loss must be consistently below this value for this number of
    iterations in order to stop the optimisation."""
    start_from_step: int     #<-----------------------------------
    """Number of steps to wait before the callback has any effect."""

    # caches
    _loss_window: deque
    _patience_counter: int

    def __init__(
        self,
        target: float,
        monitor: str = "mean",
        *,
        smoothing_window: int = 10,
        patience: int = 10,
        start_from_step: int = 0,
    ):
        """
        Construct a callback stopping the optimisation when the monitored quantity
        gets below a certain threshold for at least `patience` steps.

        Args:
            target: the threshold value for the monitored quantity. Training will stop if the driver drops below this value.
            monitor: a string with the name of the quantity to be monitored. This
                is applied to the standard loss optimised by a driver, such as the
                Energy for the VMC driver. Should be one of
                'mean', 'variance', 'error_of_mean' (default: 'mean').
            smoothing_window: an integer number of steps over which the monitored value
                is averaged before comparing to target.
            patience: Number of steps to wait before stopping the execution after
                the tracked quantity drops below the target value (default 0, meaning
                that it stops immediately).
        """
        self.target = target
        self.monitor = monitor
        self.start_from_step = start_from_step  #<-----------------
        self.smoothing_window = smoothing_window
        self.patience = patience

        self._loss_window: deque = deque([], maxlen=self.smoothing_window)
        self._patience_counter: int = 0

    def __call__(self, step, log_data, driver):
        """
        A boolean function that determines whether or not to stop training.

        Args:
            step: An integer corresponding to the step (iteration or epoch) in training.
            log_data: A dictionary containing log data for training.
            driver: A NetKet variational driver.

        Returns:
            A boolean. If True, training continues, else, it does not.
        """
        if step < self.start_from_step:  #<--------------------------
            return True
        
        loss = np.asarray(np.real(getattr(log_data[driver._loss_name], self.monitor)))

        self._loss_window.append(loss)
        loss_smooth = np.mean(self._loss_window)

        if loss_smooth <= self.target:
            self._patience_counter += 1
        else:
            self._patience_counter = 0

        if self._patience_counter > self.patience:
            return False

        return True
