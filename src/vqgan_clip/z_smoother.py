# Implements a class that applies smoothign to a series of tensors. For example,
# a series of changing tensors could have an elementwise moving average applied.

import torch
import collections

class Z_Smoother:
    def __init__(self, buffer_len, alpha=0.7, init_z=None):
        self._buffer_len = buffer_len
        self._alpha = alpha
        self._data = collections.deque()
        # if given initial data, populate the entire buffer
        if init_z != None:
            self.fill_buffer(init_z)

        # pre-calculate weights for EWMA of length buffer_len
        self._ewma_wts = self._calc_ewma_wts(buffer_len, alpha)

    def fill_buffer(self, init_z):
        """fill the Z_Smoother buffer with self.buffer_len copies of the same tensor, init_z.

        Args:
            init_z (tensor): A pytorch tensor.
        """
        with torch.inference_mode():
            for _ in range(0,self._buffer_len):
                self._data.appendleft(init_z.clone())

    def append(self, new_tensor): 
        """Remove the oldest element from the buffer and then add the new tensor.

        Args:
            new_tensor (Tensor): A pytorch tensor.
        """
        if len(self._data) == self._buffer_len:
            self._data.pop()
        self._data.appendleft(new_tensor.clone())

    def smooth(self, method):
        """Returns the tensor data in the buffer with the selected smoothing method applied.

        Args:
            method (str): Smoothing method to use. E.g. 'mean'
        """
        if method.lower() == 'mean':
            return self._mean()

    def _mean(self):
        """Return the elementwise average of all tensors in the buffer
        """
        # elementwise average of tensors
        with torch.inference_mode():
            mean_z = torch.mean(torch.stack(list(self._data)), dim=0)
            return mean_z

    @staticmethod
    def _calc_ewma_wts(buffer_len, alpha):
        # Calculate "two-sided" EWMA weights from the midpoint of the buffer.
        # weights are biggest in the center, and decrease toward the end.
        # alpha sets how much the weights decrease with each element. Weights are (1-alpha)**index.
        if buffer_len % 2 == 0:
            raise NameError("Z_Smoother buffers must have odd length. The midpoint data is given the highest weight.")
        buffer_midpt = buffer_len // 2
        ewma_wts = [0.0] * buffer_len
        ewma_numerators = [None] * buffer_len
        for idx in range(0,buffer_midpt+1):
            ewma_numerators[buffer_midpt-idx] = (1-alpha)**idx
            ewma_numerators[buffer_midpt+idx] = (1-alpha)**idx
        ewma_denominator = sum(ewma_numerators)
        for idx in range(0,buffer_len):
            ewma_wts[idx] = ewma_numerators[idx]/ewma_denominator
        return ewma_wts

    def _mid_ewma(self):
        # Compute 2-sided EWMA to data in the buffer.
        # truncate the buffer to an odd number of elements
        if len(self._data) % 2 == 0:
            this_buffer = list(self._data)[0:-1]
        else:
            this_buffer = list(self._data)
        buffer_len = len(this_buffer)
        ewma_wts = self._calc_ewma_wts(buffer_len, self._alpha)
        with torch.inference_mode():
            scaled_data = [None] * buffer_len
            for idx in range(0,buffer_len):
                scaled_data[idx] = torch.mul(this_buffer[idx], ewma_wts[idx])
            mid_ewma_z = torch.sum(torch.stack(scaled_data), dim=0)
            return mid_ewma_z
