# Implements a class that applies smoothign to a series of tensors. For example,
# a series of changing tensors could have an elementwise moving average applied.

import torch
import collections

class Z_Smoother:
    def __init__(self, buffer_len, init_z=None):
        self.buffer_len = buffer_len
        self._data = collections.deque()
        # if given initial data, populate the entire buffer
        if init_z != None:
            self.fill_buffer(init_z)

    def fill_buffer(self, init_z):
        """fill the Z_Smoother buffer with self.buffer_len copies of the same tensor, init_z.

        Args:
            init_z (tensor): A pytorch Tensor.
        """
        with torch.inference_mode():
            for _ in range(1,self.buffer_len):
                self._data.appendleft(init_z.clone())

    def append(self, new_tensor): 
        """Remove the oldest element from the buffer and then add the new tensor.

        Args:
            new_tensor (Tensor): A pytorch tensor.
        """
        self._data.pop()
        self._data.appendleft(new_tensor.clone())

    def smoothed_ma(self):
        """Return the elementwise average of all tensors in the buffer
        """
        # elementwise average of tensors
        with torch.inference_mode():
            mean_z = torch.mean(torch.stack(list(self._data)), dim=0)
            return mean_z


# output_tensor = eng.synth(mean_z)
# with torch.inference_mode():
#     TF.to_pil_image(output_tensor[0].cpu()).save(filepath_to_save, pnginfo=png_info_chunks(png_info))
