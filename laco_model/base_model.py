import utils 
from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.net, self.net)
        self.net.to(self.device)

    def get_model(self):
        return self.model.networks