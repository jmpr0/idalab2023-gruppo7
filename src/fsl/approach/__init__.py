# Meta Learning
from .meta_module import LightningMetaModule
from .anil import LightningANIL
from .maml import LightningMAML
from .matching_net import LightningMatchingNetworks
from .metaoptnet import LightningMetaOptNet
from .proto_net import LightningPrototypicalNetworks
from .relation_net import LightningRelationNetworks

# Transfer Learning
from .tl_module import LightningTLModule
from .pre_training import PreTraining
from .finetuning import LightningFineTuning
from .freezing import LightningFreezing