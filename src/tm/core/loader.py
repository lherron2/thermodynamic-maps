import torch
import logging
import numpy as np

from einops import rearrange
from torch.utils.data import Dataset
from collections import OrderedDict, Counter

from slurmflow.serializer import ObjectSerializer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Transform:
    def __init__(self, data):
        pass


class NormalTransform(Transform):
    def __init__(self, data):
        super().__init__(data)
        self.mean = data.mean(0)
        self.std = data.std(0)

    def forward(self, x):
        try:
            (_, nc, _, _) = x.shape
            return (x - self.mean[:nc, :, :]) / (self.std[:nc, :, :])
        except:
            return (x - self.mean[-1, :, :]) / (self.std[-1, :, :])

    def reverse(self, x):
        try:
            (_, nc, _, _) = x.shape
            return x * (self.std[:nc, :, :]) + self.mean[:nc, :, :]
        except:
            return x * (self.std[-1, :, :]) + self.mean[-1, :, :]


class MinMaxTransform(Transform):
    def __init__(self, data, dim, pos):
        super().__init__(data, dim, pos)

        self.min_data = data.min(0)[pos]
        self.max_data = data.max(0)[pos]

    def forward(self, x):
        return (x - self.min_data) / (2 * (self.max_data - self.min_data))

    def reverse(self, x):
        return 2 * (self.max_data - self.min_data) * x + self.min_data


class IdentityTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x

    def reverse(self, x):
        return x


TRANSFORMS = {
    "normal": NormalTransform,
    "min_max": MinMaxTransform,
    "identity": IdentityTransform,
}


class Dequantizer:
    def __init__(self, scale):
        self.scale = scale


class NormalDequantization(Dequantizer):
    def __init__(self, scale):
        super().__init__(scale)

    def forward(self, x):
        return x + torch.randn(*x.shape) * self.scale


class UniformDequantization(Dequantizer):
    def __init__(self, scale):
        super().__init__(scale)

    def forward(self, x):
        return x + torch.rand(*x.shape) * self.scale


DEQUANTIZERS = {"normal": NormalDequantization, "uniform": UniformDequantization}

class VectorDataLoader(Dataset):

    def __init__(self, 
                 multitraj_path,
                 rmsf_cutoff: float = 1e-3,
                 control_axis: int = 1,
                 dequantize: bool = True,
                 dequantize_type: str = "normal",
                 dequantize_scale: float = 1e-1,
                 DEQUANTIZERS: dict = DEQUANTIZERS,
                ):

        self.dataset_path = multitraj_path
        self.rmsf_cutoff = rmsf_cutoff
        self.lookup_table, self.traj_table = self.construct_tables(multitraj_path)
        self.control_axis = control_axis
        self.control_dims, self.data_dim = self.get_dims(self.traj_table)
        self.control_slice = self.build_control_slice(self.control_axis, self.control_dims, self.data_dim)
        self.dequantize = dequantize
        self.dequantizer = DEQUANTIZERS[dequantize_type](dequantize_scale)

        # Build cluster labels list
        self.cluster_labels = []
        for cluster in self.lookup_table:
            num_frames = self.lookup_table[cluster]
            self.cluster_labels.extend([cluster] * num_frames)
        
        # Compute cluster counts
        cluster_counts = Counter(self.cluster_labels)
        
        # Assign weights inversely proportional to cluster sizes
        self.weights = [1.0 / cluster_counts[cluster_label] for cluster_label in self.cluster_labels]
        
        # Set total frames
        self.total_frames = len(self.cluster_labels)
        
        # This attr must come last!
        self.data_shape = [1] + list(self.__getitem__(0)[-1].shape)

    @staticmethod
    def reshape(input):
        return rearrange(input, 'b h w c -> b c h w')

    def build_control_slice(self, control_axis, control_dims, data_dim):
        if control_axis is None and control_dims is None:
            control_slice = [slice(None) for _ in range(data_dim)]
        else:
            control_slice = [slice(None, None) for _ in range(data_dim)] 
            for axis in [control_axis]:
                control_slice[axis] = slice(control_dims[0], control_dims[1])    

        return tuple(control_slice)
        
    def get_dims(self, traj_table):
        cluster = list(traj_table.keys())[0]
        rmsf = traj_table[cluster]['rmsf']
        n_fluct_ch = rmsf.shape[self.control_axis]
        coords = traj_table[cluster]['coords']
        coords = coords.reshape([1] + list(coords.shape))
        n_coord_ch = coords.shape[self.control_axis]
        control_dims = (n_coord_ch, n_coord_ch + n_fluct_ch)
        data_dim = len(coords.shape)
        return control_dims, data_dim

    def construct_tables(self, multitraj_path):
        lookup_table = OrderedDict()
        traj_table = OrderedDict()
        
        OS = ObjectSerializer()
        top_level_paths = OS.get_summary(multitraj_path, depth=0)
        clusters = [x for x in top_level_paths if 'cluster' in x]
        logger.info(f"Found {len(clusters)} clusters in {multitraj_path}")

        for cluster in clusters:
            traj = OS.load(multitraj_path, cluster)
            lookup_table[cluster] = len(traj['coords'])
            traj_table[cluster] = traj
    
        return lookup_table, traj_table
            
    def get_cluster_and_frame_for_index(self, index):
        total_frames = sum(self.lookup_table.values())
        cumsum = np.cumsum(list(self.lookup_table.values()))
        indices_of_false = np.where((index < cumsum) == False)[0]
        indices_of_true = np.where((index < cumsum) == True)[0]
        
        last_false_index = indices_of_false[-1] if indices_of_false.size > 0 else None
        first_true_index = indices_of_true[0] if indices_of_true.size > 0 else -1
        
        offset = cumsum[last_false_index] if last_false_index is not None else 0
        cluster_frame = index - offset
        cluster = list(self.lookup_table.keys())[first_true_index]
    
        return cluster, cluster_frame

    def __getitem__(self, index):
        cluster, frame = self.get_cluster_and_frame_for_index(index)
    
        # Get coordinate data and reshape
        coords = self.traj_table[cluster]['coords'][frame]
        coords = coords.reshape([1] + list(coords.shape))
     
        temps = torch.Tensor(fluct)[0]
        sample_ = np.concatenate([coords, fluct], axis=self.control_axis)
        
        # Convert to tensor (remove extra batch dim) and apply dequantization
        sample = self.dequantizer.forward(torch.Tensor(sample_)[0])
    
        return temps, sample


    def __len__(self):
        return self.total_frames

