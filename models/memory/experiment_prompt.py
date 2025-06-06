from dataclasses import dataclass
import torch
import torch.nn as nn
import pdb
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from typing import List, override, Any
from models.prompt import Prompt


@dataclass(frozen=True, slots=True)
class WeightInfo:
    level: int
    W: Any
    task_id: int
    class_labels: list[int]


class ExperimentPrompt(Prompt):
    @override
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, args=None):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim, args)

        # Task_id -> List[weight info]
        self.weight_info_dict: dict[int, List[WeightInfo]] = {}


    @override
    def forward(self, x_querry, l, x_block, train=False, task_id=None, class_labels=None):
        # e prompts
        if len(x_querry.shape) !=2:
            #print('yes')
            query_wt = self.query_tf(x_querry.view(x_querry.shape[0],-1)) # why learn it here instead of directly learn Q -> K by A
            x_querry = x_querry * query_wt.unsqueeze(-1)
            x_querry = x_querry.sum(dim=1)

        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            K = self.e_k[str(l)]
            A = self.e_a[str(l)]
            p = self.e_p[str(l)]

            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)

            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)

            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)

            aq_k_save = aq_k.detach().tolist()

            if class_labels is not None:
                class_labels_list = []
                for class_ls in class_labels:
                    if isinstance(class_ls, torch.Tensor):
                        class_labels_list.append(class_ls.detach().tolist())
                    else:
                        class_labels_list.append(class_ls)
            else:
                class_labels_list = None

            #store aq_k in weight_info_dict
            if task_id is not None:
                if task_id not in self.weight_info_dict:
                    self.weight_info_dict[task_id] = []

                self.weight_info_dict[task_id].append(WeightInfo(level=l, W=aq_k_save, task_id=task_id, class_labels=class_labels_list))
            else:
                # print warning without task_id
                print(f"Warning: task_id is None in ExperimentPrompt for level {l}")
                # store in -1
                if -1 not in self.weight_info_dict:
                    self.weight_info_dict[-1] = []
                self.weight_info_dict[-1].append(WeightInfo(level=l, W=aq_k_save, task_id=-1, class_labels=class_labels))

            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block
    
    def save_weights_info_dict(self, path):
        """
        Save the weight info dictionary to a file.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.weight_info_dict, f)

def visualize_weights_info(weights_info_dict, save_path=None):
    tasks = sorted(weights_info_dict.keys())
    cmap = plt.get_cmap('tab10', len(tasks))  # get a colormap with one color per task

    plt.figure(figsize=(10, 6))

    for idx, task in enumerate(tasks):
        color = cmap(idx)
        label_added = False  # track label addition per task to avoid duplicate legend entries

        for wt_info in weights_info_dict[task]:
            # ensure W is a numpy array
            W = wt_info.W if isinstance(wt_info.W, np.ndarray) else np.array(wt_info.W)
            batch_size = W.shape[0]
            for i in range(batch_size):
                x = np.arange(W.shape[1])
                y = W[i]
                # add label only once per task
                if not label_added:
                    plt.plot(x, y, color=color, label=f"Task {task}")
                    label_added = True
                else:
                    plt.plot(x, y, color=color)

    plt.xlabel("Index")
    plt.ylabel("Weight Value")
    plt.title("Weights Visualization by Task")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def visualize_weights_by_class(weights_info_dict, save_path=None, limit_classes=None, alpha=0.5):
    """
    Visualize weights info per class label.

    For each WeightInfo, if all class labels (after flattening) are identical,
    the weights are plotted against the index using a consistent color for that class.

    Args:
        weights_info_dict: Dictionary mapping task ids to lists of WeightInfo.
        save_path: Optional file path to save the plot.
        limit_classes: Optional set or list of class labels to visualize.
        alpha: Transparency value for the plots (default: 0.5).
    """
    def extract_labels(class_labels):
        # If elements are tensors with more than one element, flatten them.
        new_labels = []
        for item in class_labels:
            if isinstance(item, torch.Tensor):
                if item.numel() > 1:
                    new_labels.extend(item.cpu().tolist())
                else:
                    new_labels.append(item.item())
            else:
                new_labels.append(item)
        return new_labels

    # Group WeightInfo entries by unique class label (only if they have a single unique label)
    class_dict = {}
    for task in weights_info_dict:
        for wt_info in weights_info_dict[task]:
            if isinstance(wt_info.class_labels, torch.Tensor):
                labels = wt_info.class_labels.cpu().tolist()
            elif isinstance(wt_info.class_labels, list):
                labels = extract_labels(wt_info.class_labels)
            else:
                labels = wt_info.class_labels

            if not labels:
                print(f"Warning: WeightInfo for task {task} has no class labels.")
                continue

            unique_labels = set(labels)
            if len(unique_labels) == 1:
                label = unique_labels.pop()
                if (limit_classes is None) or (label in limit_classes):
                    if label not in class_dict:
                        class_dict[label] = []
                    class_dict[label].append(wt_info)
    
    # Get sorted list of class labels to visualize
    sorted_labels = sorted(class_dict.keys())
    cmap = plt.get_cmap('tab10', len(sorted_labels))
    
    plt.figure(figsize=(10, 6))
    
    for idx, label in enumerate(sorted_labels):
        color = cmap(idx)
        label_added = False  # ensure only one legend entry per class
        for wt_info in class_dict[label]:
            W = wt_info.W if isinstance(wt_info.W, np.ndarray) else np.array(wt_info.W)
            batch_size = W.shape[0]
            for i in range(batch_size):
                x = np.arange(W.shape[1])
                y = W[i]
                if not label_added:
                    plt.plot(x, y, color=color, alpha=alpha, label=f"Class {label}")
                    label_added = True
                else:
                    plt.plot(x, y, color=color, alpha=alpha)
    
    plt.xlabel("Index")
    plt.ylabel("Weight Value")
    plt.title("Weights Visualization by Class Label")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def visualize_weights_info_aggregated_advanced(weights_info_dict, save_path=None, 
                                               line_visual=["mean"], area_visual=["std"],
                                               alpha=0.3):
    """
    Aggregated visualization per task.
    
    For each task in weights_info_dict, all weight curves (concatenated across
    WeightInfo entries and batch dimension) are aggregated. The user can pass in:
    
      line_visual: A list telling which statistic lines to plot.
          Options: "mean", "median"
      area_visual: A list telling which area to fill.
          Options: "std" (used with "mean") and "min_max" (used with "median")
    
    The area is filled with the same base color as its corresponding line (with alpha).
    
    Args:
        weights_info_dict: Dict mapping task ids to lists of WeightInfo.
        save_path: Optional file path to save the plot.
        line_visual: List of strings ["mean", "median"] to plot as lines.
        area_visual: List of strings ["std", "min_max"] to plot as shaded area.
        alpha: Transparency for the shading.
    """

    tasks = sorted(weights_info_dict.keys())
    cmap = plt.get_cmap('tab10', len(tasks))
    
    plt.figure(figsize=(10, 6))
    x = None
    
    for idx, task in enumerate(tasks):
        # Gather curves for the task from all WeightInfo entries.
        curves_list = []
        for wt_info in weights_info_dict[task]:
            W = wt_info.W if isinstance(wt_info.W, np.ndarray) else np.array(wt_info.W)
            curves_list.append(W)
        if not curves_list:
            continue
        # Concatenate along first axis; shape becomes (num_curves, curve_length)
        curves = np.concatenate(curves_list, axis=0)
        x = np.arange(curves.shape[1])
        base_color = cmap(idx)
        
        # Plot mean line & area (if requested)
        if "mean" in line_visual:
            mean_curve = np.mean(curves, axis=0)
            plt.plot(x, mean_curve, color=base_color, lw=2, label=f"Task {task} Mean")
            if "std" in area_visual:
                std_curve = np.std(curves, axis=0)
                lower = mean_curve - std_curve
                upper = mean_curve + std_curve
                plt.fill_between(x, lower, upper, color=base_color, alpha=alpha)
        
        # Plot median line & area (if requested)
        if "median" in line_visual:
            median_curve = np.median(curves, axis=0)
            plt.plot(x, median_curve, color=base_color, lw=2, linestyle="--", label=f"Task {task} Median")
            if "min_max" in area_visual:
                lower = np.min(curves, axis=0)
                upper = np.max(curves, axis=0)
                plt.fill_between(x, lower, upper, color=base_color, alpha=alpha)
    
    plt.xlabel("Index")
    plt.ylabel("Weight Value")
    plt.title("Aggregated Weights Visualization by Task")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def visualize_weights_by_class_aggregated_advanced(weights_info_dict, save_path=None, 
                                                   limit_classes=None, 
                                                   line_visual=["mean"], area_visual=["std"],
                                                   alpha=0.3):
    """
    Aggregated visualization per class label.
    
    For each WeightInfo (grouped by class label, where the class labels are provided as a
    list of lists (batch_size x class_labels)), this function first flattens the nested list.
    If the flattened list consists of only one unique value, then that WeightInfo is grouped.
    
    The user can choose:
      line_visual: A list specifying which statistic lines to plot. Options: "mean", "median"
      area_visual: A list specifying which area to fill. Options: "std" (with "mean") and "min_max" (with "median")
    
    The plotted area uses the same base color (with transparency set by alpha).
    
    Args:
        weights_info_dict: Dict mapping task ids to lists of WeightInfo.
        save_path: Optional file path to save the plot.
        limit_classes: Optional list or set of class labels to visualize.
        line_visual: List of statistics for lines (e.g. ["mean", "median"]).
        area_visual: List of statistics for the area (e.g. ["std", "min_max"]).
        alpha: Transparency for the shading.
    """

    def extract_labels(class_labels):
        new_labels = []
        # class_labels can be a list of lists, so flatten all elements.
        for item in class_labels:
            if isinstance(item, list):
                new_labels.extend(item)
            elif isinstance(item, torch.Tensor):
                # If tensor is not scalar, flatten it.
                if item.numel() > 1:
                    new_labels.extend(item.cpu().tolist())
                else:
                    new_labels.append(item.item())
            else:
                new_labels.append(item)
        return new_labels

    # Group WeightInfo entries by unique class label (only if flattened list has a single unique value)
    class_dict = {}
    for task in weights_info_dict:
        for wt_info in weights_info_dict[task]:
            # Process class_labels that could be a list of lists.
            if isinstance(wt_info.class_labels, torch.Tensor):
                labels = wt_info.class_labels.cpu().tolist()
            elif isinstance(wt_info.class_labels, list):
                labels = extract_labels(wt_info.class_labels)
            else:
                labels = wt_info.class_labels

            if not labels:
                print(f"Warning: WeightInfo for task {task} has no class labels.")
                continue

            unique_labels = set(labels)
            if len(unique_labels) == 1:
                label = unique_labels.pop()
                if (limit_classes is None) or (label in limit_classes):
                    if label not in class_dict:
                        class_dict[label] = []
                    class_dict[label].append(wt_info)

    sorted_labels = sorted(class_dict.keys())
    cmap = plt.get_cmap('tab10', len(sorted_labels))
    plt.figure(figsize=(10, 6))
    x = None

    for idx, label in enumerate(sorted_labels):
        curves_list = []
        for wt_info in class_dict[label]:
            W = wt_info.W if isinstance(wt_info.W, np.ndarray) else np.array(wt_info.W)
            curves_list.append(W)
        if not curves_list:
            continue
        curves = np.concatenate(curves_list, axis=0)
        x = np.arange(curves.shape[1])
        base_color = cmap(idx)

        if "mean" in line_visual:
            mean_curve = np.mean(curves, axis=0)
            plt.plot(x, mean_curve, color=base_color, lw=2, label=f"Class {label} Mean")
            if "std" in area_visual:
                std_curve = np.std(curves, axis=0)
                lower = mean_curve - std_curve
                upper = mean_curve + std_curve
                plt.fill_between(x, lower, upper, color=base_color, alpha=alpha)
        if "median" in line_visual:
            median_curve = np.median(curves, axis=0)
            plt.plot(x, median_curve, color=base_color, lw=2, linestyle="--", label=f"Class {label} Median")
            if "min_max" in area_visual:
                lower = np.min(curves, axis=0)
                upper = np.max(curves, axis=0)
                plt.fill_between(x, lower, upper, color=base_color, alpha=alpha)

    plt.xlabel("Index")
    plt.ylabel("Weight Value")
    plt.title("Aggregated Weights Visualization by Class Label")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def get_all_class_labels(weights_info_dict):
    """
    Extracts and returns all unique class labels from a weights_info_dict,
    ignoring any entries that are None.
    
    Args:
        weights_info_dict: Dictionary mapping task ids to lists of WeightInfo.
    
    Returns:
        A sorted list of unique class labels present across all WeightInfo entries.
    """
    import torch

    def flatten_labels(class_labels):
        """Helper function to flatten nested class_labels."""
        flat = []
        for item in class_labels:
            if item is None:
                continue
            # If item is a list (e.g., when class_labels is a list of lists)
            if isinstance(item, list):
                flat.extend(flatten_labels(item))
            # If it's a torch tensor, convert to list (flatten if needed)
            elif isinstance(item, torch.Tensor):
                if item.numel() > 1:
                    flat.extend(item.cpu().tolist())
                else:
                    flat.append(item.item())
            else:
                flat.append(item)
        return flat

    all_labels = set()
    for task in weights_info_dict:
        for wt_info in weights_info_dict[task]:
            # Process the class_labels from each WeightInfo
            if wt_info.class_labels is None:
                continue
            if isinstance(wt_info.class_labels, torch.Tensor):
                labels = wt_info.class_labels.cpu().tolist()
            elif isinstance(wt_info.class_labels, list):
                labels = flatten_labels(wt_info.class_labels)
            else:
                labels = [wt_info.class_labels]

            # Filter out None values from labels
            labels = [lbl for lbl in labels if lbl is not None]
            all_labels.update(labels)
    
    return sorted(all_labels)



