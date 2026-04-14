import os
import yaml
import socket
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision

def setup(rank, world_size, port='9987'):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = port
	os.environ["NCCL_DEBUG"] = "INFO"

	dist.init_process_group("nccl", rank=rank, world_size=world_size)
		
def setup_for_distributed(is_master):
	"""
	This function disables printing when not in master process
	"""
	import builtins as __builtin__
	builtin_print = __builtin__.print

	def print(*args, **kwargs):
		force = kwargs.pop('force', False)
		if is_master or force:
			builtin_print(*args, **kwargs)

	__builtin__.print = print

def match_name_keywords(n, name_keywords):
	out = False
	for b in name_keywords:
		if b in n:
			out = True
			break
	return out

def print_final(out_dir, start_task=1, n_tasks=4):
	outputs = []
	for i in range(start_task,n_tasks+1):
		outputs.append('------------------------------------------------------------------------------- \n')
		outputs.append('Evaluating Task '+str(i)+'\n')
		outputs.append('------------------------------------------------------------------------------- \n\n')
		stats = open(out_dir+'/Task_'+str(i)+'/stats.txt').readlines()
		outputs.extend(stats)
	
	with open(out_dir+'/final_stats.txt', 'w') as f:
		f.writelines(outputs)
	f.close()

def collate_fn(batch, processor):
  pixel_values = [item[0] for item in batch]
  #print(pixel_values)
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch


def map_task_ids(dataset, task_id_names):
	base_ds = dataset.coco
	ids = base_ds.getCatIds(catNms=task_id_names)
	class_ids=[]

	for i in ids:
		class_ids+=base_ds.getImgIds(catIds=[i])

	print ('Number of samples', len(class_ids), len(set(class_ids)))
	dataset.ids = list(set(class_ids))

	return dataset, base_ds

class SmoothedValue(object):
	"""Track a series of values and provide access to smoothed values over a
	window or the global series average.
	"""

	def __init__(self, window_size=20, fmt=None):
		if fmt is None:
			fmt = "{median:.4f} ({global_avg:.4f})"
		self.deque = deque(maxlen=window_size)
		self.total = 0.0
		self.count = 0
		self.fmt = fmt

	def update(self, value, n=1):
		self.deque.append(value)
		self.count += n
		self.total += value * n

	def synchronize_between_processes(self):
		"""
		Warning: does not synchronize the deque!
		"""
		if not is_dist_avail_and_initialized():
			return
		t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
		dist.barrier()
		dist.all_reduce(t)
		t = t.tolist()
		self.count = int(t[0])
		self.total = t[1]

	@property
	def median(self):
		d = torch.tensor(list(self.deque))
		return d.median().item()

	@property
	def avg(self):
		d = torch.tensor(list(self.deque), dtype=torch.float32)
		return d.mean().item()

	@property
	def global_avg(self):
		return self.total / self.count

	@property
	def max(self):
		return max(self.deque)

	@property
	def value(self):
		return self.deque[-1]

	def __str__(self):
		return self.fmt.format(
			median=self.median,
			avg=self.avg,
			global_avg=self.global_avg,
			max=self.max,
			value=self.value)

def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def all_gather(data):
	"""
	Run all_gather on arbitrary picklable data (not necessarily tensors)
	Args:
		data: any picklable object
	Returns:
		list[data]: list of data gathered from each rank
	"""
	world_size = get_world_size()
	if world_size == 1:
		return [data]

	# serialized to a Tensor
	buffer = pickle.dumps(data)
	storage = torch.ByteStorage.from_buffer(buffer)
	tensor = torch.ByteTensor(storage).to("cuda")

	# obtain Tensor size of each rank
	local_size = torch.tensor([tensor.numel()], device="cuda")
	size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
	dist.all_gather(size_list, local_size)
	size_list = [int(size.item()) for size in size_list]
	max_size = max(size_list)

	# receiving Tensor from all ranks
	# we pad the tensor because torch all_gather does not support
	# gathering tensors of different shapes
	tensor_list = []
	for _ in size_list:
		tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
	if local_size != max_size:
		padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
		tensor = torch.cat((tensor, padding), dim=0)
	dist.all_gather(tensor_list, tensor)

	data_list = []
	for size, tensor in zip(size_list, tensor_list):
		buffer = tensor.cpu().numpy().tobytes()[:size]
		data_list.append(pickle.loads(buffer))

	return data_list


def reduce_dict(input_dict, average=True):
	"""
	Args:
		input_dict (dict): all the values will be reduced
		average (bool): whether to do average or sum
	Reduce the values in the dictionary from all processes so that all processes
	have the averaged results. Returns a dict with the same fields as
	input_dict, after reduction.
	"""
	world_size = get_world_size()
	if world_size < 2:
		return input_dict
	with torch.no_grad():
		names = []
		values = []
		# sort the keys so that they are consistent across processes
		for k in sorted(input_dict.keys()):
			names.append(k)
			values.append(input_dict[k])
		values = torch.stack(values, dim=0)
		dist.all_reduce(values)
		if average:
			values /= world_size
		reduced_dict = {k: v for k, v in zip(names, values)}
	return reduced_dict


class MetricLogger(object):
	def __init__(self, delimiter="\t"):
		self.meters = defaultdict(SmoothedValue)
		self.delimiter = delimiter

	def update(self, **kwargs):
		for k, v in kwargs.items():
			if isinstance(v, torch.Tensor):
				v = v.item()
			assert isinstance(v, (float, int))
			self.meters[k].update(v)

	def __getattr__(self, attr):
		if attr in self.meters:
			return self.meters[attr]
		if attr in self.__dict__:
			return self.__dict__[attr]
		raise AttributeError("'{}' object has no attribute '{}'".format(
			type(self).__name__, attr))

	def __str__(self):
		loss_str = []
		for name, meter in self.meters.items():
			loss_str.append(
				"{}: {}".format(name, str(meter))
			)
		return self.delimiter.join(loss_str)

	def synchronize_between_processes(self):
		for meter in self.meters.values():
			meter.synchronize_between_processes()

	def add_meter(self, name, meter):
		self.meters[name] = meter

	def log_every(self, iterable, print_freq, header=None):
		i = 0
		if not header:
			header = ''
		start_time = time.time()
		end = time.time()
		iter_time = SmoothedValue(fmt='{avg:.4f}')
		data_time = SmoothedValue(fmt='{avg:.4f}')
		space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
		if torch.cuda.is_available():
			log_msg = self.delimiter.join([
				header,
				'[{0' + space_fmt + '}/{1}]',
				'eta: {eta}',
				'{meters}',
				'time: {time}',
				'data: {data}',
				'max mem: {memory:.0f}'
			])
		else:
			log_msg = self.delimiter.join([
				header,
				'[{0' + space_fmt + '}/{1}]',
				'eta: {eta}',
				'{meters}',
				'time: {time}',
				'data: {data}'
			])
		MB = 1024.0 * 1024.0
		for obj in iterable:
			data_time.update(time.time() - end)
			yield obj
			iter_time.update(time.time() - end)
			if i % print_freq == 0 or i == len(iterable) - 1:
				eta_seconds = iter_time.global_avg * (len(iterable) - i)
				eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
				if torch.cuda.is_available():
					print(log_msg.format(
						i, len(iterable), eta=eta_string,
						meters=str(self),
						time=str(iter_time), data=str(data_time),
						memory=torch.cuda.max_memory_allocated() / MB))
				else:
					print(log_msg.format(
						i, len(iterable), eta=eta_string,
						meters=str(self),
						time=str(iter_time), data=str(data_time)))
			i += 1
			end = time.time()
		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print('{} Total time: {} ({:.4f} s / it)'.format(
			header, total_time_str, total_time / len(iterable)))


def compute_memory_orthogonality_loss(prompts, task_id, device):
	"""
	Compute orthogonality regularization loss for memory keys.

	Implements Approach 3: Inter-task + Intra-task orthogonality.
	- Inter-task: Forces K vectors from different tasks to be orthogonal
	- Intra-task: Forces K vectors within current task to be orthogonal

	Args:
		prompts: Memory module (DynamicPrompt or subclass like SimpleProposalMemory)
		task_id: Current task ID (int)
		device: torch.device for creating identity matrices

	Returns:
		tuple: (loss_inter, loss_intra)
			- loss_inter: Inter-task orthogonality loss (scalar tensor)
			- loss_intra: Intra-task orthogonality loss (scalar tensor)
	"""
	import torch.nn.functional as F

	loss_ortho_inter = torch.tensor(0.0, device=device)
	loss_ortho_intra = torch.tensor(0.0, device=device)

	# Check if prompts has layer_memories attribute
	if not hasattr(prompts, 'layer_memories'):
		return loss_ortho_inter, loss_ortho_intra

	for layer_name, task_dict in prompts.layer_memories.items():
		K_current_list = []
		K_old_list = []

		for tid, mem in task_dict.items():
			_, K, _ = mem.forward()  # (U_task, 256)

			if tid == str(task_id):
				K_current_list.append(K)
			else:
				# CRITICAL: Detach old memories to prevent gradient flow to frozen tasks
				K_old_list.append(K.detach())

		# Only need current task data to compute losses
		if K_current_list:
			K_current = torch.cat(K_current_list, dim=0)  # (U_current, 256)
			K_current_norm = F.normalize(K_current, dim=1)

			# Intra-task orthogonality: Always compute for current task
			# Encourages diversity among memory units within the task
			n = len(K_current)
			self_gram = K_current_norm @ K_current_norm.T  # (U_current, U_current)
			identity = torch.eye(n, device=device)
			loss_ortho_intra = loss_ortho_intra + (self_gram - identity).pow(2).sum()

			# Inter-task orthogonality: Only compute when old tasks exist
			# Prevents interference between current and previous task memories
			if K_old_list:
				K_old = torch.cat(K_old_list, dim=0)  # (U_old, 256)
				K_old_norm = F.normalize(K_old, dim=1)
				cross_gram = K_current_norm @ K_old_norm.T  # (U_current, U_old)
				loss_ortho_inter = loss_ortho_inter + cross_gram.pow(2).sum()

	return loss_ortho_inter, loss_ortho_intra


def save_experiment_config(args, out_dir, engine_name):
    """Save full experiment configuration to YAML and print a summary.

    Creates 'experiment_config.yaml' in out_dir with structured key parameters
    for quick reference and full reproducibility.
    """
    memory_type = 'SelectiveProposalMemory' if getattr(args, 'use_selective_memory', False) else 'SimpleProposalMemory'

    config = {
        'experiment': {
            'timestamp': datetime.datetime.now().isoformat(),
            'engine': engine_name,
            'hostname': socket.gethostname(),
        },
        'memory': {
            'type': memory_type,
            'injection_strategy': getattr(args, 'injection_strategy', 'prefix'),
            'memory_focus': getattr(args, 'memory_focus', 10.0),
            'num_null_units': getattr(args, 'num_null_units', 2),
            'use_prompts': args.use_prompts,
            'local_query': args.local_query,
        },
        'losses': {
            'use_bg_suppression': getattr(args, 'use_bg_suppression', False),
            'lambda_bg': getattr(args, 'lambda_bg', 0.1),
            'use_ortho_regularization': getattr(args, 'use_ortho_regularization', False),
            'lambda_ortho_inter': args.lambda_ortho_inter,
            'lambda_ortho_intra': args.lambda_ortho_intra,
            'use_query_loss': getattr(args, 'use_query_loss', False),
            'lambda_query': args.lambda_query,
            'cls_loss_coef': args.cls_loss_coef,
            'bbox_loss_coef': args.bbox_loss_coef,
            'giou_loss_coef': args.giou_loss_coef,
            'focal_alpha': args.focal_alpha,
            'set_cost_class': args.set_cost_class,
            'set_cost_bbox': args.set_cost_bbox,
            'set_cost_giou': args.set_cost_giou,
        },
        'training': {
            'lr': args.lr,
            'lr_old': args.lr_old,
            'weight_decay': args.weight_decay,
            'clip_max_norm': args.clip_max_norm,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'save_epochs': args.save_epochs,
            'eval_epochs': args.eval_epochs,
            'freeze': args.freeze,
            'new_params': args.new_params,
            'seed': args.seed,
            'resume': args.resume,
            'n_gpus': args.n_gpus,
        },
        'continual': {
            'n_tasks': args.n_tasks,
            'start_task': args.start_task,
            'n_classes': args.n_classes,
            'split_point': args.split_point,
            'task_order': getattr(args, 'task_order', None),
            'bg_thres': args.bg_thres,
            'bg_thres_topk': args.bg_thres_topk,
            'mask_gradients': args.mask_gradients,
        },
        'other_features': {
            'use_correspondence_embedding': getattr(args, 'use_correspondence_embedding', False),
            'use_positional_embedding_for_correspondence': getattr(args, 'use_positional_embedding_for_correspondence', False),
            'use_dual_memory_model': getattr(args, 'use_dual_memory_model', False),
            'dual_memory_strategy': getattr(args, 'dual_memory_strategy', 'hybrid_everywhere'),
            'q_to_ek_strategy': getattr(args, 'q_to_ek_strategy', 'query_bias'),
        },
        'paths': {
            'output_dir': args.output_dir,
            'repo_name': args.repo_name,
            'checkpoint_dir': args.checkpoint_dir,
            'checkpoint_base': args.checkpoint_base,
            'checkpoint_next': args.checkpoint_next,
            'train_img_dir': args.train_img_dir,
            'test_img_dir': args.test_img_dir,
            'task_ann_dir': args.task_ann_dir,
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, 'experiment_config.yaml')
    with open(config_path, 'w') as f:
        f.write(f"# Experiment configuration — {engine_name}\n")
        f.write(f"# Generated: {config['experiment']['timestamp']}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    m = config['memory']
    lo = config['losses']
    t = config['training']
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"  Engine:           {engine_name}")
    print(f"  Memory:           {m['type']}")
    print(f"  Injection:        {m['injection_strategy']}")
    print(f"  Focus:            {m['memory_focus']}")
    print(f"  Null units:       {m['num_null_units']}")
    print(f"  BG suppression:   {lo['use_bg_suppression']} (lambda={lo['lambda_bg']})")
    print(f"  Ortho:            {lo['use_ortho_regularization']} (inter={lo['lambda_ortho_inter']}, intra={lo['lambda_ortho_intra']})")
    print(f"  Query loss:       {lo['use_query_loss']} (lambda={lo['lambda_query']})")
    print(f"  Freeze:           {t['freeze']}")
    print(f"  LR:               {t['lr']} / LR_old: {t['lr_old']}")
    print(f"  Epochs:           {t['epochs']} / Batch: {t['batch_size']}")
    print(f"  Config saved to:  {config_path}")
    print("=" * 60)