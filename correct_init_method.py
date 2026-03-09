
	def __init__(self, train_loader, val_loader, test_dataset, args, local_evaluator, task_id, eval_mode=False):
		super().__init__()

		# --- Main Model Initialization ---
		detr_config = DeformableDetrConfig()
		detr_config.num_labels = args.n_classes #+ 1
		detr_config.PREV_INTRODUCED_CLS = args.task_map[task_id][1]
		detr_config.CUR_INTRODUCED_CLS = args.task_map[task_id][2]
		seen_classes = detr_config.PREV_INTRODUCED_CLS + detr_config.CUR_INTRODUCED_CLS
		
		detr_config.use_prompts = args.use_prompts
		detr_config.n_tasks = args.n_tasks
		detr_config.num_prompts = args.num_prompts
		detr_config.prompt_len = args.prompt_len
		detr_config.local_query = args.local_query
		detr_config.use_correspondence_embedding = args.use_correspondence_embedding
		detr_config.use_positional_embedding_for_correspondence = args.use_positional_embedding_for_correspondence
		detr_config.dual_memory_strategy = args.dual_memory_strategy
		detr_config.dual_memory_switch_layer = args.dual_memory_switch_layer
		detr_config.q_to_ek_strategy = args.q_to_ek_strategy

		self.invalid_cls_logits = list(range(seen_classes, args.n_classes-1))
		ModelClass = get_model_class(args.use_dual_memory_model)

		if args.repo_name:
			self.model =  ModelClass.from_pretrained(args.repo_name,config=detr_config,
																	ignore_mismatched_sizes=True,
																	default=not(args.mask_gradients), log_file=args.log_file)
			self.processor = DeformableDetrImageProcessor.from_pretrained(args.repo_name)
		else:
			self.model = ModelClass(detr_config, default=not(args.mask_gradients),
												 log_file=args.log_file)
			self.processor = DeformableDetrImageProcessor()

		# --- Query Function Model Initialization (NEW) ---
		print("--- [ENGINE_TEST] Initializing separate, frozen query function model ---")
		query_fn_config = deepcopy(detr_config)
		query_fn_config.use_prompts = False # Ensure it's a vanilla DETR
		
		QueryFnModelClass = get_model_class(False) # Always use the standard model for the query function

		if args.repo_name:
			self.query_function_model = QueryFnModelClass.from_pretrained(args.repo_name, config=query_fn_config,
																		  ignore_mismatched_sizes=True)
		else:
			self.query_function_model = QueryFnModelClass(query_fn_config)
		
		# Freeze the entire query function model
		for param in self.query_function_model.parameters():
			param.requires_grad = False
		self.query_function_model.eval()
		print("--- [ENGINE_TEST] Query function model is frozen. ---")

		# --- Rest of __init__ ---
		if getattr(self.model.model, 'prompts', None):
			prompts = self.model.model.prompts
			if isinstance(prompts, ClassWiseDynamicPrompt):
				object_class_names = []
				for tid in range(1, task_id+1):
					class_names, _, _ = args.task_map[tid]
					object_class_names.extend(class_names)
				stadardized_object_class_names = [stardardize_object_class_name(name) for name in object_class_names]
				prompts.initialize_for_task(task_id, object_classes=stadardized_object_class_names)
			else:
				for tid in range(1, task_id+1):
						prompts.initialize_for_task(tid)
			prompts.set_task_id(task_id - 1)
			prompts.reset_parameters()
			self.prompts = prompts

		find_param_nans(self.model)

		self.mem_probe = MemoryProbe(out_dir=f"{args.output_dir}/mem_trace/mem_traces_task{task_id}")
		self.query_probe = QueryProbe(out_dir=f"{args.output_dir}/query_probe/query_traces_task{task_id}")
		
		self.task_id = task_id
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_dataset = test_dataset # THIS LINE WAS MISSING
		self.args = args
		self.eval_mode = eval_mode
		self.print_count = 0
		self.evaluator = local_evaluator
		self.evaluator.model = self.model
		self.evaluator.invalid_cls_logits = self.invalid_cls_logits
		self.PREV_INTRODUCED_CLS = args.task_map[task_id][1]
