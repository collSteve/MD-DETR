# ==============================================================================
# The CORRECT and FINAL code for the resume method is below.
# ==============================================================================

def resume(self, load_path=''):
    print('\n Resuming model for task ', self.task_id, ' from : ',load_path, file=self.args.log_file)
    if load_path:
        checkpoint = torch.load(load_path, map_location='cpu')
        # Load state ONLY into the main training model.
        # The query_function_model remains in its pristine, off-the-shelf state.
        self.model.load_state_dict(checkpoint['model'], strict=False)

    if not self.args.eval and self.args.freeze:
        
        freeze = self.args.freeze.split(',')
        for id, (name, params) in enumerate(self.model.named_parameters()):
            params.requires_grad = True
            flag = False
            for n in name.split('.'):
                if n in freeze:
                    params.requires_grad = False
                    flag = True
            if not flag:
                print ('Trainable ..', name, "  Req grad .. ",params.requires_grad, file=self.args.log_file)

