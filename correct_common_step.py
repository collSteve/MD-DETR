
# ==============================================================================
# The CORRECT and COMPLETE code for the common_step method is below.
# ==============================================================================

def common_step(self, batch, batch_idx, return_outputs=None, train=False, class_wise=True):
    pixel_values = batch["pixel_values"].to(self.device)
    pixel_mask = batch["pixel_mask"].to(self.device)
    labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    
    if self.args.use_prompts:
        with torch.no_grad():
            # Use the separate, frozen model to generate the query
            outputs = self.query_function_model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels,  train=False, task_id=self.task_id)

            if not self.args.local_query:
                query = outputs.last_hidden_state.mean(dim=1)
            else:
                query = outputs.last_hidden_state

                if self.args.record_queries and not train:
                    for i in range(len(labels)):
                        record = QueryRecord(
                            epoch=self.current_epoch,
                            image_id=labels[i]['image_id'].item(),
                            task_id=self.task_id,
                            object_queries=query[i],
                            gt_class_ids=labels[i]['class_labels'].tolist()
                        )
                        self.query_probe(record)

                # THIS IS THE BLOCK THAT WAS INCORRECTLY DELETED. IT IS NOW RESTORED.
                outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs" and k != "enc_outputs"}
                indices = self.model.matcher(outputs_without_aux, labels)
                one_hot_proposals = torch.zeros((len(labels),300)).to(self.device)
                for i,ind in enumerate(indices):
                    for j in ind[0]:
                        one_hot_proposals[i][j] = 1
                query_wt = self.model.model.prompts.query_tf(query.view(query.shape[0],-1))
                query_loss = F.cross_entropy(query_wt, one_hot_proposals)
                
            if self.args.bg_thres and not return_outputs:
                results = self.processor.post_process(outputs, target_sizes=orig_target_sizes, bg_thres_topk=self.args.bg_thres_topk)
    else:
        query = None
    
    # BG thresholding on previously seen classes
    if self.args.bg_thres and not return_outputs and self.args.use_prompts:
        labels = self.BG_thresholding(results=results, labels=labels)

    # Main forward pass with the training model
    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, query=query, train=True, task_id=self.task_id)

    loss = outputs.loss
    loss_dict = outputs.loss_dict

    if self.args.local_query and self.args.use_prompts:
        loss_dict['query_loss'] = query_loss
        loss += self.args.lambda_query * query_loss

    if return_outputs:
        if self.args.mask_gradients:
            outputs.logits[:,:, self.invalid_cls_logits] = -10e10
            outputs.logits = outputs.logits[:,:,:self.args.n_classes-1]
        results = self.processor.post_process(outputs, target_sizes=orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(labels, results)}
        res = self.evaluator.prepare_for_coco_detection(res)
        return loss, loss_dict, res

    return loss, loss_dict
