
# ==============================================================================
# The code for the forward method in DeformableDetrForObjectDetection is below.
# ==============================================================================

def forward(
    self,
    pixel_values: torch.FloatTensor,
    pixel_mask: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.FloatTensor] = None,
    encoder_outputs: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[List[dict]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    query=None,
    train=False,
    task_id=0,
    switch_off_prompts=False,
) -> Union[Tuple[torch.FloatTensor], DeformableDetrObjectDetectionOutput]:
    r"""
    labels (`List[Dict]` of len `(batch_size,)`, *optional*):
        Labels for computing the bipartite matching loss. List of dicts, each dictionary containing:
            - 'class_labels': a `torch.LongTensor` of shape `(number_of_objects,)`
            - 'boxes': a `torch.FloatTensor` of shape `(number_of_objects, 4)`
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # First, sent images through DETR base model to get encoder-decoder outputs
    outputs = self.model(
        pixel_values,
        pixel_mask=pixel_mask,
        decoder_attention_mask=decoder_attention_mask,
        encoder_outputs=encoder_outputs,
        inputs_embeds=inputs_embeds,
        decoder_inputs_embeds=decoder_inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        query=query,
        train=train,
        task_id=task_id,
        switch_off_prompts=switch_off_prompts,
    )

    # THIS IS THE FIX: Check if prompts exist before trying to access them.
    if hasattr(self.model, 'prompts') and self.model.prompts is not None:
        self.model.prompts.set_image_ids([label["image_id"] for label in labels])

    hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
    init_reference_points = outputs.init_reference_points if return_dict else outputs[0]
    inter_reference_points = outputs.intermediate_reference_points if return_dict else outputs[3]

    # perform predictions
    # ... (rest of the function is unchanged)
