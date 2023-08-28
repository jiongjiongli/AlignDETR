# AlignDETR

## Installation

### Old

```

module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

cd ~/align_detr
mkdir legacy
cd ~/align_detr/legacy

python -m venv  ~/align_detr/legacy/py38
source ~/align_detr/legacy/py38/bin/activate

pip install --upgrade pip
pip install ipython

# Option 1: install directly.
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Option 2: Download and then install.
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl

pip install ~/data/torch-1.9.0+cu111-cp38-cp38-linux_x86_64.whl
pip install ~/data/torchvision-0.10.0+cu111-cp38-cp38-linux_x86_64.whl

git clone https://github.com/IDEA-Research/detrex.git
cd detrex
git submodule init
git submodule update

pip install -e detectron2
pip install -e .


```



### New

```
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

cd ~/align_detr
mkdir new
cd ~/align_detr/new

python -m venv  ~/align_detr/new/align_py38

source ~/align_detr/new/align_py38/bin/activate

pip install --upgrade pip
pip install ipython


# Option 1: install directly.
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Option 2: Download and then install.
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl
wget -P ~/data/ https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp38-cp38-linux_x86_64.whl

pip install ~/data/torch-1.9.0+cu111-cp38-cp38-linux_x86_64.whl
pip install ~/data/torchvision-0.10.0+cu111-cp38-cp38-linux_x86_64.whl

pip install mmengine==0.8.4
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html

git clone -b align_detr https://gitee.com/jiongjiongli/mmdetection_dev.git mmdetection

cd ~/align_detr/new/mmdetection
pip install -r requirements.txt

```



## Train

### Launch

#### Old

```python
python tools/train_net.py --config-file  projects/aligndetr/configs/aligndetr_k=2_r50_4scale_12ep.py --num-gpus 8

python tools/train_net.py --config-file  projects/aligndetr/configs/aligndetr_k=2_r50_4scale_12ep.py dataloader.train.num_workers=2 dataloader.train.total_batch_size=2 train.log_period=1
```



#### New

```
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

source ~/align_detr/new/align_py38/bin/activate

cd ~/align_detr/new/mmdetection
# pip install -v -e .
export PYTHONPATH=$PYTHONPATH:

export CUDA_VISIBLE_DEVICES=-1

python tools/train.py configs/align_detr/align_detr-4scale_r50_8xb2-12e_coco.py --cfg-options model.backbone.init_cfg=None


```





# Prepare Data



## New

```
mkdir -p ~/align_detr/new/mmdetection/data/
ln -s ~/data/lvis ~/align_detr/new/mmdetection/data/coco

```



## train_net

```
tools/train_net.py

# D:\proj\git\detrex\detectron2\detectron2\engine\defaults.py
parser = default_argument_parser()
args = parser.parse_args()

# D:\proj\git\detrex\detectron2\detectron2\engine\launch.py
  ->
        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,               = main
                world_size,              = num_machines * num_gpus_per_machine
                num_gpus_per_machine,    = args.num_gpus
                machine_rank,            = args.machine_rank
                dist_url,                = args.dist_url
                args,                    = (args,)
                timeout,                 = DEFAULT_TIMEOUT
            ),
            daemon=False,
        )
#
launch(
    main,
    args.num_gpus,
    num_machines=args.num_machines,
    machine_rank=args.machine_rank,
    dist_url=args.dist_url,
    args=(args,),
)
```



```



def main(args):
	# D:\proj\git\detrex\detectron2\detectron2\config\lazy.py
	# Returns DictConfig
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    
    # D:\proj\git\detrex\detectron2\detectron2\engine\defaults.py
    	->
    	output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    	PathManager.mkdirs(output_dir)
    	setup_logger(output_dir, ...)
    	Save config to output_dir / config.yaml.
    	Set seed.
    	Set torch.backends.cudnn.benchmark.
    #
    default_setup(cfg, args)
    # import ipdb;ipdb.set_trace()
    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model,cfg.train.output_dir).resume_or_load(cfg.train.init_checkpoint)
        do_test(cfg, model)
    else:
	    # D:\proj\git\AlignDETR\tools\train_net.py
        do_train(args, cfg)


```



## do_train

```
D:\proj\git\AlignDETR\tools\train_net.py

def do_train(args, cfg):
	model = instantiate(cfg.model)
	model.to(cfg.train.device)
	cfg.optimizer.params.model = model
	
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
	-> build_detection_train_loader

    model = create_ddp_model(model, **cfg.train.ddp)
    
    # Trainer: SimpleTrainer: TrainerBase
    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )

	checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )
    -> D:\proj\git\detrex\detectron2\detectron2\engine\train_loop.py
    	TrainerBase.register_hooks
    	
    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
	
    trainer.train(start_iter, cfg.train.max_iter)
	-> 	D:\proj\git\detrex\detectron2\detectron2\engine\train_loop.py
		TrainerBase.train
		
        ...
        for self.iter in range(start_iter, max_iter):
            self.run_step()
            -> Trainer.run_step
            	
            	# self._data_loader_iter=iter(self.data_loader)
            	data = next(self._data_loader_iter)
            	
            	loss_dict = self.model(data)
            	losses = sum(loss_dict.values())
            	
            	self.optimizer.zero_grad()
                losses.backward()
                if self.clip_grad_params is not None:
                    self.clip_grads(self.model.parameters())
                self.optimizer.step()

            	self._write_metrics(loss_dict, data_time)

```



## Config

```yaml
# %load /kaggle/working/AlignDETR/output/aligndetr_k2_12ep/config.yaml
dataloader:
  evaluator: {_target_: detectron2.evaluation.COCOEvaluator, dataset_name: '${..test.dataset.names}', output_dir: ./output/aligndetr_k2_12ep}
  test:
    _target_: detectron2.data.build_detection_test_loader
    dataset: {_target_: detectron2.data.get_detection_dataset_dicts, filter_empty: false, names: coco_2017_val}
    mapper:
      _target_: detrex.data.DetrDatasetMapper
      augmentation:
      - {_target_: detectron2.data.transforms.ResizeShortestEdge, max_size: 1333, short_edge_length: 800}
      augmentation_with_crop: null
      img_format: RGB
      is_train: false
      mask_on: false
    num_workers: 4
  train:
    _target_: detectron2.data.build_detection_train_loader
    dataset: {_target_: detectron2.data.get_detection_dataset_dicts, names: coco_2017_train}
    mapper:
      _target_: detrex.data.DetrDatasetMapper
      augmentation:
      - {_target_: detectron2.data.transforms.RandomFlip}
      - _target_: detectron2.data.transforms.ResizeShortestEdge
        max_size: 1333
        sample_style: choice
        short_edge_length: [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
      augmentation_with_crop:
      - {_target_: detectron2.data.transforms.RandomFlip}
      - _target_: detectron2.data.transforms.ResizeShortestEdge
        sample_style: choice
        short_edge_length: [400, 500, 600]
      - _target_: detectron2.data.transforms.RandomCrop
        crop_size: [384, 600]
        crop_type: absolute_range
      - _target_: detectron2.data.transforms.ResizeShortestEdge
        max_size: 1333
        sample_style: choice
        short_edge_length: [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
      img_format: RGB
      is_train: true
      mask_on: false
    num_workers: 2
    total_batch_size: 2
lr_multiplier:
  _target_: detectron2.solver.WarmupParamScheduler
  scheduler:
    _target_: fvcore.common.param_scheduler.MultiStepParamScheduler
    milestones: [82500, 90000]
    values: [1.0, 0.1]
  warmup_factor: 0.001
  warmup_length: 0.0
  warmup_method: linear
model:
  _target_: projects.aligndetr.modeling.AlignDETR
  aux_loss: true
  backbone:
    _target_: detectron2.modeling.ResNet
    freeze_at: 1
    out_features: [res3, res4, res5]
    stages: {_target_: detectron2.modeling.ResNet.make_default_stages, depth: 50, norm: FrozenBN, stride_in_1x1: false}
    stem: {_target_: detectron2.modeling.backbone.BasicStem, in_channels: 3, norm: FrozenBN, out_channels: 64}
  box_noise_scale: 1.0
  criterion:
    _target_: aligndetr.criterions.AlignDETRCriterion
    alpha: 0.25
    gamma: 2.0
    match_num: [2, 2, 2, 2, 2, 2, 1]
    matcher: {_target_: aligndetr.matchers.MixedMatcher, alpha: 0.25, cost_bbox: 5.0, cost_class: 2.0, cost_class_type: focal_loss_cost, cost_giou: 2.0, gamma: 2.0}
    num_classes: ${..num_classes}
    tau: 1.5
    two_stage_binary_cls: false
    weight_dict: {loss_bbox: 5.0, loss_bbox_0: 5.0, loss_bbox_1: 5.0, loss_bbox_2: 5.0, loss_bbox_3: 5.0, loss_bbox_4: 5.0, loss_bbox_dn: 5.0, loss_bbox_dn_0: 5.0, loss_bbox_dn_1: 5.0, loss_bbox_dn_2: 5.0, loss_bbox_dn_3: 5.0, loss_bbox_dn_4: 5.0, loss_bbox_dn_enc: 5.0, loss_bbox_enc: 5.0, loss_class: 1, loss_class_0: 1, loss_class_1: 1, loss_class_2: 1, loss_class_3: 1, loss_class_4: 1, loss_class_dn: 1, loss_class_dn_0: 1, loss_class_dn_1: 1, loss_class_dn_2: 1, loss_class_dn_3: 1, loss_class_dn_4: 1, loss_class_dn_enc: 1, loss_class_enc: 1, loss_giou: 2.0, loss_giou_0: 2.0, loss_giou_1: 2.0, loss_giou_2: 2.0, loss_giou_3: 2.0, loss_giou_4: 2.0, loss_giou_dn: 2.0, loss_giou_dn_0: 2.0, loss_giou_dn_1: 2.0, loss_giou_dn_2: 2.0, loss_giou_dn_3: 2.0, loss_giou_dn_4: 2.0, loss_giou_dn_enc: 2.0, loss_giou_enc: 2.0}
  device: cuda
  dn_number: 100
  embed_dim: 256
  label_noise_ratio: 0.5
  neck:
    _target_: detrex.modeling.ChannelMapper
    in_features: [res3, res4, res5]
    input_shapes:
      res3: !!python/object:detectron2.layers.shape_spec.ShapeSpec {channels: 512, height: null, stride: null, width: null}
      res4: !!python/object:detectron2.layers.shape_spec.ShapeSpec {channels: 1024, height: null, stride: null, width: null}
      res5: !!python/object:detectron2.layers.shape_spec.ShapeSpec {channels: 2048, height: null, stride: null, width: null}
    kernel_size: 1
    norm_layer: {_target_: torch.nn.GroupNorm, num_channels: 256, num_groups: 32}
    num_outs: 4
    out_channels: 256
  num_classes: 80
  num_queries: 900
  pixel_mean: [123.675, 116.28, 103.53]
  pixel_std: [58.395, 57.12, 57.375]
  position_embedding: {_target_: detrex.layers.PositionEmbeddingSine, normalize: true, num_pos_feats: 128, offset: -0.5, temperature: 10000}
  prior_init: 0.01
  transformer:
    _target_: projects.aligndetr.modeling.Transformer
    decoder: {_target_: projects.aligndetr.modeling.TransformerDecoder, attn_dropout: 0.0, embed_dim: 256, feedforward_dim: 2048, ffn_dropout: 0.0, num_feature_levels: '${..num_feature_levels}', num_heads: 8, num_layers: 6, return_intermediate: true}
    encoder: {_target_: projects.aligndetr.modeling.TransformerEncoder, attn_dropout: 0.0, embed_dim: 256, feedforward_dim: 2048, ffn_dropout: 0.0, num_feature_levels: '${..num_feature_levels}', num_heads: 8, num_layers: 6, post_norm: false}
    num_feature_levels: 4
    two_stage_num_proposals: ${..num_queries}
optimizer:
  _target_: torch.optim.AdamW
  betas: [0.9, 0.999]
  lr: 0.0001
  params: {_target_: detectron2.solver.get_default_optimizer_params, base_lr: '${..lr}', lr_factor_func: !!python/name:None.%3Clambda%3E '', weight_decay_norm: 0.0}
  weight_decay: 0.0001
train:
  amp: {enabled: false}
  checkpointer: {max_to_keep: 100, period: 1}
  clip_grad:
    enabled: true
    params: {max_norm: 0.1, norm_type: 2}
  ddp: {broadcast_buffers: false, find_unused_parameters: false, fp16_compression: false}
  device: cuda
  eval_period: 3
  fast_dev_run: {enabled: false}
  init_checkpoint: detectron2://ImageNetPretrained/torchvision/R-50.pkl
  log_period: 1
  max_iter: 3
  model_ema: {decay: 0.999, device: '', enabled: false, use_ema_weights_for_eval_only: false}
  output_dir: ./output/aligndetr_k2_12ep
  seed: -1
  wandb:
    enabled: false
    params: {dir: ./wandb_output, name: detrex_experiment, project: detrex}
```





## Model

### AlignDETR.init

| Atrribute Name                                     | Input Variable Name            | Input Variable Value                    |
| -------------------------------------------------- | ------------------------------ | --------------------------------------- |
| backbone                                           | backbone                       | detectron2.modeling.ResNet              |
| position_embedding                                 | position_embedding             | detrex.layers.PositionEmbeddingSine     |
| neck                                               | neck                           | detrex.modeling.ChannelMapper           |
| transformer                                        | transformer                    | projects.aligndetr.modeling.Transformer |
| embed_dim                                          | embed_dim                      | 256                                     |
| num_classes                                        | num_classes                    | 80                                      |
| num_queries                                        | num_queries                    | 900                                     |
| criterion                                          | criterion                      | aligndetr.criterions.AlignDETRCriterion |
| pixel_mean                                         | pixel_mean                     | [123.675, 116.28, 103.53]               |
| pixel_std                                          | pixel_std                      | [58.395, 57.12, 57.375]                 |
| normalizer                                         |                                |                                         |
| aux_loss                                           | aux_loss                       | true                                    |
| select_box_nums_for_evaluation                     | select_box_nums_for_evaluation | 300 # Default                           |
| device                                             | device                         | cuda                                    |
| dn_number                                          | dn_number                      | 100                                     |
| label_noise_ratio                                  | label_noise_ratio              | 0.5                                     |
| box_noise_scale                                    | box_noise_scale                | 1.0                                     |
| prior_prob                                         | prior_init                     | 0.01                                    |
| class_embed                                        |                                |                                         |
| bbox_embed                                         |                                |                                         |
| label_enc `= nn.Embedding(num_classes, embed_dim)` |                                |                                         |





### AlignDETR.forward

```
	def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        -> 	For each image: image - mean / std, to device.
        
        	images = ImageList inst.
        	images.tensor=batched_imgs, tensor, shape: [B, C, H=max_h, W=max_W]
        	images.image_sizes= [(image_h, image_w) for each image]
        
        img_masks, shape [B, H=max_h, W=max_W], 0 within image_h, image_w for each image.
        
        # Inputs 
        [B, C=3, H, W]
        # Outputs
        	{
        		'res3': shape [B, C= 512, H /  8, W /  8], 
        		'res4': shape [B, C=1024, H / 16, W / 16], 
        		'res5': shape [B, C=2048, H / 32, W / 32]
        	}
        #
        features = self.backbone(images.tensor)
        
        # self.neck: ChannelMapper inst.
        # Outputs
        # 	(
        		shape [B, C=256, H /  8, W /  8],
        		shape [B, C=256, H / 16, W / 16],
        		shape [B, C=256, H / 32, W / 32],
        		shape [B, C=256, H / 64, W / 64]
        	)
        #
        multi_level_feats = self.neck(features)
        
        multi_level_masks: [shape [B, Hi, Wi], ...]
        
        multi_level_position_embeddings: [shape [B, C=256, Hi, Wi], ...]
        
        targets = self.prepare_targets(gt_instances)
        -> [
        		{
        			'labels': [num_gt_bbox], gt class index.
        			'boxes': [num_gt_bbox, 4], normed gt cxcywh.
        		}, 
        		...
        	]
        
        input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number, 100
                label_noise_ratio=self.label_noise_ratio, 0.5
                box_noise_scale=self.box_noise_scale, 1.0
                num_queries=self.num_queries, 900
                num_classes=self.num_classes, 80
                hidden_dim=self.embed_dim, 256
                label_enc=self.label_enc, nn.Embedding(num_classes, embed_dim)
            )
        -> 
            # Outputs:
            input_query_label: shape [B, num_dino_groups * 2 * max_num_gt_bboxes, hidden_dim=256],
                gt class index embedding with noise.
            input_query_bbox : shape [B, num_dino_groups * 2 * max_num_gt_bboxes, 4],
                un_sigmoid gt cxcywh with noise.
            attn_mask: shape [tgt_size, tgt_size], masked=True,
                tgt_size = num_dino_groups * 2 * max_num_gt_bboxes + num_queries
            dn_meta = {
                "single_padding": 2 * max_num_gt_bboxes,
                "dn_num": num_dino_groups
            }
            #
            
        
        (
            inter_states, 
            	# shape [num_layers=6, B, num_dino_queries + num_queries, C]
            init_reference,
            	# shape [B, num_dino_queries + num_queries, 4], 
            	# denoise bbox input + detached encoder bbox output sigmoid cxcywh
            inter_references,
            	# shape [num_layers=6, B, num_dino_queries + num_queries, 4], sigmoid cxcywh
            enc_state,
            	# shape [B, num_queries, C]
                    memory + mask invalid point and memory_padding_mask point to 0.
                        + Linear + LayerNorm + topk.
                #
            enc_reference,
            	# shape [B, num_queries, 4], encoder bbox output sigmoid cxcywh
        ) 
        # projects.aligndetr.modeling.Transformer
        = self.transformer(
            multi_level_feats, # (shape [B, C=256, Hi, Wi], ...)
            multi_level_masks, # [shape [B, Hi, Wi], ...]
            multi_level_position_embeddings, # [shape [B, C=256, Hi, Wi], ...]
            query_embeds, 
            	# (
            		input_query_label: 
            			shape [B, num_dino_groups * 2 * max_num_gt_bboxes, hidden_dim=256],
                		gt class index embedding with noise.
            		input_query_bbox: 
            			shape [B, num_dino_groups * 2 * max_num_gt_bboxes, 4],
                		un_sigmoid gt cxcywh with noise.
                  )
            	#
            attn_masks=[attn_mask, None], 
            	# attn_mask: shape [tgt_size, tgt_size], masked=True,
            		tgt_size = num_dino_groups * 2 * max_num_gt_bboxes + num_queries
            	#
        )
        
        outputs_class, shape [num_layers=6, B, num_dino_queries + num_queries, num_classes]
        outputs_coord, shape [num_layers=6, B, num_dino_queries + num_queries, 4], sigmoid cxcywh
        
        outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )
            ->
            output_known_class, shape [num_layers=6, B, num_dino_queries, num_classes]
            output_known_coord, shape [num_layers=6, B, num_dino_queries, 4], sigmoid cxcywh
            dn_meta["output_known_lbs_bboxes"] = {
            	'pred_logits': output_known_class[-1], 
            	'pred_boxes': output_known_coord[-1], sigmoid cxcywh
            	'aux_outputs': self._set_aux_loss(output_known_class, output_known_coord)
            		-> [
            			{
                            'pred_logits': output_known_class[i], 0 <= i < 5
                            'pred_boxes': output_known_coord[i], sigmoid cxcywh
            			}, 
            			...
            		   ]
            	}
            outputs_class, shape [num_layers=6, B, num_queries, num_classes]
        	outputs_coord, shape [num_layers=6, B, num_queries, 4], sigmoid cxcywh
            
        output = {
        	'pred_logits': outputs_class[-1], 
        	'pred_boxes': outputs_coord[-1],, sigmoid cxcywh
        	'aux_outputs': self._set_aux_loss(outputs_class, outputs_coord)
        		-> [
                    {
                        'pred_logits': outputs_class[i], 0 <= i < 5
                        'pred_boxes': outputs_coord[i], sigmoid cxcywh
                    }, 
                    ...
                   ]
        	}
        
        # enc_reference, shape [B, num_queries, 4], encoder bbox output sigmoid cxcywh
		interm_coord = enc_reference
		
		# enc_state, shape [B, num_queries, C]
                    memory + mask invalid point and memory_padding_mask point to 0.
                        + Linear + LayerNorm + topk.
		#
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        
        output['enc_outputs'] = {
            'pred_logits': interm_class, shape [B, num_queries, num_classes]
            'pred_boxes': interm_coord, shape [B, num_queries, 4], encoder bbox output sigmoid cxcywh
            }

        if self.training:
        	# aligndetr.criterions.AlignDETRCriterion
            loss_dict = self.criterion(output, targets, dn_meta)
            ->
            # Inputs:
            ...,
            targets,
            [
        		{
        			'labels': [num_gt_bbox], gt class index.
        			'boxes': [num_gt_bbox, 4], normed gt cxcywh.
        		}, 
        		...
        	]
            ...
        	#
            
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
```



### AlignDETR.prepare_for_cdn

```
    def prepare_for_cdn(...):
    	# num_dino_groups
    	dn_number = max(1, 100 // max_num_gt)
    	
    	labels: shape [num_all_gt], all gt class index.
    	boxes: shape [num_all_gt, 4], all normed gt cxcywh.
    	batch_idx: shape [num_all_gt], batch index.
    	
    	known_labels: shape [num_dino_groups * 2 * num_all_gt], all gt class index for each group.
    	known_bid: shape [num_dino_groups * 2 * num_all_gt], batch index for each group.
    	known_bboxs: shape [num_dino_groups * 2 * num_all_gt, 4], all normed gt cxcywh for each group.
    	
    	# Randomly generate 1/4 gt class index.
    	known_labels_expaned: shape [num_dino_groups * 2 * num_all_gt]
    	
        known_bbox_expand = known_bboxs.clone()
        
        positive_idx: shape [num_dino_groups, num_all_gt], = [0, ... num_all_gt-1] for each group.
        positive_idx: shape [num_dino_groups * num_all_gt], = num_all_gt * 2 * group_index + [0, ... num_all_gt-1] for each group.
        
        negative_idx: shape [num_dino_groups * num_all_gt], = positive_idx + num_all_gt
        
        known_bbox_: shape [num_dino_groups * 2 * num_all_gt, 4], all normed gt x1y1x2y2 for each group.
        
        diff: shape [num_dino_groups * 2 * num_all_gt, 4], all normed gt [w/2, h/2, w/2, h/2] for each group.
        
        rand_sign: shape [num_dino_groups * 2 * num_all_gt, 4], random 1 / -1.
        rand_part: shape [num_dino_groups * 2 * num_all_gt, 4],  
        		   +/- [0, 1) for positive,
        		   +/- [1, 2) for negative.
        		   
		known_bbox_: shape [num_dino_groups * 2 * num_all_gt, 4], 
			all normed gt x1y1x2y2 for each group
                +/- [0, 1) * normed gt [w/2, h/2, w/2, h/2] for positive, 
                +/- [1, 2) * normed gt [w/2, h/2, w/2, h/2] for negative.
			Clamp to [0, 1].
			
		known_bbox_expand: shape [num_dino_groups * 2 * num_all_gt, 4], known_bbox_ to cxcywh.
		
		input_label_embed: shape [num_dino_groups * 2 * num_all_gt, embed_dim=256], 
			gt class index embedding with noise,
			= label_enc(known_labels_expaned.long().to("cuda"))
		
        input_bbox_embed: shape [num_dino_groups * 2 * num_all_gt, 4], 
        	un_sigmoid gt cxcywh with noise,
        	= inverse_sigmoid(known_bbox_expand),

		input_query_label: shape [B, num_dino_groups * 2 * max_num_gt_bboxes, hidden_dim=256], 0
		input_query_bbox: shape [B, num_dino_groups * 2 * max_num_gt_bboxes, 4], 0
		
		map_known_indice: shape [ num_dino_groups * 2 * num_all_gt]
		
		tgt_size = num_dino_groups * 2 * max_num_gt_bboxes + num_queries
		attn_mask, shape [tgt_size, tgt_size], = False
		# Queries cannot see dino groups.
		# Different groups in `num_dino_groups` can not see each other.
		
		dn_meta = {
            "single_padding": 2 * max_num_gt_bboxes,
            "dn_num": num_dino_groups
        }
	
		# Outputs:
		input_query_label: shape [B, num_dino_groups * 2 * max_num_gt_bboxes, hidden_dim=256],
			gt class index embedding with noise.
		input_query_bbox : shape [B, num_dino_groups * 2 * max_num_gt_bboxes, 4],
			un_sigmoid gt cxcywh with noise.
		attn_mask: shape [tgt_size, tgt_size], masked=True
			tgt_size = num_dino_groups * 2 * max_num_gt_bboxes + num_queries
		dn_meta = {
            "single_padding": 2 * max_num_gt_bboxes,
            "dn_num": num_dino_groups
        }
```





## ResNet

```
D:\proj\git\detrex\detectron2\detectron2\modeling\backbone\resnet.py

class ResNet(Backbone):
    def __init__(...):
 		# Inputs:
 			stem, BasicStem,
 			stages, make_default_stages,
 			num_classes=None,
 			out_features, [res3, res4, res5],
 			freeze_at, 1
 		#
 		
    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
                
        return outputs
```





## BasicStem

```
class BasicStem(CNNBlockBase):
    def __init__(...):
    	self.conv1 = Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(FrozenBN, 64) = FrozenBatchNorm2d(64)
        )
        
    def forward(self, x):
    	# Conv + FrozenBatchNorm2d + relu + max_pool
    	
        x = self.conv1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

```



## make_default_stages

```
D:\proj\git\detrex\detectron2\detectron2\modeling\backbone\resnet.py

class ResNet(Backbone):
    @staticmethod
	def make_default_stages(depth, block_class=None, **kwargs):
		# Inputs
            depth: 50, 
            norm: FrozenBN, 
            stride_in_1x1: false
    	#
    	
    	in_channels = [64, 256, 512, 1024]
        out_channels = [256, 512, 1024, 2048]
        strides = [1, 2, 2, 2]
        num_blocks_per_stage[50] = [3, 4, 6, 3]
        
        # Return list of 4 stages.
        # Each stage = 
        	ResNet.make_stage(
                    block_class=BottleneckBlock,
                    num_blocks=num_blocks,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=in_channel,
                    out_channels=out_channel,
                    norm=FrozenBN, 
            		stride_in_1x1=false,
            		bottleneck_channels= out_channel // 4
                )
                
    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
		# Inputs
            block_class=BottleneckBlock,
            num_blocks=num_blocks,
            stride_per_block=[s] + [1] * (num_blocks - 1),
            in_channels=in_channel,
            out_channels=out_channel,
            norm=FrozenBN, 
            stride_in_1x1=false,
            bottleneck_channels= out_channel // 4
		#
		
		# Return list of blocks.
		# Each block =
			BottleneckBlock(in_channels=in_channels, out_channels=out_channels, 
				stride=stride_per_block[i], 
				norm=FrozenBN, 
                stride_in_1x1=false,
                bottleneck_channels= out_channels // 4)
                
            in_channels = out_channels

    def forward(self, x):
    	outputs = {}
    	x = self.stem(x)
    	
    	for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
                
        return outputs
```



## BottleneckBlock

```
D:\proj\git\detrex\detectron2\detectron2\modeling\backbone\resnet.py

class BottleneckBlock(CNNBlockBase):

	def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out

```



## ChannelMapper

```
class ChannelMapper(nn.Module):
    def __init__(...):
	    # Inputs
    		input_shapes, = {
                                "res3": ShapeSpec(channels=512),
                                "res4": ShapeSpec(channels=1024),
                                "res5": ShapeSpec(channels=2048)
                            },
    		in_features, = [res3, res4, res5],
    		out_channels, = 256,
    		kernel_size, = 1,
    		stride = 1
    		bias = True,
    		groups = 1,
    		dilation = 1,
    		norm_layer, = nn.GroupNorm(num_groups=32, num_channels=256),
    		activation = None,
    		num_outs, = 4
    	#
```





## Transformer

### Transformer.forward

```
D:\proj\git\AlignDETR\projects\aligndetr\modeling\transformer.py

class Transformer(nn.Module):
    def __init__(...):
		# Inputs:
            encoder, TransformerDecoder
            decoder, TransformerEncoder
            num_feature_levels=4, =4
            two_stage_num_proposals=900, =num_queries=900
            learnt_init_query=True,
        #
        
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
        self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)
        
        
    def forward(
            self,
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            query_embed,
            attn_masks,
            **kwargs,
    	):
    	# Inputs:
        multi_level_feats, # (shape [B, C=256, Hi, Wi], ...)
        multi_level_masks, # [shape [B, Hi, Wi], ...]
        multi_level_pos_embeds, multi_level_position_embeddings, # [shape [B, C=256, Hi, Wi], ...]
        query_embed, query_embeds, 
            # (
                input_query_label: 
                    shape [B, num_dino_groups * 2 * max_num_gt_bboxes, hidden_dim=256],
                    gt class index embedding with noise.
                input_query_bbox: 
                    shape [B, num_dino_groups * 2 * max_num_gt_bboxes, 4],
                    un_sigmoid gt cxcywh with noise.
              )
            #
        attn_masks=[attn_mask, None], 
            # attn_mask: shape [tgt_size, tgt_size], masked=True,
                tgt_size = num_dino_groups * 2 * max_num_gt_bboxes + num_queries
            #
        kwargs, {}
        #
        
        feat_flatten, shape [B, num_anchors, C]
        mask_flatten, shape [B, num_anchors]
        lvl_pos_embed_flatten, shape [B, num_anchors, C]
        spatial_shapes, shape [num_feature_levels, 2], H_i, W_i
        level_start_index, shape [num_feature_levels]
        
        valid_ratios, shape [B, num_feature_levels, 2], w_i_ratio, h_i_ratio
        
        # reference_points, shape [B, num_anchors, num_feature_levels, 2]
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )
        
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs, # {}
        )
        -> TransformerEncoder inst(...)
            # Inputs:
            query=feat_flatten,                      shape [B, num_anchors, C]
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,         shape [B, num_anchors, C]
            query_key_padding_mask=mask_flatten,     shape [B, num_anchors]
            spatial_shapes=spatial_shapes,           shape [num_feature_levels, 2], H_i, W_i
            reference_points=reference_points,       shape [B, num_anchors, num_feature_levels, 2]
            level_start_index=level_start_index,     shape [num_feature_levels]
            valid_ratios=valid_ratios,               shape [B, num_feature_levels, 2], w_i_ratio, h_i_ratio
            # Outputs:
            memory,                                  shape [B, num_anchors, C]
            #

		output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, shape [B, num_anchors, C]
            mask_flatten, shape [B, num_anchors]
            spatial_shapes, shape [num_feature_levels, 2], H_i, W_i
        )
        -> 	output_proposals, shape [B, num_anchors, 4], 
        		[anchor_cx / valid_wi,anchor_cy / valid_hi, anchor_w_i, anchor_h_i].
        	
        	output_proposals_valid, shape [B, num_anchors, 1], 0.01 < point in output_proposals < 0.99

			output_proposals, shape [B, num_anchors, 4], 
        		[anchor_cx / valid_wi,anchor_cy / valid_hi, anchor_w_i, anchor_h_i] + unsigmoid.
        		+ mask invalid point and memory_padding_mask point to 'inf'.
        		
        	output_memory = memory, shape [B, num_anchors, C]
        		mask invalid point and memory_padding_mask point to 0.
        		+ Linear + LayerNorm.
        		
        	return output_memory, output_proposals
        	
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        # shape [B, num_anchors, num_classes]
        
        enc_outputs_coord_unact = (
            self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )
        # shape [B, num_anchors, 4], un_sigmoid cxcywh
        
        topk = self.two_stage_num_proposals, 900
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
		# shape [B, num_queries], topk largest value's index for each image.        
        
        # shape [B, num_queries, 4], un_sigmoid cxcywh
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )
        
        # shape [B, num_queries, 4], sigmoid cxcywh
        reference_points = topk_coords_unact.detach().sigmoid()
        
        # shape [B, num_dino_groups * 2 * max_num_gt_bboxes + topk, 4], sigmoid cxcywh
        reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        
        init_reference_out = reference_points
        
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        # shape [B, num_queries, C]
        	memory + mask invalid point and memory_padding_mask point to 0.
        		+ Linear + LayerNorm + topk.
        #
        
        target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        # shape [B, num_queries, C]
        
        target = torch.cat([query_embed[0], target], 1)
        # shape [B, num_dino_groups * 2 * max_num_gt_bboxes + num_queries, C]
        
        inter_states, inter_references = self.decoder(
            query=target,
        		# shape [B, num_dino_queries + num_queries, C]
            key=memory,
            	# shape [B, num_anchors, C]
            value=memory,
	            # shape [B, num_anchors, C]
            query_pos=None,
            
            key_padding_mask=mask_flatten,
            	shape [B, num_anchors]
            reference_points=reference_points,
            	# shape [B, num_dino_queries + num_queries, 4], 
            	# denoise bbox input + detached encoder bbox output sigmoid cxcywh
            spatial_shapes=spatial_shapes, 
            	# shape [num_feature_levels, 2], H_i, W_i
            level_start_index=level_start_index,
	            # shape [num_feature_levels]
            valid_ratios=valid_ratios,
            	# shape [B, num_feature_levels, 2], w_i_ratio, h_i_ratio
            attn_masks=attn_masks,
            	# shape [tgt_size, tgt_size], masked=True,
                #	tgt_size = num_dino_queries + num_queries
            **kwargs,
            	# {}
        )
        # Outputs:
        	inter_states, shape [num_layers=6, B, num_dino_queries + num_queries, C]
        	inter_references, shape [num_layers=6, B, num_dino_queries + num_queries, 4], sigmoid cxcywh
        #
        
        inter_references_out = inter_references
        
        return (
            inter_states,
            	# shape [num_layers=6, B, num_dino_queries + num_queries, C]
            init_reference_out,
            	# shape [B, num_dino_queries + num_queries, 4], 
            	# denoise bbox input + detached encoder bbox output sigmoid cxcywh
            inter_references_out,
            	# shape [num_layers=6, B, num_dino_queries + num_queries, 4], sigmoid cxcywh
            target_unact,
                # shape [B, num_queries, C]
                    memory + mask invalid point and memory_padding_mask point to 0.
                        + Linear + LayerNorm + topk.
                #
            topk_coords_unact.sigmoid(),
            	# shape [B, num_queries, 4], encoder bbox output sigmoid cxcywh
        )
        
        
        
		
		
```



### Transformer.get_reference_points

```
    def get_reference_points(spatial_shapes, valid_ratios, device):
		# Inputs:
			spatial_shapes, shape [num_feature_levels, 2], H_i, W_i
			valid_ratios, shape [B, num_feature_levels, 2], w_i_ratio, h_i_ratio
		#

        for lvl, (H, W) in enumerate(spatial_shapes):
        	ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
        	# shape: [1, num_anchors_i] / [B, 1] = [B, num_anchors_i]
        	
        	ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
        	# shape: [1, num_anchors_i] / [B, 1] = [B, num_anchors_i]
        	
        	ref = torch.stack((ref_x, ref_y), -1)
        	# shape: [B, num_anchors_i, 2]
        	
		reference_points = torch.cat(reference_points_list, 1)
		# shape: [B, num_anchors, 2]
		
		reference_points = reference_points[:, :, None] * valid_ratios[:, None]
		# shape: [B, num_anchors, 1, 2] * [B, 1, num_feature_levels, 2]
		# = [B, num_anchors, num_feature_levels, 2]

```



## TransformerEncoder

```
D:\proj\git\AlignDETR\projects\aligndetr\modeling\transformer.py

class TransformerEncoder(TransformerLayerSequence):
    def __init__(...):
        # Inputs:
        	embed_dim: int = 256, 256
            num_heads: int = 8, 8
            feedforward_dim: int = 1024, 2048
            attn_dropout: float = 0.1, 0.0
            ffn_dropout: float = 0.1, 0.0
            num_layers: int = 6, 6
            post_norm: bool = False, false
            num_feature_levels: int = 4, num_feature_levels = 4
        #
		->	D:\proj\git\detrex\detrex\layers\transformer.py
			class TransformerLayerSequence(nn.Module):
                self.num_layers = num_layers = 6
                self.layers = nn.ModuleList(), 6 BaseTransformerLayer inst.

			self.post_norm_layer = None


    def forward(...):
		# Inputs:
        query=feat_flatten,                      shape [B, num_anchors, C]
        key=None,
        value=None,
        query_pos=lvl_pos_embed_flatten,         shape [B, num_anchors, C]
        
        key_pos=None,
        attn_masks=None,

        query_key_padding_mask=mask_flatten,     shape [B, num_anchors], used to mask `value`.
        
        key_padding_mask=None,
        
        kwargs:
            spatial_shapes=spatial_shapes,           shape [num_feature_levels, 2], H_i, W_i
            reference_points=reference_points,       shape [B, num_anchors, num_feature_levels, 2]
            level_start_index=level_start_index,     shape [num_feature_levels]
            valid_ratios=valid_ratios,               shape [B, num_feature_levels, 2], w_i_ratio, h_i_ratio
        #

        for layer in self.layers:
            query = layer(
                query, shape [B, num_anchors, C]
                key, None
                value, None
                query_pos=query_pos, shape [B, num_anchors, C]
                attn_masks=attn_masks, None
                query_key_padding_mask=query_key_padding_mask, shape [B, num_anchors]
                key_padding_mask=key_padding_mask, None
                **kwargs,
                    spatial_shapes=spatial_shapes,           shape [num_feature_levels, 2], H_i, W_i
                    reference_points=reference_points,       shape [B, num_anchors, num_feature_levels, 2]
                    level_start_index=level_start_index,     shape [num_feature_levels]
                    valid_ratios=valid_ratios,               shape [B, num_feature_levels, 2], 
                    										 		w_i_ratio, h_i_ratio
            )
            
		return query
```





## TransformerDecoder

```

class TransformerDecoder(TransformerLayerSequence):
    def __init__(...):
        # Inputs:
            embed_dim: int = 256, 256
            num_heads: int = 8, 8
            feedforward_dim: int = 1024, 2048
            attn_dropout: float = 0.1, 0.0
            ffn_dropout: float = 0.1, 0.0
            num_layers: int = 6, 6
            return_intermediate: bool = True, true
            num_feature_levels: int = 4, num_feature_levels = 4
            look_forward_twice=True,
        #
        ->	D:\proj\git\detrex\detrex\layers\transformer.py
			class TransformerLayerSequence(nn.Module):
                self.num_layers = num_layers = 6
                self.layers = nn.ModuleList(), 6 BaseTransformerLayer inst.
                
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice =True
        self.norm = nn.LayerNorm(embed_dim)

    def forward(...):
	    # Inputs:
            query, =target,
        		# shape [B, num_dino_queries + num_queries, C]
            key, =memory,
            	# shape [B, num_anchors, C]
            value, =memory,
	            # shape [B, num_anchors, C]
            query_pos=None, =None,
            
            key_pos=None,
            
            attn_masks=None, =attn_masks,
                # shape [tgt_size, tgt_size], masked=True,
                #	tgt_size = num_dino_queries + num_queries
                
            query_key_padding_mask=None,
            
            key_padding_mask=None, =mask_flatten,
            	# shape [B, num_anchors]
            reference_points=None,=reference_points,
            	# shape [B, num_dino_queries + num_queries, 4], 
            	# denoise bbox input + detached encoder bbox output sigmoid cxcywh
            valid_ratios=None, =valid_ratios,
            	# shape [B, num_feature_levels, 2], w_i_ratio, h_i_ratio
            **kwargs,
            	spatial_shapes=spatial_shapes, 
                    # shape [num_feature_levels, 2], H_i, W_i
                level_start_index=level_start_index,
                    # shape [num_feature_levels]
        #

        for layer_idx, layer in enumerate(self.layers):
        	reference_points_input, shape [B, num_dino_queries + num_queries, num_feature_levels, 4],
        		sigmoid cxcywh
        
        	# reference_points_input[:, :, 0, :], shape [B, num_dino_queries + num_queries, 4]
        	# query_sine_embed, shape [B, num_dino_queries + num_queries, 4 * 128 = 2 * C],
        	# 					cycxwh
        	query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
        	
        	# query_pos, shape [B, num_dino_queries + num_queries, C]
            query_pos = self.ref_point_head(query_sine_embed)
			
			output = layer(
                output, # shape [B, num_dino_queries + num_queries, C]
                key,   # shape [B, num_anchors, C]
                value, # shape [B, num_anchors, C]
                query_pos=query_pos, # shape [B, num_dino_queries + num_queries, C]
                
                key_pos=key_pos, # None
                
                query_sine_embed=query_sine_embed, # shape [B, num_dino_queries + num_queries, 2 * C],
                								   # cycxwh
                attn_masks=attn_masks, # shape [tgt_size, tgt_size], masked=True,
                					   # tgt_size = num_dino_queries + num_queries
                query_key_padding_mask=query_key_padding_mask, # None
                
                key_padding_mask=key_padding_mask, # shape [B, num_anchors]
                
                reference_points=reference_points_input, 
                			# shape [B, num_dino_queries + num_queries, num_feature_levels, 4],
        					# sigmoid cxcywh
                **kwargs,
                    spatial_shapes=spatial_shapes, 
                        # shape [num_feature_levels, 2], H_i, W_i
                    level_start_index=level_start_index,
                        # shape [num_feature_levels]
            )
            # Outputs:
            	shape [B, num_query=num_dino_queries + num_queries, C]
            #
            
            # shape [B, num_query=num_dino_queries + num_queries, 4]
            tmp = self.bbox_embed[layer_idx](output)
            new_reference_points = tmp + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
            
            reference_points = new_reference_points.detach()
            
            intermediate.append(self.norm(output))
            
            intermediate_reference_points.append(new_reference_points)
            
        return
        	# shape [num_layers, B, num_query=num_dino_queries + num_queries, C]
        	torch.stack(intermediate), 
        
        	# shape [num_layers, B, num_query=num_dino_queries + num_queries, 4], sigmoid cxcywh
        	torch.stack(intermediate_reference_points)
```





## BaseTransformerLayer

```
D:\proj\git\detrex\detrex\layers\transformer.py

class BaseTransformerLayer(nn.Module):
    def __init__(...):
    	self.operation_order=
    		 # Encoder: ("self_attn", "norm", "ffn", "norm")
    		 # Decoder: ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
		
		self.pre_norm = False
		
		self.attentions = nn.ModuleList() 	
			# Ecoder: 1 MultiScaleDeformableAttention inst.
			# Decoder: 1 MultiheadAttention inst, 1 MultiScaleDeformableAttention inst.
		
		self.embed_dim = self.attentions[0].embed_dim
		
		self.ffns = nn.ModuleList() # Contains: 1 FFN inst.
		
		self.norms = nn.ModuleList() 
			# Encoder: 2 nn.LayerNorm(embed_dim) inst.
			# Decoder: 3 nn.LayerNorm(embed_dim) inst.
			
	# Encoder:
    def forward(...):
    	# Inputs:
        query, shape [B, num_anchors, C]
        key, None
        value, None
        query_pos, shape [B, num_anchors, C]
        key_pos: = None,
        attn_masks, None
        query_key_padding_mask, shape [B, num_anchors]
        key_padding_mask, None
        **kwargs,
            spatial_shapes=spatial_shapes,           shape [num_feature_levels, 2], H_i, W_i
            reference_points=reference_points,       shape [B, num_anchors, num_feature_levels, 2]
            level_start_index=level_start_index,     shape [num_feature_levels]
            valid_ratios=valid_ratios,               shape [B, num_feature_levels, 2], w_i_ratio, h_i_ratio
		#
		
        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query, shape [B, num_anchors, C]
                    query,
                    query,
                    None,
                    query_pos=query_pos, shape [B, num_anchors, C]
                    key_pos=query_pos,
                    attn_mask=None,
                    key_padding_mask=query_key_padding_mask, shape [B, num_anchors]
                    **kwargs,
                )
                attn_index += 1
        		...


	# Decoder:
    def forward(...):
    	# Inputs:
    	query,                      # shape [B, num_dino_queries + num_queries, C]
        key,                        # shape [B, num_anchors, C]
        value,                      # shape [B, num_anchors, C]
        query_pos,                  # shape [B, num_dino_queries + num_queries, C]
        
        key_pos: = None, None
        attn_masks,                 # shape [tgt_size, tgt_size], masked=True,
        			                  	tgt_size = num_dino_queries + num_queries
        query_key_padding_mask, None
        
        key_padding_mask,             shape [B, num_anchors]
        **kwargs,
        	query_sine_embed=query_sine_embed, 
        	                          # shape [B, num_dino_queries + num_queries, 2 * C],
                					  # cycxwh
            reference_points=reference_points_input, 
                			          # shape [B, num_dino_queries + num_queries, num_feature_levels, 4],
        					          # sigmoid cxcywh
        	spatial_shapes=spatial_shapes, 
                                      # shape [num_feature_levels, 2], H_i, W_i
            level_start_index=level_start_index,
                                      # shape [num_feature_levels]
		#
    	
        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query, # shape [B, num_dino_queries + num_queries, C]
                    query,
                    query,
                    None,
                    query_pos=query_pos, # shape [B, num_dino_queries + num_queries, C]
                    key_pos=query_pos,
                    attn_mask=attn_masks, # shape [tgt_size, tgt_size], masked=True,
        			                  		tgt_size = num_dino_queries + num_queries
                    key_padding_mask=None,
                    **kwargs,
                )
                ...
                
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query, # shape [B, num_dino_queries + num_queries, C]
                    key,   # shape [B, num_anchors, C]
                    value, # shape [B, num_anchors, C]
                    None,
                    query_pos=query_pos, # shape [B, num_dino_queries + num_queries, C]
                    key_pos=key_pos, None
                    attn_mask=attn_masks,# shape [tgt_size, tgt_size], masked=True,
        			                  		tgt_size = num_dino_queries + num_queries
                    key_padding_mask=key_padding_mask, # shape [B, num_anchors]
                    **kwargs,
                )
                ...
```



## MultiheadAttention

```
D:\proj\git\detrex\detrex\layers\attention.py

class MultiheadAttention(nn.Module):
    def __init__(...):
	    # Inputs:
            embed_dim: int, 256
            num_heads: int, 8
            attn_drop: float = 0.0, 0.0
            proj_drop: float = 0.0,
            batch_first: bool = False, True
        #
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=batch_first
        )

        self.proj_drop = nn.Dropout(proj_drop)


    def forward(...):
    	# Inputs:
    		query, query, # shape [B, num_dino_queries + num_queries, C]
    		key, query,
    		value, query,
    		identity, None,
    		query_pos, query_pos, # shape [B, num_dino_queries + num_queries, C]
    		key_pos, query_pos,
    		attn_mask, attn_masks, # shape [tgt_size, tgt_size], masked=True,
        			               		tgt_size = num_dino_queries + num_queries
    		key_padding_mask, None,
    		**kwargs
    	#
    	
    	out = self.attn(
            query=query + query_pos,
            key=key + key_pos,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        
        return query + self.proj_drop(out)
```





## MultiScaleDeformableAttention

```
D:\proj\git\detrex\detrex\layers\multi_scale_deform_attn.py

class MultiScaleDeformableAttention(nn.Module):
    def __init__(...):
	    # Inputs:
            embed_dim: int = 256, 256
            num_heads: int = 8, 8
            num_levels: int = 4, 4
            num_points: int = 4,
            img2col_step: int = 64,
            dropout: float = 0.1, 0.0
            batch_first: bool = False, True
    	#
    	
    	self.im2col_step = img2col_step = 64
    	
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)


	# Encoder:
    def forward(...):
    	# Inputs:
    	query, query, shape [B, num_query=num_anchors, C]
        key,   query, shape [B, num_value=num_anchors, C]
        value, query, shape [B, num_value=num_anchors, C]
        identity, None,
        query_pos=query_pos, shape [B, num_query=num_anchors, C]
        
        key_padding_mask=query_key_padding_mask, shape [B, num_value=num_anchors]
        
        spatial_shapes=spatial_shapes,           shape [num_feature_levels, 2], H_i, W_i
        reference_points=reference_points,       shape [B, num_query=num_anchors, num_feature_levels, 2]
        level_start_index=level_start_index,     shape [num_feature_levels]
        
        **kwargs,
        	key_pos=query_pos,
        	attn_mask=None,
        	
            valid_ratios=valid_ratios,           shape [B, num_feature_levels, 2], w_i_ratio, h_i_ratio
        #
        identity = query
        
        query = query + query_pos
        value = self.value_proj(value)
        
        value = value.masked_fill(key_padding_mask[..., None], float(0))
        
        # shape [B, num_value=num_anchors, num_heads, head_dim]
        value = value.view(bs, num_value, self.num_heads, -1)
        
        # shape [B, num_query=num_anchors, num_heads, num_levels, num_points, 2]
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        
        # shape [B, num_query=num_anchors, num_heads, num_levels * num_points]
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        
        # shape [B, num_query=num_anchors, num_heads, num_levels, num_points]
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )
        
        if reference_points.shape[-1] == 2:
            # shape [num_feature_levels, 2], Wi, Hi
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            # shape [B, num_query=num_anchors, num_heads, num_levels, num_point, 2]
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                    # shape [B, num_query=num_anchors, 1, num_feature_levels, 1, 2]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                    # shape [B, num_query=num_anchors, num_heads, num_levels, num_point, 2]
                    # / [1, 1, 1, num_feature_levels, 1, 2], W_i, H_i
            )
        
        elif reference_points.shape[-1] == 4:
        	# shape [B, num_query=num_anchors, num_heads, num_levels, num_point, 2]
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                    # shape [B, num_query=num_anchors, 1, num_feature_levels, 1, 2]
                + sampling_offsets
                    # shape [B, num_query=num_anchors, num_heads, num_levels, num_point, 2]
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                    # shape [B, num_query=num_anchors, 1, num_feature_levels, 1, 2]
                * 0.5
            )
            
        output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
        ->
            # Inputs:
                value, shape [B, num_value=num_anchors, num_heads, head_dim]
                spatial_shapes, shape [num_feature_levels, 2], Hi, Wi
                sampling_locations, shape [B, num_query=num_anchors, num_heads, num_levels, num_point, 2]
                attention_weights,  shape [B, num_query=num_anchors, num_heads, num_levels, num_points]
            #
            
            # shape [B, num_query=num_anchors, num_heads, num_levels, num_point, 2]
            sampling_grids = 2 * sampling_locations - 1

            for level, (H_, W_) in enumerate(value_spatial_shapes):
            	# value_list[level], shape [B, num_value_i=num_anchors_i, num_heads, head_dim]
            	# 	.flatten(2), shape [B, num_value_i=num_anchors_i, num_heads * head_dim]
            	# 	.transpose(1, 2), shape [B, num_heads * head_dim, num_value_i=num_anchors_i]
            	# 	.reshape, shape [B * num_heads, head_dim, value_Hi=Hi, value_Wi=Wi]
            	value_l_ = (
                    value_list[level].flatten(2).transpose(1, 2)
                    	.reshape(bs * num_heads, embed_dims, H_, W_)
                )
                
                # sampling_grids[:, :, :, level], shape [B, num_query=num_anchors, num_heads, num_point, 2]
                # 	.transpose(1, 2), shape [B, num_heads, num_query=num_anchors, num_point, 2]
                # 	.flatten(0, 1), shape [B * num_heads, num_query=num_anchors, num_point, 2]
                sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
                
                # For each point in sampling_grid_l_, get its value from value_l_,
                # 	point count = num_query * num_point
                # sampling_value_l_, shape [B * num_heads, head_dim, num_query=num_anchors, num_point]
                sampling_value_l_ = F.grid_sample(
                    value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
                )
                
            # attention_weights, shape [B, num_query=num_anchors, num_heads, num_levels, num_points]
            # 	.transpose(1, 2), shape [B, num_heads, num_query=num_anchors, num_levels, num_points]
            # 	.reshape, shape [B * num_heads, 1, num_query=num_anchors, num_levels * num_points]
            attention_weights = attention_weights.transpose(1, 2).reshape(
                bs * num_heads, 1, num_queries, num_levels * num_points
            )
            
            # torch.stack(sampling_value_list, dim=-2), 
            		shape [B * num_heads, head_dim, num_query=num_anchors, num_levels, num_point]
            # 	.flatten(-2), shape [B * num_heads, head_dim, num_query=num_anchors, num_levels * num_point]
            # 	* attention_weights, 
            # 				  shape [B * num_heads, head_dim, num_query=num_anchors, num_levels * num_point]
            # 	.sum(-1), shape [B * num_heads, head_dim, num_query=num_anchors]
            # 	.view, shape [B, num_heads * head_dim, num_query=num_anchors]
            output = (
                (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
                .sum(-1)
                .view(bs, num_heads * embed_dims, num_queries)
            )
            # shape [B, num_query=num_anchors, C]
            return output.transpose(1, 2).contiguous()
        
        output = self.output_proj(output)
        
        return self.dropout(output) + identity
        
        

	# Decoder:
    def forward(...):
    	# Inputs:
    	query, query, shape [B, num_query=num_dino_queries + num_queries, C]
        key, key,     shape [B, num_value=num_anchors, C]
        value, value, shape [B, num_value=num_anchors, C]
        identity, None,
        query_pos=query_pos, shape [B, num_query=num_dino_queries + num_queries, C]
        
        key_padding_mask=query_key_padding_mask, shape [B, num_value=num_anchors]
        
        spatial_shapes=spatial_shapes,           shape [num_feature_levels, 2], H_i, W_i
        reference_points=reference_points, 
        						shape [B, num_query=num_dino_queries + num_queries, num_feature_levels, 4],
        					          	sigmoid cxcywh
        level_start_index=level_start_index,     shape [num_feature_levels]
        
        **kwargs,
        	key_pos=key_pos, None
            attn_mask=attn_masks,# shape [tgt_size, tgt_size], masked=True,
                                    tgt_size = num_dino_queries + num_queries
        	
            query_sine_embed=query_sine_embed, 
        	                          # shape [B, num_dino_queries + num_queries, 2 * C],
                					  # cycxwh
        #
        # Outputs:
        	shape [B, num_query=num_dino_queries + num_queries, C]
        #
```



## FFN

```
D:\proj\git\detrex\detrex\layers\mlp.py

class FFN(nn.Module):
    def __init__(...):
    	# Inputs:
            embed_dim=256, 256
            feedforward_dim=1024, 2048
            output_dim=None, 256
            num_fcs=2, 2
            activation=nn.ReLU(inplace=True),
            ffn_drop=0.0, 0.0
            fc_bias=True,
            add_identity=True,
        #
        
        self.layers = nn.Sequential(*layers)
        # layers:
            nn.Linear, 256 -> 2048, bias=True
            activation: nn.ReLU(inplace=True),
            nn.Dropout(0.0)
            nn.Linear, 2048 -> 256, bias=True
            nn.Dropout(0.0)
        #
        
    def forward(self, x, identity=None):
    	# Inputs:
    		x, shape [B, num_anchors, C]
    		identity=None
    	#
    	out = self.layers(x)
    	
    	return x + out
```



## MLP

```
D:\proj\git\detrex\detrex\layers\mlp.py

class MLP(nn.Module):
    def forward(self, x):
		# nn.Linear 2 * embed_dim -> embed_dim,
		# F.relu
		# nn.Linear embed_dim -> embed_dim
```





# Data

## build_detection_train_loader

```
D:\proj\git\detrex\detectron2\detectron2\data\build.py

def build_detection_train_loader(...):
	# Inputs
	dataset,
	mapper,
	sampler=None,
    total_batch_size, # =16
    aspect_ratio_grouping=True,
    num_workers=0, # =16
    collate_fn=None,
	#
	
	# Serialize each dataset.
	dataset = DatasetFromList(dataset, copy=False)
	
	dataset = MapDataset(dataset, mapper)
	# Inputs: 
		size=len(dataset)=1,
		shuffle=True,
		seed=None
	#
	sampler = TrainingSampler(len(dataset))
	
	return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
```



## build_batch_data_loader

```

def build_batch_data_loader(...):
	# Inputs:
        dataset, # DatasetFromList
        sampler, # TrainingSampler
        total_batch_size,
        aspect_ratio_grouping=False, # =True
        num_workers=0, # =16
        collate_fn=None, # =None
	#
	
	batch_size = total_batch_size // world_size
	
	dataset = ToIterableDataset(dataset, sampler)
	
	data_loader = torchdata.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
        worker_init_fn=worker_init_reset_seed,
    )  # yield individual mapped dict
    
    # Batch data that have similar aspect ratio (w / h > 1 vs else) together.
    # class AspectRatioGroupedDataset(data.IterableDataset)
    data_loader = AspectRatioGroupedDataset(data_loader, batch_size)

    return data_loader
```







## register_all_coco

```
D:\proj\git\detrex\detectron2\detectron2\data\datasets\builtin.py

def register_all_coco(root):
	-> 
		# dataset_name, key, json_file, image_root from _PREDEFINED_SPLITS_COCO
		# dataset_name='coco'
		# key='coco_2017_train'
		# root='datasets'
		# json_file='coco/annotations/instances_train2017.json'
		# image_root='coco/train2017'
		register_coco_instances(
            key,
            _get_builtin_metadata(dataset_name),
	            -> _get_coco_instances_meta()
	            -> {
	            		thing_dataset_id_to_contiguous_id: {class_id: class_index},
	            		thing_classes: [class_name, ...], # Same order as sorted class_index
	            		thing_colors: [[R, G, B], ...]    # Same order as sorted class_index
	               }
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
        -> 	# name='coco_2017_train'
        	# json_file='datasets/coco/annotations/instances_train2017.json'
        	# image_root='datasets/coco/train2017'
        	DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
            MetadataCatalog.get(name).set(
                json_file=json_file, 
                image_root=image_root, 
                evaluator_type="coco", 
                **metadata)
	...

```



## get_detection_dataset_dicts

```
D:\proj\git\detrex\detectron2\detectron2\data\build.py

def get_detection_dataset_dicts(...):
	# Inputs
	names='coco_2017_train'
	filter_empty=True,
    min_keypoints=0,
    proposal_files=None,
    check_consistency=True
    #
    
    names = [names]
    
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
	-> 	D:\proj\git\detrex\detectron2\detectron2\data\datasets\coco.py
		# dataset_name='coco_2017_train'
        # json_file='datasets/coco/annotations/instances_train2017.json'
        # image_root='datasets/coco/train2017'
		load_coco_json(json_file, image_root, dataset_name)
		-> 	meta = MetadataCatalog.get(dataset_name)
			cat_ids = sorted(coco_api.getCatIds())
			cats = coco_api.loadCats(cat_ids)
			thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        	meta.thing_classes = thing_classes
        	
        	if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
                id_map = {v: i for i, v in enumerate(cat_ids)}
                meta.thing_dataset_id_to_contiguous_id = id_map
                
            # Returns:
            {
            	file_name: 'datasets/coco/train2017' / file_name, str
            	height: ,
            	width: ,
            	image_id: ,
            	# Filter out annotations with num_segmentation_values not even or < 6
            	annotations: [
                             	{
                             		# if key exists in annotation:
                                        iscrowd: , 
                                        bbox: , 
                                        keypoints: , 
                                        category_id: , 
                                        segmentation: , 
                                        keypoints: ,
                             		
                             		bbox_mode: BoxMode.XYWH_ABS,
                             		
                             		# if id_map:
                             			category_id: id_map[obj['category_id']] 
                             	}, 
                             	...
                             ]
            }
            
	# Filter out images with no annotations or only crowd annotations
	dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
	
	class_names = MetadataCatalog.get(names[0]).thing_classes
	
	# All datasets in `names` should have the same metadata 'thing_classes'.
    check_metadata_consistency("thing_classes", names)
    print_instances_class_histogram(dataset_dicts, class_names)
    return dataset_dicts
```







## DetrDatasetMapper

### DetrDatasetMapper.init

```
D:\proj\git\detrex\detrex\data\detr_dataset_mapper.py

class DetrDatasetMapper:
    def __init__(...):
    	# Inputs
    	augmentation,
    	augmentation_with_crop,
    	is_train=True,
    	mask_on=False,
    	img_format='RGB'
    	#
    	
```



### DetrDatasetMapper.call

```
D:\proj\git\detrex\detrex\data\detr_dataset_mapper.py

	def __call__(self, dataset_dict):
		dataset_dict = copy.deepcopy(dataset_dict)
		
    	image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
    	-> 	with PathManager.open(file_name, "rb") as f:
                image = PIL.Image.open(f)
                image = _apply_exif_orientation(image)
                
                return convert_PIL_to_numpy(image, format)
                ->  image = image.convert('RGB')
                	image = np.asarray(image)
                	
    	# If width, height != image size, then raise error.
    	utils.check_image_size(dataset_dict, image)
    	
    	# 50% percent:
    	image, transforms = T.apply_transform_gens(self.augmentation, image)
    	# The other 50% percent:
    	image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)
    	-> 	D:\proj\git\detrex\detectron2\detectron2\data\transforms\augmentation.py
    		T.apply_augmentations(self.augmentation, image)
    	-> 	inputs = AugInput(image)
	    	
	    	tfms = inputs.apply_augmentations(augmentations)
	    	-> return AugmentationList(augmentations)(inputs)
	    	-> 	aug_list = AugmentationList(augmentations)
	    		-> aug_list.augs = [_transform_to_aug(x) for x in augmentations]
	    		-> aug_list.augs = [x for x in augmentations]
	    		
	    		aug_list(inputs)
	    		-> 	tfms = []
                    for x in aug_list.augs:
                    	# x: Base=Augmentation instance.
                    	# inputs: AugInput instance.
                        tfm = x(inputs)
                        -> Augmentation.__call__(inputs):
                        	args = _get_aug_input_args(x, inputs)
                        	-> 	x.input_args = ('image',)
                        		args = [inputs.image]
                        	
                            tfm = x.get_transform(*args)
                            
                            assert isinstance(tfm, (Transform, TransformList)), (
                                f"{type(self)}.get_transform must return an instance of Transform! "
                                f"Got {type(tfm)} instead."
                            )
                            
                            inputs.transform(tfm)
                            -> inputs.image = tfm.apply_image(inputs.image)
                            
                            return tfm
                        
                        tfms.append(tfm)
                        
                    return TransformList(tfms)
                    	-> fvcore.transforms.transform.TransformList(tfms)
	    	
    		return inputs.image, tfms
    	
    	image_shape = image.shape[:2] # After transform.
    	dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    	
    	for anno in dataset_dict["annotations"]:
    		anno.pop("segmentation", None)
            anno.pop("keypoints", None)
            
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        -> 	For each not crowded annotation: # Filter out crowded.
        		# XYWH_ABS -> XYXY_ABS
        		bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        		
                bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
                -> fvcore.transforms.transform.TransformList(tfms).apply_box(
                											np.array([bbox]))[0].clip(min=0)	
                											
                -> TransformList.__getattribute__('apply_box')(np.array([bbox]))[0].clip(min=0)
                -> TransformList._apply(np.array([bbox]), 'apply_box')[0].clip(min=0)
                	-> _apply:
                		# Inputs:  bbox, shape: [1, 4], XYXY_ABS
                		# Outputs: bbox, shape: [1, 4], XYXY_ABS
                		#
                		for transform in tfms:
            				bbox = getattr(transform, 'apply_box')(bbox)
            				
            			return bbox
                		
                	Clip transformed bbox >= 0,bbox shape: [4], XYXY_ABS
                	
                # Clip transformed bbox <= image size,bbox shape: [4], XYXY_ABS
                annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
                annotation["bbox_mode"] = BoxMode.XYXY_ABS
        
        instances = utils.annotations_to_instances(annos, image_shape)
        -> 	target = Instances(image_size)
        	target.gt_boxes = Boxes(boxes) # boxes, ndarray, shape: [num_gt_bboxes, 4], XYXY_ABS
        	-> Boxes inst.tensor, tensor, shape: [num_gt_bboxes, 4], dtype=torch.float32, XYXY_ABS
        	
        	target.gt_classes = classes  # classes,  tensor, shape: [num_gt_bboxes], dtype=torch.int64
        	return target
        	
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        -> # D:\proj\git\detrex\detectron2\detectron2\data\detection_utils.py
        	
        	# Only keep gt_bbox with w and h > box_threshold=1e-5
        	m= instances.gt_boxes.nonempty(threshold=box_threshold=1e-5)
        	return instances[m]
```



## Augmentations

### RandomFlip

```
D:\proj\git\detrex\detectron2\detectron2\data\transforms\augmentation_impl.py

class RandomFlip(Augmentation):
    def get_transform(self, image):
	    h, w = image.shape[:2]
    	# 50% return HFlipTransform(w)
    	-> fvcore.transforms.transform.HFlipTransform
    	
    	# Else: return NoOpTransform()
    	-> fvcore.transforms.transform.NoOpTransform
```



### ResizeShortestEdge



```
D:\proj\git\detrex\detectron2\detectron2\data\transforms\augmentation_impl.py

class ResizeShortestEdge(Augmentation):
    def get_transform(self, image):
    	# If sample_style == 'range':
    	# 	1. Resize short edge to short_edge_length. Keep scale.
    	# 	2. If new long edge > max_size:
                Resize new long edge to max_size. Keep scale.
        
        # Else:
        # 	1. short_edge_length = Randomly choice one from list.
    	# 	2. Resize short edge to short_edge_length. Keep scale.
    	# 	3. If new long edge > max_size: # max_size may = sys.maxsize
                Resize new long edge to max_size. Keep scale.
        
        h, w = image.shape[:2]
        return ResizeTransform(h, w, newh, neww, self.interp)
        # D:\proj\git\detrex\detectron2\detectron2\data\transforms\transform.py
```



### RandomCrop

```
D:\proj\git\detrex\detectron2\detectron2\data\transforms\augmentation_impl.py

class RandomCrop(Augmentation):
    def get_transform(self, image):
        cropped_w <= image_w and in range [crop_size[0], crop_size[1]] if available
        cropped_h <= image_h and in range [crop_size[0], crop_size[1]] if available
		
		# Randomly select w0, h0
        return CropTransform(w0, h0, cropw, croph)
        -> fvcore.transforms.transform.CropTransform
```



## Transforms

### ResizeTransform

```
D:\proj\git\detrex\detectron2\detectron2\data\transforms\transform.py

class ResizeTransform(Transform):
    def apply_image(self, img, interp=None):
    	# PIL.Image.resize(PIL.Image.fromarray(img))
    	# Return np.asarray
    	
    def apply_coords(self, coords):
    	# resize x, y with scale=new_size / size.
```



### HFlipTransform

```
D:\proj\git\fvcore\fvcore\transforms\transform.py

class HFlipTransform(Transform):
    def apply_image(self, img):
		# return np.flip

    def apply_coords(self, coords):
	    coords[:, 0] = self.width - coords[:, 0]
```

### CropTransform

```
D:\proj\git\fvcore\fvcore\transforms\transform.py

class CropTransform(Transform):
    def apply_image(self, img):
    	# return img[...]

    def apply_coords(self, coords):
    	coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
```



### Transform

```
class Transform(metaclass=ABCMeta):
    def apply_box(self, box):
        # Inputs:  bbox, shape: [1, 4], XYXY_ABS
        # Outputs: bbox, shape: [1, 4], XYXY_ABS
        #


        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        -> 	self.apply_coords(coords)
                #  Inputs: coords, shape: [4, 2], 
                        [left_top, right_top, left_bottom, right_bottom]
                # Outputs: coords, shape: [4, 2]
                #
                
        # coords, shape: [1, 4, 2]
        ...
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        # shape: [1, 4]
```





# Loss

### init

| AlignDETRCriterion | TwoStageCriterion    | ManyToOneCriterion         | BaseCriterion             | Value |
| ------------------ | -------------------- | -------------------------- | ------------------------- | ----- |
|                    |                      | num_classes                | num_classes               |       |
|                    |                      | matcher                    | matcher                   |       |
|                    |                      | pos_norm_type`='softmax'`  | pos_norm_type`='softmax'` |       |
|                    |                      | weight_dict                | weight_dict               |       |
|                    |                      | match_number               |                           |       |
|                    |                      | gamma                      |                           |       |
|                    |                      | alpha                      |                           |       |
|                    |                      | weight_table`=Tensor[7,2]` |                           |       |
|                    | two_stage_binary_cls |                            |                           | false |



### AlignDETRCriterion

#### AlignDETRCriterion.forward

```
    def forward(self, outputs, targets, dn_metas=None):
		# Inputs:
			outputs, output = 
					 {
                        pred_logits: ,
					 	pred_boxes: ,
					 	aux_outputs: [
                                         {
                                            pred_logits: , 
                                            pred_boxes: 
                                         },
                                         ...
					 				 ]
					 	enc_outputs: {
					 				 	pred_logits: , 
					 				 	pred_boxes: 
					 				 }
					 },
			targets = 
					 [
					 	{
                        	labels,
                        	boxes cxcywh,
                        },
                        ...
					 ]
			dn_metas, dn_meta = 
					 	{
                            single_padding: max_num_gt * 2,
                            dn_num: num_denoise_group,
                            output_known_lbs_bboxes: {
												 	pred_logits: ,
                                                    pred_boxes: ,
                                                    aux_outputs: [
                                                                     {
                                                                        pred_logits: , 
                                                                        pred_boxes: 
                                                                     },
                                                                     ...
                                                                 ]
												 }
					 	}
        #
        D:\proj\git\AlignDETR\aligndetr\criterions\base_criterion.py
        
        BaseCriterion.forward:
        	losses = {}
        	num_boxes= num_all_gt_bboxes
            num_layers=6
            l_dict, indices = self.get_loss(outputs_without_aux, targets,  num_boxes, num_layers)
            losses.update(l_dict)
        
            for i in range(num_layers - 1):
                aux_outputs = outputs['aux_outputs'][i]
                
                # 1 <= i+1 <= 5
                l_dict, indices = self.get_loss(aux_outputs, targets, num_boxes , i+1)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                indices_list.append(indices)
                losses.update(l_dict)

		TwoStageCriterion.forward:
			enc_outputs = outputs["enc_outputs"]
            l_dict, indices = self.get_loss( enc_outputs, targets, num_boxes, 0)
            l_dict = {k + "_enc": v for k, v in l_dict.items()}
            losses.update(l_dict)


        AlignDETRCriterion.compute_dn_loss:
        	aux_num=5
        	num_boxes= num_all_gt_bboxes
        	dn_num=num_dino_groups
        	
            l_dict.update(
                self.get_loss(
                    output_known_lbs_bboxes, 
                    targets, 
                    num_boxes * dn_num, 
                    specify_indices=dn_idx, 
                    layer_spec=aux_num + 1
                )[0]
            )

            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)

            for i in range(aux_num):
                l_dict = {}
                    l_dict.update(
                        self.get_loss(
                            output_known_lbs_bboxes_aux,
                            targets,
                            num_boxes * dn_num,
                            specify_indices=dn_idx, 
                            layer_spec=i+1
                        )[0]
                    )
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}

                losses.update(l_dict)
                
class ManyToOneCriterion(BaseCriterion):        
    def get_loss(...):
    	target_boxes, assigned_gt_bbox, normed gt cxcywh, shape [num_all_assigned, 4]
    	target_classes_o, shape [num_all_assigned], assigned_gt_class_index
    	
    	pos_idx, (	batch_index, shape [num_all_assigned], 
    				assigned_pred_index, shape [num_all_assigned]
    			 )
    	
    	pos_idx_c, (	batch_index, shape [num_all_assigned], 
    					assigned_pred_index, shape [num_all_assigned]
    					assigned_gt_class_index, shape [num_all_assigned]
    			   )
    	
    	src_boxes, assigned_pred_bbox, sigmoid cxcywh, shape [num_all_assigned, 4]
    	
    	
        loss_class,loc_weight = IA_BCE_loss(src_logits, pos_idx_c, src_boxes, 
                                target_boxes, indices, num_boxes, 
                                self.alpha, self.gamma, 
                                w_prime,)
		# Inputs:
			src_logits, pred_logits, shape [B, num_queries, num_classes]
			pos_idx_c, 	  (	batch_index, shape [num_all_assigned], 
                       	  	assigned_pred_index, shape [num_all_assigned]
                          	assigned_gt_class_index, shape [num_all_assigned]
                      	  )
            src_boxes,    assigned_pred_bbox, sigmoid cxcywh, shape [num_all_assigned, 4]
            target_boxes, assigned_gt_bbox, normed gt cxcywh, shape [num_all_assigned, 4]
            indices,      [ 
            				[
            					src_ind, shape [num_assigned_image_i]
            					tgt_ind, shape [num_assigned_image_i]
							],
							...
                          ]
            avg_factor=num_boxes,    num_all_gt * num_dino_groups if dino else num_all_gt
            self.alpha, 0.25
            self.gamma, 2.0
            w_prime, [1, e^(-1/1.5) or 0] if not specify_indices, else 1
        #
        -> 	t = prob[pos_idx_c]**alpha * iou_scores ** (1-alpha)
        	# prob[pos_idx_c], pred class score for gt, shape [num_all_assigned]
        	# iou_scores, pred and gt bbox iou score, shape [num_all_assigned]
        	# t, shape [num_all_assigned]
        	
        	t = torch.clamp(t, 0.01).detach()
        	
        	# rank, shape [num_all_assigned]. For each gt, rank order of match score in descending order.
        	rank = get_local_rank(t, indices)
        	
        	rank_weight, shape [num_all_assigned].
        
```



# Matcher

## MixedMatcher



```
C = focal_loss + L1 + GIOU

Focal loss = positive loss - negative loss
positive loss = -       alpha * (1-p)^gamma * ln(p)
negative loss = - (1 - alpha) *     p^gamma * ln(1 - p)


indices, [ 
            [
                src_ind, shape [num_assigned_image_i]
                tgt_ind, shape [num_assigned_image_i]
            ],
            ...
         ]
```

