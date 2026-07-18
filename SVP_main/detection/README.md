# Downstream tasks training scripts

This folder contains reference training scripts for Faster/Mask/Keypoint-RCNN-ResNet50-FPN for object detection, segmentation and keypoint detection.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Weights backbone is adopted from the classification task, therefore it should be the compressed resnet50 with coressponding pruning ratio.

As [recommended](https://github.com/pytorch/vision/blob/87d54c4e583207e7b003d6b59f1e7f49167f68f1/references/detection/train.py#L85) by `torchvision`, default learning rate and batch size values go along with 8xV100. Please modify them to match with your numbers of gpus, *e.g.,* `--nproc_per_node=1 --lr 0.02 -b 2`.

### Faster R-CNN ResNet-50 FPN
```
torchrun --nproc_per_node=8 train.py --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone resnet50.pt -cpr [0.55]*20
```

### Mask R-CNN
```
torchrun --nproc_per_node=8 train.py --dataset coco --model maskrcnn_resnet50_fpn --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone resnet50.pt -cpr [0.55]*20
```

### Keypoint R-CNN
```
torchrun --nproc_per_node=8 train.py --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3 --weights-backbone resnet50.pt -cpr [0.55]*20
```


# Visualizing model inference
Compressed models can be deployed as [torchvision's guide](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection).

For your convenience, we have also prepared an example script, [visualize.py](./visualize.py), that emphasizes the enhanced FPS achieved by the pruned model. Below is the example usage:

```
python visualize.py --input birthday.mp4 --custom -cpr [0.55]*20 --weight model_24.pth
```

By using this script, you can effortlessly visualize and compare the inference speed of both the baseline and pruned models. This provides a clear demonstration of the substantial throughput acceleration achieved by our framework.




# Verification
Baseline models can be found [here](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection).

To verify their performances:

```bash
git clone https://github.com/pytorch/vision.git
cd vision/references/detection
python train.py --dataset coco --model fasterrcnn_resnet50_fpn --weights FasterRCNN_ResNet50_FPN_Weights.COCO_V1 --test-only --data-path ~/data/coco/
python train.py --dataset coco --model maskrcnn_resnet50_fpn --weights MaskRCNN_ResNet50_FPN_Weights.COCO_V1 --test-only --data-path ~/data/coco/
python train.py --dataset coco_kp --model keypointrcnn_resnet50_fpn --weights KeypointRCNN_ResNet50_FPN_Weights.COCO_V1 --test-only --data-path ~/data/coco/
```

<details>

```bash
Not using distributed mode
Namespace(data_path='/home/van-tien.pham/data/coco/', dataset='coco', model='fasterrcnn_resnet50_fpn', device='cuda', batch_size=2, epochs=26, workers=4, opt='sgd', lr=0.02, momentum=0.9, weight_decay=0.0001, norm_weight_decay=None, lr_scheduler='multisteplr', lr_step_size=8, lr_steps=[16, 22], lr_gamma=0.1, print_freq=20, output_dir='.', resume='', start_epoch=0, aspect_ratio_group_factor=3, rpn_score_thresh=None, trainable_backbone_layers=None, data_augmentation='hflip', sync_bn=False, test_only=True, use_deterministic_algorithms=False, world_size=1, dist_url='env://', weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1', weights_backbone=None, amp=False, use_copypaste=False, backend='pil', use_v2=False, distributed=False)
Loading data
Loading annotations into memory...
Done (t=17.27s)
Creating index...
index created!
Loading annotations into memory...
Done (t=0.69s)
Creating index...
index created!
Creating data loaders
Using [0, 0.5, 0.6299605249474366, 0.7937005259840997, 1.0, 1.2599210498948732, 1.5874010519681994, 2.0, inf] as bins for aspect ratio quantization
Count of instances per bin: [  104   982 24236  2332  8225 74466  5763  1158]
Creating model
Test:  [   0/5000]  eta: 6:45:52  model_time: 4.3980 (4.3980)  evaluator_time: 0.0183 (0.0183)  time: 4.8706  data: 0.4510  max mem: 611
Test:  [ 100/5000]  eta: 0:08:16  model_time: 0.0282 (0.0835)  evaluator_time: 0.0068 (0.0091)  time: 0.0522  data: 0.0039  max mem: 663
Test:  [ 200/5000]  eta: 0:05:51  model_time: 0.0257 (0.0575)  evaluator_time: 0.0056 (0.0092)  time: 0.0449  data: 0.0027  max mem: 663
Test:  [ 300/5000]  eta: 0:04:57  model_time: 0.0282 (0.0486)  evaluator_time: 0.0069 (0.0091)  time: 0.0472  data: 0.0027  max mem: 663
Test:  [ 400/5000]  eta: 0:04:27  model_time: 0.0255 (0.0437)  evaluator_time: 0.0052 (0.0089)  time: 0.0420  data: 0.0026  max mem: 663
Test:  [ 500/5000]  eta: 0:04:03  model_time: 0.0251 (0.0404)  evaluator_time: 0.0063 (0.0087)  time: 0.0370  data: 0.0024  max mem: 663
Test:  [ 600/5000]  eta: 0:03:50  model_time: 0.0260 (0.0385)  evaluator_time: 0.0067 (0.0089)  time: 0.0436  data: 0.0034  max mem: 663
Test:  [ 700/5000]  eta: 0:03:38  model_time: 0.0253 (0.0369)  evaluator_time: 0.0056 (0.0090)  time: 0.0379  data: 0.0028  max mem: 663
Test:  [ 800/5000]  eta: 0:03:27  model_time: 0.0252 (0.0356)  evaluator_time: 0.0080 (0.0091)  time: 0.0403  data: 0.0028  max mem: 663
Test:  [ 900/5000]  eta: 0:03:19  model_time: 0.0254 (0.0348)  evaluator_time: 0.0061 (0.0091)  time: 0.0414  data: 0.0029  max mem: 663
Test:  [1000/5000]  eta: 0:03:10  model_time: 0.0254 (0.0341)  evaluator_time: 0.0048 (0.0090)  time: 0.0402  data: 0.0026  max mem: 663
Test:  [1100/5000]  eta: 0:03:02  model_time: 0.0259 (0.0334)  evaluator_time: 0.0073 (0.0090)  time: 0.0440  data: 0.0030  max mem: 663
Test:  [1200/5000]  eta: 0:02:55  model_time: 0.0255 (0.0328)  evaluator_time: 0.0065 (0.0090)  time: 0.0390  data: 0.0028  max mem: 663
Test:  [1300/5000]  eta: 0:02:49  model_time: 0.0254 (0.0323)  evaluator_time: 0.0047 (0.0090)  time: 0.0364  data: 0.0025  max mem: 663
Test:  [1400/5000]  eta: 0:02:43  model_time: 0.0267 (0.0319)  evaluator_time: 0.0055 (0.0090)  time: 0.0423  data: 0.0028  max mem: 663
Test:  [1500/5000]  eta: 0:02:37  model_time: 0.0250 (0.0316)  evaluator_time: 0.0054 (0.0089)  time: 0.0363  data: 0.0027  max mem: 663
Test:  [1600/5000]  eta: 0:02:31  model_time: 0.0253 (0.0312)  evaluator_time: 0.0070 (0.0090)  time: 0.0429  data: 0.0026  max mem: 663
Test:  [1700/5000]  eta: 0:02:26  model_time: 0.0245 (0.0310)  evaluator_time: 0.0045 (0.0090)  time: 0.0417  data: 0.0026  max mem: 663
Test:  [1800/5000]  eta: 0:02:19  model_time: 0.0212 (0.0306)  evaluator_time: 0.0043 (0.0088)  time: 0.0352  data: 0.0027  max mem: 663
Test:  [1900/5000]  eta: 0:02:14  model_time: 0.0242 (0.0303)  evaluator_time: 0.0041 (0.0088)  time: 0.0355  data: 0.0026  max mem: 663
Test:  [2000/5000]  eta: 0:02:09  model_time: 0.0260 (0.0302)  evaluator_time: 0.0057 (0.0088)  time: 0.0415  data: 0.0027  max mem: 663
Test:  [2100/5000]  eta: 0:02:04  model_time: 0.0257 (0.0300)  evaluator_time: 0.0054 (0.0088)  time: 0.0416  data: 0.0025  max mem: 663
Test:  [2200/5000]  eta: 0:02:00  model_time: 0.0258 (0.0299)  evaluator_time: 0.0053 (0.0088)  time: 0.0371  data: 0.0026  max mem: 663
Test:  [2300/5000]  eta: 0:01:55  model_time: 0.0255 (0.0297)  evaluator_time: 0.0055 (0.0088)  time: 0.0351  data: 0.0025  max mem: 663
Test:  [2400/5000]  eta: 0:01:50  model_time: 0.0258 (0.0296)  evaluator_time: 0.0062 (0.0087)  time: 0.0397  data: 0.0026  max mem: 663
Test:  [2500/5000]  eta: 0:01:45  model_time: 0.0262 (0.0295)  evaluator_time: 0.0056 (0.0087)  time: 0.0399  data: 0.0028  max mem: 663
Test:  [2600/5000]  eta: 0:01:41  model_time: 0.0256 (0.0293)  evaluator_time: 0.0058 (0.0087)  time: 0.0371  data: 0.0026  max mem: 663
Test:  [2700/5000]  eta: 0:01:36  model_time: 0.0247 (0.0292)  evaluator_time: 0.0053 (0.0087)  time: 0.0346  data: 0.0025  max mem: 663
Test:  [2800/5000]  eta: 0:01:31  model_time: 0.0221 (0.0290)  evaluator_time: 0.0052 (0.0087)  time: 0.0336  data: 0.0027  max mem: 663
Test:  [2900/5000]  eta: 0:01:27  model_time: 0.0226 (0.0288)  evaluator_time: 0.0065 (0.0087)  time: 0.0343  data: 0.0025  max mem: 663
Test:  [3000/5000]  eta: 0:01:23  model_time: 0.0261 (0.0288)  evaluator_time: 0.0053 (0.0087)  time: 0.0402  data: 0.0028  max mem: 663
Test:  [3100/5000]  eta: 0:01:18  model_time: 0.0260 (0.0287)  evaluator_time: 0.0056 (0.0086)  time: 0.0388  data: 0.0027  max mem: 663
Test:  [3200/5000]  eta: 0:01:14  model_time: 0.0254 (0.0286)  evaluator_time: 0.0057 (0.0086)  time: 0.0359  data: 0.0026  max mem: 663
Test:  [3300/5000]  eta: 0:01:10  model_time: 0.0259 (0.0286)  evaluator_time: 0.0075 (0.0087)  time: 0.0410  data: 0.0026  max mem: 663
Test:  [3400/5000]  eta: 0:01:06  model_time: 0.0255 (0.0285)  evaluator_time: 0.0052 (0.0087)  time: 0.0375  data: 0.0028  max mem: 663
Test:  [3500/5000]  eta: 0:01:01  model_time: 0.0256 (0.0284)  evaluator_time: 0.0046 (0.0087)  time: 0.0364  data: 0.0025  max mem: 663
Test:  [3600/5000]  eta: 0:00:57  model_time: 0.0255 (0.0284)  evaluator_time: 0.0058 (0.0087)  time: 0.0417  data: 0.0026  max mem: 663
Test:  [3700/5000]  eta: 0:00:53  model_time: 0.0225 (0.0283)  evaluator_time: 0.0062 (0.0087)  time: 0.0345  data: 0.0025  max mem: 663
Test:  [3800/5000]  eta: 0:00:49  model_time: 0.0250 (0.0282)  evaluator_time: 0.0060 (0.0087)  time: 0.0396  data: 0.0024  max mem: 663
Test:  [3900/5000]  eta: 0:00:44  model_time: 0.0243 (0.0281)  evaluator_time: 0.0051 (0.0087)  time: 0.0341  data: 0.0024  max mem: 663
Test:  [4000/5000]  eta: 0:00:40  model_time: 0.0257 (0.0280)  evaluator_time: 0.0051 (0.0086)  time: 0.0366  data: 0.0026  max mem: 663
Test:  [4100/5000]  eta: 0:00:36  model_time: 0.0254 (0.0280)  evaluator_time: 0.0058 (0.0087)  time: 0.0392  data: 0.0027  max mem: 663
Test:  [4200/5000]  eta: 0:00:32  model_time: 0.0256 (0.0280)  evaluator_time: 0.0057 (0.0087)  time: 0.0378  data: 0.0026  max mem: 663
Test:  [4300/5000]  eta: 0:00:28  model_time: 0.0255 (0.0279)  evaluator_time: 0.0063 (0.0086)  time: 0.0377  data: 0.0026  max mem: 663
Test:  [4400/5000]  eta: 0:00:24  model_time: 0.0253 (0.0279)  evaluator_time: 0.0066 (0.0086)  time: 0.0407  data: 0.0026  max mem: 663
Test:  [4500/5000]  eta: 0:00:20  model_time: 0.0255 (0.0278)  evaluator_time: 0.0067 (0.0086)  time: 0.0414  data: 0.0027  max mem: 663
Test:  [4600/5000]  eta: 0:00:16  model_time: 0.0251 (0.0278)  evaluator_time: 0.0058 (0.0086)  time: 0.0360  data: 0.0027  max mem: 663
Test:  [4700/5000]  eta: 0:00:12  model_time: 0.0257 (0.0278)  evaluator_time: 0.0061 (0.0086)  time: 0.0409  data: 0.0026  max mem: 663
Test:  [4800/5000]  eta: 0:00:08  model_time: 0.0249 (0.0278)  evaluator_time: 0.0057 (0.0086)  time: 0.0398  data: 0.0026  max mem: 663
Test:  [4900/5000]  eta: 0:00:04  model_time: 0.0251 (0.0277)  evaluator_time: 0.0062 (0.0086)  time: 0.0375  data: 0.0026  max mem: 663
Test:  [4999/5000]  eta: 0:00:00  model_time: 0.0248 (0.0277)  evaluator_time: 0.0055 (0.0086)  time: 0.0379  data: 0.0024  max mem: 663
Test: Total time: 0:03:21 (0.0403 s / it)
Averaged stats: model_time: 0.0248 (0.0277)  evaluator_time: 0.0055 (0.0086)
Accumulating evaluation results...
DONE (t=8.12s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.36919
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.58536
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.39610
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.21218
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.40304
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.48142
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.30733
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.48463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.50840
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.31743
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.54427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.64892
```

</details>

<details>
  
```bash
Not using distributed mode
Namespace(data_path='/home/van-tien.pham/data/coco/', dataset='coco', model='maskrcnn_resnet50_fpn', device='cuda', batch_size=2, epochs=26, workers=4, opt='sgd', lr=0.02, momentum=0.9, weight_decay=0.0001, norm_weight_decay=None, lr_scheduler='multisteplr', lr_step_size=8, lr_steps=[16, 22], lr_gamma=0.1, print_freq=20, output_dir='.', resume='', start_epoch=0, aspect_ratio_group_factor=3, rpn_score_thresh=None, trainable_backbone_layers=None, data_augmentation='hflip', sync_bn=False, test_only=True, use_deterministic_algorithms=False, world_size=1, dist_url='env://', weights='MaskRCNN_ResNet50_FPN_Weights.COCO_V1', weights_backbone=None, amp=False, use_copypaste=False, backend='pil', use_v2=False, distributed=False)
Loading data
Loading annotations into memory...
Done (t=19.31s)
Creating index...
index created!
Loading annotations into memory...
Done (t=0.64s)
Creating index...
index created!
Creating data loaders
Using [0, 0.5, 0.6299605249474366, 0.7937005259840997, 1.0, 1.2599210498948732, 1.5874010519681994, 2.0, inf] as bins for aspect ratio quantization
Count of instances per bin: [  104   982 24236  2332  8225 74466  5763  1158]
Creating model
Downloading: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth" to /home/van-tien.pham/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
100%|██████████████████████████████████████████████████████████████████████████████| 170M/170M [00:02<00:00, 85.8MB/s]
Test:  [   0/5000]  eta: 5:17:12  model_time: 3.2241 (3.2241)  evaluator_time: 0.1573 (0.1573)  time: 3.8066  data: 0.4233  max mem: 620
Test:  [ 100/5000]  eta: 0:14:58  model_time: 0.1018 (0.1191)  evaluator_time: 0.0561 (0.0536)  time: 0.1785  data: 0.0030  max mem: 673
Test:  [ 200/5000]  eta: 0:12:48  model_time: 0.0447 (0.0986)  evaluator_time: 0.0339 (0.0532)  time: 0.1344  data: 0.0029  max mem: 673
Test:  [ 300/5000]  eta: 0:11:49  model_time: 0.0517 (0.0906)  evaluator_time: 0.0383 (0.0529)  time: 0.1437  data: 0.0028  max mem: 673
Test:  [ 400/5000]  eta: 0:11:14  model_time: 0.0431 (0.0870)  evaluator_time: 0.0280 (0.0524)  time: 0.1124  data: 0.0028  max mem: 673
Test:  [ 500/5000]  eta: 0:10:35  model_time: 0.0768 (0.0827)  evaluator_time: 0.0539 (0.0516)  time: 0.1605  data: 0.0030  max mem: 673
Test:  [ 600/5000]  eta: 0:10:19  model_time: 0.0503 (0.0813)  evaluator_time: 0.0292 (0.0526)  time: 0.1247  data: 0.0028  max mem: 673
Test:  [ 700/5000]  eta: 0:10:03  model_time: 0.0580 (0.0805)  evaluator_time: 0.0353 (0.0530)  time: 0.1197  data: 0.0030  max mem: 673
Test:  [ 800/5000]  eta: 0:09:48  model_time: 0.0708 (0.0801)  evaluator_time: 0.0486 (0.0533)  time: 0.1435  data: 0.0029  max mem: 673
Test:  [ 900/5000]  eta: 0:09:28  model_time: 0.0785 (0.0793)  evaluator_time: 0.0476 (0.0530)  time: 0.1503  data: 0.0028  max mem: 673
Test:  [1000/5000]  eta: 0:09:09  model_time: 0.0460 (0.0785)  evaluator_time: 0.0268 (0.0525)  time: 0.1034  data: 0.0028  max mem: 673
Test:  [1100/5000]  eta: 0:08:54  model_time: 0.0623 (0.0779)  evaluator_time: 0.0429 (0.0526)  time: 0.1370  data: 0.0032  max mem: 673
Test:  [1200/5000]  eta: 0:08:39  model_time: 0.0704 (0.0776)  evaluator_time: 0.0544 (0.0528)  time: 0.1442  data: 0.0028  max mem: 673
Test:  [1300/5000]  eta: 0:08:26  model_time: 0.0402 (0.0774)  evaluator_time: 0.0225 (0.0532)  time: 0.1008  data: 0.0028  max mem: 673
Test:  [1400/5000]  eta: 0:08:11  model_time: 0.0444 (0.0769)  evaluator_time: 0.0276 (0.0533)  time: 0.1299  data: 0.0029  max mem: 673
Test:  [1500/5000]  eta: 0:07:56  model_time: 0.0672 (0.0765)  evaluator_time: 0.0468 (0.0532)  time: 0.1294  data: 0.0026  max mem: 673
Test:  [1600/5000]  eta: 0:07:46  model_time: 0.0771 (0.0769)  evaluator_time: 0.0515 (0.0539)  time: 0.1786  data: 0.0029  max mem: 673
Test:  [1700/5000]  eta: 0:07:31  model_time: 0.0429 (0.0766)  evaluator_time: 0.0250 (0.0538)  time: 0.1152  data: 0.0033  max mem: 673
Test:  [1800/5000]  eta: 0:07:13  model_time: 0.0485 (0.0760)  evaluator_time: 0.0247 (0.0534)  time: 0.1024  data: 0.0029  max mem: 673
Test:  [1900/5000]  eta: 0:07:00  model_time: 0.0419 (0.0760)  evaluator_time: 0.0204 (0.0536)  time: 0.1108  data: 0.0028  max mem: 673
Test:  [2000/5000]  eta: 0:06:46  model_time: 0.0563 (0.0758)  evaluator_time: 0.0328 (0.0534)  time: 0.1402  data: 0.0028  max mem: 673
Test:  [2100/5000]  eta: 0:06:32  model_time: 0.0506 (0.0755)  evaluator_time: 0.0412 (0.0535)  time: 0.1442  data: 0.0027  max mem: 673
Test:  [2200/5000]  eta: 0:06:17  model_time: 0.0487 (0.0753)  evaluator_time: 0.0324 (0.0532)  time: 0.1058  data: 0.0030  max mem: 673
Test:  [2300/5000]  eta: 0:06:03  model_time: 0.0493 (0.0752)  evaluator_time: 0.0328 (0.0532)  time: 0.1091  data: 0.0030  max mem: 673
Test:  [2400/5000]  eta: 0:05:48  model_time: 0.0714 (0.0749)  evaluator_time: 0.0450 (0.0530)  time: 0.1439  data: 0.0028  max mem: 673
Test:  [2500/5000]  eta: 0:05:35  model_time: 0.0489 (0.0748)  evaluator_time: 0.0356 (0.0531)  time: 0.1243  data: 0.0032  max mem: 673
Test:  [2600/5000]  eta: 0:05:22  model_time: 0.0493 (0.0748)  evaluator_time: 0.0387 (0.0533)  time: 0.1169  data: 0.0029  max mem: 673
Test:  [2700/5000]  eta: 0:05:07  model_time: 0.0443 (0.0745)  evaluator_time: 0.0278 (0.0532)  time: 0.0955  data: 0.0028  max mem: 673
Test:  [2800/5000]  eta: 0:04:53  model_time: 0.0521 (0.0744)  evaluator_time: 0.0389 (0.0532)  time: 0.1217  data: 0.0030  max mem: 673
Test:  [2900/5000]  eta: 0:04:40  model_time: 0.0656 (0.0743)  evaluator_time: 0.0468 (0.0531)  time: 0.1335  data: 0.0027  max mem: 673
Test:  [3000/5000]  eta: 0:04:26  model_time: 0.0530 (0.0742)  evaluator_time: 0.0374 (0.0531)  time: 0.1273  data: 0.0027  max mem: 673
Test:  [3100/5000]  eta: 0:04:12  model_time: 0.0421 (0.0739)  evaluator_time: 0.0317 (0.0528)  time: 0.1166  data: 0.0040  max mem: 673
Test:  [3200/5000]  eta: 0:04:01  model_time: 0.0691 (0.0752)  evaluator_time: 0.0456 (0.0528)  time: 0.1291  data: 0.0028  max mem: 673
Test:  [3300/5000]  eta: 0:03:48  model_time: 0.0554 (0.0754)  evaluator_time: 0.0348 (0.0530)  time: 0.1506  data: 0.0029  max mem: 673
Test:  [3400/5000]  eta: 0:03:34  model_time: 0.0406 (0.0753)  evaluator_time: 0.0250 (0.0530)  time: 0.1025  data: 0.0027  max mem: 673
Test:  [3500/5000]  eta: 0:03:21  model_time: 0.0438 (0.0751)  evaluator_time: 0.0271 (0.0530)  time: 0.1009  data: 0.0027  max mem: 673
Test:  [3600/5000]  eta: 0:03:07  model_time: 0.0505 (0.0750)  evaluator_time: 0.0437 (0.0531)  time: 0.1550  data: 0.0027  max mem: 673
Test:  [3700/5000]  eta: 0:02:53  model_time: 0.0476 (0.0747)  evaluator_time: 0.0346 (0.0529)  time: 0.1209  data: 0.0028  max mem: 673
Test:  [3800/5000]  eta: 0:02:40  model_time: 0.0461 (0.0747)  evaluator_time: 0.0414 (0.0529)  time: 0.1527  data: 0.0029  max mem: 673
Test:  [3900/5000]  eta: 0:02:26  model_time: 0.0468 (0.0744)  evaluator_time: 0.0302 (0.0527)  time: 0.0983  data: 0.0035  max mem: 673
Test:  [4000/5000]  eta: 0:02:12  model_time: 0.0430 (0.0743)  evaluator_time: 0.0252 (0.0526)  time: 0.1124  data: 0.0041  max mem: 673
Test:  [4100/5000]  eta: 0:01:59  model_time: 0.0512 (0.0744)  evaluator_time: 0.0312 (0.0528)  time: 0.1222  data: 0.0027  max mem: 673
Test:  [4200/5000]  eta: 0:01:46  model_time: 0.0526 (0.0743)  evaluator_time: 0.0313 (0.0529)  time: 0.1258  data: 0.0029  max mem: 673
Test:  [4300/5000]  eta: 0:01:33  model_time: 0.0533 (0.0743)  evaluator_time: 0.0421 (0.0529)  time: 0.1450  data: 0.0028  max mem: 673
Test:  [4400/5000]  eta: 0:01:19  model_time: 0.0649 (0.0739)  evaluator_time: 0.0489 (0.0527)  time: 0.1446  data: 0.0041  max mem: 673
Test:  [4500/5000]  eta: 0:01:06  model_time: 0.0564 (0.0738)  evaluator_time: 0.0412 (0.0527)  time: 0.1641  data: 0.0027  max mem: 673
Test:  [4600/5000]  eta: 0:00:52  model_time: 0.0465 (0.0737)  evaluator_time: 0.0333 (0.0526)  time: 0.1070  data: 0.0026  max mem: 673
Test:  [4700/5000]  eta: 0:00:39  model_time: 0.0516 (0.0736)  evaluator_time: 0.0365 (0.0526)  time: 0.1384  data: 0.0028  max mem: 673
Test:  [4800/5000]  eta: 0:00:26  model_time: 0.0475 (0.0735)  evaluator_time: 0.0389 (0.0526)  time: 0.1329  data: 0.0027  max mem: 673
Test:  [4900/5000]  eta: 0:00:13  model_time: 0.0501 (0.0732)  evaluator_time: 0.0349 (0.0525)  time: 0.1064  data: 0.0027  max mem: 673
Test:  [4999/5000]  eta: 0:00:00  model_time: 0.0458 (0.0731)  evaluator_time: 0.0358 (0.0525)  time: 0.1339  data: 0.0027  max mem: 673
Test: Total time: 0:10:57 (0.1316 s / it)
Averaged stats: model_time: 0.0458 (0.0731)  evaluator_time: 0.0358 (0.0525)
Accumulating evaluation results...
DONE (t=8.39s).
Accumulating evaluation results...
DONE (t=8.30s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.37834
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.59155
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.41119
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.21570
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.41370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.49368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.31477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.49589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.52011
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.32554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.55816
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.66628
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.34522
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.56020
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.36730
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.15870
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.37265
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.50659
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.29658
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.45563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.47560
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.27311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.51396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.63791
```

</details>

<details>

```bash
Not using distributed mode
Namespace(data_path='/home/van-tien.pham/data/coco/', dataset='coco_kp', model='keypointrcnn_resnet50_fpn', device='cuda', batch_size=2, epochs=26, workers=4, opt='sgd', lr=0.02, momentum=0.9, weight_decay=0.0001, norm_weight_decay=None, lr_scheduler='multisteplr', lr_step_size=8, lr_steps=[16, 22], lr_gamma=0.1, print_freq=20, output_dir='.', resume='', start_epoch=0, aspect_ratio_group_factor=3, rpn_score_thresh=None, trainable_backbone_layers=None, data_augmentation='hflip', sync_bn=False, test_only=True, use_deterministic_algorithms=False, world_size=1, dist_url='env://', weights='KeypointRCNN_ResNet50_FPN_Weights.COCO_V1', weights_backbone=None, amp=False, use_copypaste=False, backend='pil', use_v2=False, distributed=False)
Loading data
Loading annotations into memory...
Done (t=9.13s)
Creating index...
index created!
Loading annotations into memory...
Done (t=0.37s)
Creating index...
index created!
Creating data loaders
Using [0, 0.5, 0.6299605249474366, 0.7937005259840997, 1.0, 1.2599210498948732, 1.5874010519681994, 2.0, inf] as bins for aspect ratio quantization
Count of instances per bin: [   58   492 10629  1166  3284 29808  2213   412]
Creating model
Downloading: "https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth" to /home/van-tien.pham/.cache/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 226M/226M [00:02<00:00, 108MB/s]
Test:  [   0/5000]  eta: 3:50:46  model_time: 2.4811 (2.4811)  evaluator_time: 0.0100 (0.0100)  time: 2.7693  data: 0.2765  max mem: 678
Test:  [ 100/5000]  eta: 0:08:06  model_time: 0.0395 (0.0831)  evaluator_time: 0.0058 (0.0098)  time: 0.0639  data: 0.0025  max mem: 730
Test:  [ 200/5000]  eta: 0:06:16  model_time: 0.0381 (0.0638)  evaluator_time: 0.0065 (0.0097)  time: 0.0603  data: 0.0026  max mem: 730
Test:  [ 300/5000]  eta: 0:05:35  model_time: 0.0371 (0.0576)  evaluator_time: 0.0051 (0.0092)  time: 0.0589  data: 0.0023  max mem: 730
Test:  [ 400/5000]  eta: 0:05:12  model_time: 0.0366 (0.0544)  evaluator_time: 0.0062 (0.0091)  time: 0.0583  data: 0.0023  max mem: 730
Test:  [ 500/5000]  eta: 0:04:50  model_time: 0.0367 (0.0515)  evaluator_time: 0.0052 (0.0088)  time: 0.0499  data: 0.0024  max mem: 730
Test:  [ 600/5000]  eta: 0:04:40  model_time: 0.0365 (0.0503)  evaluator_time: 0.0058 (0.0092)  time: 0.0511  data: 0.0025  max mem: 730
Test:  [ 700/5000]  eta: 0:04:29  model_time: 0.0371 (0.0492)  evaluator_time: 0.0054 (0.0094)  time: 0.0493  data: 0.0024  max mem: 730
Test:  [ 800/5000]  eta: 0:04:17  model_time: 0.0370 (0.0479)  evaluator_time: 0.0051 (0.0093)  time: 0.0460  data: 0.0022  max mem: 730
Test:  [ 900/5000]  eta: 0:04:07  model_time: 0.0357 (0.0472)  evaluator_time: 0.0047 (0.0093)  time: 0.0484  data: 0.0019  max mem: 730
Test:  [1000/5000]  eta: 0:03:58  model_time: 0.0395 (0.0465)  evaluator_time: 0.0061 (0.0091)  time: 0.0572  data: 0.0026  max mem: 730
Test:  [1100/5000]  eta: 0:03:49  model_time: 0.0366 (0.0459)  evaluator_time: 0.0058 (0.0091)  time: 0.0539  data: 0.0023  max mem: 730
Test:  [1200/5000]  eta: 0:03:42  model_time: 0.0381 (0.0455)  evaluator_time: 0.0053 (0.0091)  time: 0.0559  data: 0.0022  max mem: 730
Test:  [1300/5000]  eta: 0:03:36  model_time: 0.0365 (0.0454)  evaluator_time: 0.0056 (0.0094)  time: 0.0482  data: 0.0021  max mem: 730
Test:  [1400/5000]  eta: 0:03:30  model_time: 0.0386 (0.0451)  evaluator_time: 0.0065 (0.0095)  time: 0.0630  data: 0.0023  max mem: 730
Test:  [1500/5000]  eta: 0:03:22  model_time: 0.0355 (0.0447)  evaluator_time: 0.0048 (0.0094)  time: 0.0495  data: 0.0021  max mem: 730
Test:  [1600/5000]  eta: 0:03:15  model_time: 0.0387 (0.0444)  evaluator_time: 0.0063 (0.0094)  time: 0.0572  data: 0.0025  max mem: 730
Test:  [1700/5000]  eta: 0:03:09  model_time: 0.0369 (0.0443)  evaluator_time: 0.0051 (0.0095)  time: 0.0597  data: 0.0023  max mem: 730
Test:  [1800/5000]  eta: 0:03:02  model_time: 0.0362 (0.0440)  evaluator_time: 0.0049 (0.0094)  time: 0.0541  data: 0.0020  max mem: 730
Test:  [1900/5000]  eta: 0:02:56  model_time: 0.0355 (0.0438)  evaluator_time: 0.0050 (0.0094)  time: 0.0469  data: 0.0020  max mem: 730
Test:  [2000/5000]  eta: 0:02:50  model_time: 0.0379 (0.0437)  evaluator_time: 0.0054 (0.0094)  time: 0.0523  data: 0.0024  max mem: 730
Test:  [2100/5000]  eta: 0:02:43  model_time: 0.0388 (0.0435)  evaluator_time: 0.0055 (0.0093)  time: 0.0553  data: 0.0023  max mem: 730
Test:  [2200/5000]  eta: 0:02:37  model_time: 0.0389 (0.0433)  evaluator_time: 0.0055 (0.0092)  time: 0.0506  data: 0.0024  max mem: 730
Test:  [2300/5000]  eta: 0:02:31  model_time: 0.0347 (0.0432)  evaluator_time: 0.0047 (0.0092)  time: 0.0454  data: 0.0020  max mem: 730
Test:  [2400/5000]  eta: 0:02:25  model_time: 0.0369 (0.0430)  evaluator_time: 0.0054 (0.0092)  time: 0.0539  data: 0.0023  max mem: 730
Test:  [2500/5000]  eta: 0:02:19  model_time: 0.0368 (0.0429)  evaluator_time: 0.0053 (0.0093)  time: 0.0523  data: 0.0024  max mem: 730
Test:  [2600/5000]  eta: 0:02:13  model_time: 0.0362 (0.0428)  evaluator_time: 0.0048 (0.0092)  time: 0.0496  data: 0.0020  max mem: 730
Test:  [2700/5000]  eta: 0:02:07  model_time: 0.0362 (0.0427)  evaluator_time: 0.0048 (0.0092)  time: 0.0464  data: 0.0022  max mem: 730
Test:  [2800/5000]  eta: 0:02:01  model_time: 0.0388 (0.0425)  evaluator_time: 0.0059 (0.0092)  time: 0.0523  data: 0.0024  max mem: 730
Test:  [2900/5000]  eta: 0:01:55  model_time: 0.0372 (0.0424)  evaluator_time: 0.0053 (0.0092)  time: 0.0498  data: 0.0022  max mem: 730
Test:  [3000/5000]  eta: 0:01:50  model_time: 0.0366 (0.0423)  evaluator_time: 0.0052 (0.0091)  time: 0.0491  data: 0.0022  max mem: 730
Test:  [3100/5000]  eta: 0:01:44  model_time: 0.0365 (0.0423)  evaluator_time: 0.0049 (0.0091)  time: 0.0480  data: 0.0022  max mem: 730
Test:  [3200/5000]  eta: 0:01:38  model_time: 0.0368 (0.0422)  evaluator_time: 0.0050 (0.0091)  time: 0.0489  data: 0.0022  max mem: 730
Test:  [3300/5000]  eta: 0:01:33  model_time: 0.0363 (0.0421)  evaluator_time: 0.0050 (0.0091)  time: 0.0531  data: 0.0022  max mem: 730
Test:  [3400/5000]  eta: 0:01:27  model_time: 0.0345 (0.0420)  evaluator_time: 0.0049 (0.0091)  time: 0.0446  data: 0.0021  max mem: 730
Test:  [3500/5000]  eta: 0:01:21  model_time: 0.0357 (0.0419)  evaluator_time: 0.0049 (0.0091)  time: 0.0467  data: 0.0021  max mem: 730
Test:  [3600/5000]  eta: 0:01:16  model_time: 0.0368 (0.0419)  evaluator_time: 0.0047 (0.0090)  time: 0.0475  data: 0.0022  max mem: 730
Test:  [3700/5000]  eta: 0:01:10  model_time: 0.0367 (0.0418)  evaluator_time: 0.0057 (0.0090)  time: 0.0559  data: 0.0023  max mem: 730
Test:  [3800/5000]  eta: 0:01:05  model_time: 0.0368 (0.0417)  evaluator_time: 0.0055 (0.0090)  time: 0.0604  data: 0.0023  max mem: 730
Test:  [3900/5000]  eta: 0:00:59  model_time: 0.0347 (0.0416)  evaluator_time: 0.0043 (0.0090)  time: 0.0431  data: 0.0020  max mem: 730
Test:  [4000/5000]  eta: 0:00:53  model_time: 0.0372 (0.0415)  evaluator_time: 0.0053 (0.0089)  time: 0.0503  data: 0.0021  max mem: 730
Test:  [4100/5000]  eta: 0:00:48  model_time: 0.0373 (0.0415)  evaluator_time: 0.0051 (0.0089)  time: 0.0481  data: 0.0022  max mem: 730
Test:  [4200/5000]  eta: 0:00:43  model_time: 0.0370 (0.0414)  evaluator_time: 0.0050 (0.0089)  time: 0.0499  data: 0.0022  max mem: 730
Test:  [4300/5000]  eta: 0:00:37  model_time: 0.0385 (0.0413)  evaluator_time: 0.0058 (0.0089)  time: 0.0507  data: 0.0025  max mem: 730
Test:  [4400/5000]  eta: 0:00:32  model_time: 0.0376 (0.0413)  evaluator_time: 0.0053 (0.0089)  time: 0.0556  data: 0.0021  max mem: 730
Test:  [4500/5000]  eta: 0:00:26  model_time: 0.0369 (0.0412)  evaluator_time: 0.0051 (0.0088)  time: 0.0552  data: 0.0021  max mem: 730
Test:  [4600/5000]  eta: 0:00:21  model_time: 0.0368 (0.0411)  evaluator_time: 0.0048 (0.0088)  time: 0.0475  data: 0.0022  max mem: 730
Test:  [4700/5000]  eta: 0:00:16  model_time: 0.0389 (0.0411)  evaluator_time: 0.0076 (0.0088)  time: 0.0638  data: 0.0025  max mem: 730
Test:  [4800/5000]  eta: 0:00:10  model_time: 0.0374 (0.0411)  evaluator_time: 0.0056 (0.0088)  time: 0.0566  data: 0.0021  max mem: 730
Test:  [4900/5000]  eta: 0:00:05  model_time: 0.0391 (0.0410)  evaluator_time: 0.0070 (0.0088)  time: 0.0555  data: 0.0025  max mem: 730
Test:  [4999/5000]  eta: 0:00:00  model_time: 0.0363 (0.0410)  evaluator_time: 0.0050 (0.0087)  time: 0.0454  data: 0.0022  max mem: 730
Test: Total time: 0:04:26 (0.0533 s / it)
Averaged stats: model_time: 0.0363 (0.0410)  evaluator_time: 0.0050 (0.0087)
Accumulating evaluation results...
DONE (t=1.21s).
Accumulating evaluation results...
DONE (t=0.36s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.54582
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.83001
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.59577
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.37962
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.62678
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.70275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.18750
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.55486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.63622
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.49383
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.69960
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.77458
IoU metric: keypoints
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.65058
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.86105
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.71384
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.60316
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.73068
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.71759
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.90743
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.77424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.66979
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.78573
```

 </details>
