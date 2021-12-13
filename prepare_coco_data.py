from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

data_root = 'data/coco/'
splits = ['train2017', 'val2017']
for split in splits:
    annFile = data_root+'annotations/instances_{}.json'.format(split)
    coco = COCO(annFile)
    indxs = coco.getImgIds()
    coco_ids = coco.getCatIds()
    continual_ids = np.arange(1, 81)
    id_mapping = {coco_id:con_id for coco_id, con_id in zip(coco_ids, continual_ids)}
    save_dir = data_root+'annotations/coco_masks/instance_{}'.format(split)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for indx in tqdm(indxs):
        img_meta = coco.loadImgs(indx)[0]
        annIds = coco.getAnnIds(imgIds=img_meta['id'])
        anns = coco.loadAnns(annIds)
        semantic_mask = np.zeros((img_meta['height'], img_meta['width']), dtype='uint8')
        for ann in anns:
            if ann['iscrowd']:
                continue
            catId = ann['category_id']
            mask = coco.annToMask(ann)
            semantic_mask[mask == 1] = id_mapping[catId]
        mask_img = Image.fromarray(semantic_mask)
        mask_img.save(os.path.join(save_dir, img_meta['file_name'].replace('jpg', 'png')))