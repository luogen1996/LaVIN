import os
import sys
import tempfile
import json
from json import encoder
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

class COCOEvaler(object):
    def __init__(self, annfile):
        super(COCOEvaler, self).__init__()
        self.coco = COCO(annfile)
        if not os.path.exists('./tmp'):
            os.mkdir('./tmp')

    def eval(self, result):
        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir='./tmp')
        json.dump(result, in_file)
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(in_file.name)
        return cocoEval.eval

evaler=COCOEvaler('./captions_test5k.json')
import re
pattern = re.compile(r'The image features (.*?\.)')
# pred = pattern.findall(result)
# preds=json.load(open('cap_preds.json'))
# preds_=[]
# for key in preds:
#     preds_.append({
#         'image_id':key,
#         'caption':pattern.findall(preds[key]['pred'])[0]
#     })
#
#
# file=json.load(open('./captions_test5k.json'))
# print(file['images'])
evaler.eval(json.load(open('cap_preds.json')))

