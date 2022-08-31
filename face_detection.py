from __future__ import print_function
import os, sys
import cv2
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import copy

sys.path.insert(0, './Retina_Face')

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from Retina_Face.utils.nms.py_cpu_nms import py_cpu_nms
from Retina_Face.models.retinaface import RetinaFace
from Retina_Face.utils.box_utils import decode, decode_landm
from Retina_Face.utils.timer import Timer

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class Arguments():
    pass
args = Arguments()
args.trained_model = "./Retina_Face/weights/Resnet50_Final.pth"
args.network = "resnet50"
args.cpu = False
args.dataset = "FDDB"
args.confidence_threshold = 0.02
args.top_k = 5
args.nms_threshold = 0.4
args.keep_top_k = 1
args.save_image = True
args.save_folder = './Retina_Face/results'
args.vis_thres = 0.5

def retina_face(global_people_info):
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # save file
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    fw = open(os.path.join(args.save_folder, args.dataset + '_dets.txt'), 'w')

    # testing dataset
    testset_folder = os.path.join('Retina_Face/data', args.dataset, 'images/')
    testset_list = os.path.join('Retina_Face/data', args.dataset, 'img_list.txt')
    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(global_people_info)

    # testing scale
    resize = 1

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    frames_after_retina = []

    for i in range(len(global_people_info)):
        if global_people_info[i].frame_number % 5 == 0:
            img_raw = copy.deepcopy(global_people_info[i].frame_person)
            [height, width, channels] = img_raw.shape
            img = np.float32(img_raw)

            if resize != 1:
                img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)

            _t['forward_pass'].tic()
            loc, conf, landms = net(img)  # forward pass
            _t['forward_pass'].toc()
            _t['misc'].tic()
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                    img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                    img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            # ignore low scores
            inds = np.where(scores > args.confidence_threshold)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1][:args.top_k]
            order = scores.argsort()[::-1]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, args.nms_threshold)

            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            dets = dets[:args.keep_top_k, :]
            landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)
            _t['misc'].toc()

            # save dets
            fw.write('{:s}\n'.format("face_idx" + str(i)))
            fw.write('{:.1f}\n'.format(dets.shape[0]))
            # for k in range(dets.shape[0]):
            if len(dets) > 0:
                ymin = dets[0, 0]
                xmin = dets[0, 1]
                ymax = dets[0, 2]
                xmax = dets[0, 3]
                score = dets[0, 4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                dets[0, 0] = max(dets[0, 0] - (30*h/100), 0)
                dets[0, 1] = max(dets[0, 1] - (30*w/100), 0)
                dets[0, 2] = min(dets[0, 2] + (30*h/100), height)
                dets[0, 3] = min(dets[0, 3] + (30*w/100), width)
                ymin = dets[0, 0]
                xmin = dets[0, 1]
                ymax = dets[0, 2]
                xmax = dets[0, 3]
                w = xmax - xmin + 1
                h = ymax - ymin + 1

                global_people_info[i].face_x_min = int(xmin)
                global_people_info[i].face_y_min = int(ymin)
                global_people_info[i].face_w = int(w)
                global_people_info[i].face_h = int(h)
                
                global_people_info[i].is_face = True
                if (global_people_info[i].face_w<0 or global_people_info[i].face_h<0):
                    global_people_info[i].is_face = False

                # print("aaaaaaaaaaaaaaaaaaaa")
                print(global_people_info[i].face_x_min, global_people_info[i].face_y_min, global_people_info[i].face_w, global_people_info[i].face_h, global_people_info[i].is_face)
                if (int(xmin)==int(xmax)):
                    print("error!!")

                if (int(ymin)==int(ymax)):
                    print("error!!")
                frame_cut = global_people_info[i].frame_person[int(xmin): int(xmax), int(ymin): int(ymax)]
                global_people_info[i].frame_face = copy.deepcopy(frame_cut)
                
                fw.write('{:d} {:d} {:d} {:d} {:.10f}\n'.format(int(xmin), int(ymin), int(w), int(h), score))

        #print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        # # show image
        # if args.save_image:
        #     for b in dets:
        #         print(b)
        #         if b[4] < args.vis_thres:
        #             continue
        #         text = "{:.4f}".format(b[4])
        #         b = list(map(int, b))
        #         cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 2)
        #         cx = b[0]
        #         cy = b[1] + 12
        #         cv2.putText(img_raw, text, (cx, cy),
        #                     cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                
        #     # save image
        #     if not os.path.exists("./results/"):
        #         os.makedirs("./results/")
        #     name = "./results/" + str(i) + ".jpg"
        #     if (len(dets) > 0):
        #         cv2.imwrite(name, frames_after_deep_sort[i][int(dets[0, 1]): int(dets[0, 3]), int(dets[0, 0]): int(dets[0, 2])])
        


    fw.close()
    return global_people_info