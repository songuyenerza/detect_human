yolo_path = "yolov5"
from distutils.log import error
import sys
from tkinter import W, Image
from turtle import window_height
import cv2
import os
import numpy as np
sys.path.append(yolo_path)
from utils.augmentations import letterbox
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
from models.common import DetectMultiBackend
import torch
import torch.backends.cudnn as cudnn
from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()
import timeit
from PIL import Image

def maxx(box):
    conf_list = []
    if len(box) > 1:
        for b in box:
            conf_list.append(b[5])
        index = np.argmax(conf_list)
        return [box[index]]
    else:
        return box

def convert_box(box, img_width, img_height, cls):
    x0 = int((box[0] - ((box[2]) / 2)*1.1) * img_width)
    y0 = int((box[1] - ((box[3]) / 2)*1.1) * img_height)
    x1 = int((box[0] + ((box[2]) / 2)*1.1) * img_width)
    y1 = int((box[1] + ((box[3]) / 2)*1.1) * img_height)
    if x0<0:
        x0 = 0
    if y0<0:
        y0 = 0
    return [x0, y0, x1, y1, cls]
def convert_box_no(box, img_width, img_height, cls, conf):
    x0 = int(box[0] * img_width)
    y1 = int(box[1] * img_height)
    w = int(box[2] * img_width)
    h = int(box[3] * img_height)
    conf = conf.cpu().data.numpy()
    return [x0, y1, w, h, cls, float(conf)]

@torch.no_grad()
def load_model(weights="",  # model.pt path(s)
        data='data/coco.yaml',  # dataset.yaml path
        imgsz=[640, 640],  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    model.warmup(imgsz=(1 , 3, *imgsz))  # warmup
    # print("device",device)
    return model,device
@torch.no_grad()
def detect_box(model,
        device,
        source,  # file/dir/URL/glob, 0 for webcam
        imgsz=[640,640],  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.7,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        ):
    
    # Load model
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    im0s = source
    img = letterbox(im0s, imgsz, stride=stride, auto=pt)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)

    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im)
    im0s = source
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    result=[]
    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0= im0s.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        box_image=[]
        box_image_no = []
        # print(det[:, :4])
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            box_image=[]
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                line=(('%g ' * len(line)).rstrip() % line)
                line=line.split(" ")

                line= [float(value) if i!=0 else int(value) for i,value in enumerate(line)]
                cls=line[0]
                box=convert_box(line[1:],im0.shape[1],im0.shape[0], cls)
                box_no = convert_box_no(line[1:],im0.shape[1],im0.shape[0], cls, conf)
                # if box[0] > int(im0.shape[1]*0.02):
                    # if int(im0.shape[1]*0.1) < box[2] < int(im0.shape[1]*0.9):
                if cls == 0:
                    box_image.append(box)
                    box_image_no.append(box_no)
       

    return box_image, box_image_no

def crop_box(img_ori, box_img,img_output, check_crop):
    img = img_ori
    img_orii = img_ori.copy()
    if len(box_img)!= 0:
        crop_list = []
        for i in range(len(box_img)):
            croped = img_ori[box_img[i][1]:box_img[i][3], box_img[i][0]: box_img[i][2]]
            crop_list.append(croped)
            img = cv2.rectangle(img_orii, (box_img[i][0],box_img[i][1]), (box_img[i][2],box_img[i][3]), (0,0,255), 2)
            cv2.putText(img, str(box_img[i][4]) + str(box_img[i]), (box_img[i][0],box_img[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 1)
            if check_crop == True:
                cv2.imwrite(img_output , croped)
        # if check_crop == False:
        #     cv2.imwrite(img_output, img)
    if check_crop == True:
        return crop_list
    else:
        return img

def getMemoryUsage():
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    return "%d %s" % (usage["used"], usage["unit"])


def mer_box(boxs, w, h):
    box0 = []
    box1 = []
    box2=[]
    box3=[]

    for box in boxs:
        box0.append(box[0])
        box1.append(box[1])
        box2.append(box[2])
        box3.append(box[3])
    x0 = min(box0)
    y0 = min(box1)
    x1 = max(box2)
    y1 = max(box3)
    
    bounding_box = (int(x0) , int (y0) , int(x1), int(y1))
    return bounding_box


if __name__ == "__main__":
    folder_img = "/home/anlab/Desktop/Songuyen/PIl_detection/data_bicycle/images/train_new_2811/"
    folder_output = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/CVNet/rerank_yolo/data20221128/train/"
    weight = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/carton_box_prj/cp/yolov5l.pt"
    #link model: https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l.pt

    torch.cuda.set_per_process_memory_fraction(0.2, 0)

    model, device = load_model(weights=weight)
    print("GPU Memory_____: %s" % getMemoryUsage())
    count = 0
    tt = 0
    err = 0
    with open('/home/anlab/Desktop/Songuyen/PIl_detection/data_bicycle/images/train_new_2811/paths.txt','r') as f:
        IMAGE_PATH_DB = [line.strip('\n') for line in f.readlines()]


    for path in IMAGE_PATH_DB:

        img_ori = cv2.imread(folder_img + path)
        # center = img_ori.shapes
        tt+=1
        start = timeit.default_timer()
        box_img, box_image_no = detect_box(model, device, img_ori,imgsz=[640,640],conf_thres=0.6, iou_thres = 0.6)
        # print("box_image_no", box_image_no)
        img_output = folder_output + path
        stop = timeit.default_timer()
        print('Time: ', stop - start)  

        img_out = crop_box(img_ori, box_img, img_output, check_crop = False) #check == True --> croped_list, check==False ---> img_rectangle

       