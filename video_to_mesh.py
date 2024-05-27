import sys
import os
import os.path as osp
import gc
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from OpenGL import platform, _configflags

# 插入所需的路徑
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
sys.path.insert(0, osp.join('..', 'ttest'))
from config import cfg
from get_yolov5 import get_yolo
from model import get_model
from local_utils.preprocessing import load_img, process_bbox, generate_patch_image
from local_utils.human_models import smpl_x
from local_utils.vis import render_mesh, save_obj
import json
import logging

# 設置日誌
white_background = load_img('white_background.jpg')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    args = parser.parse_args()

    if not args.gpu_ids:
        logging.error("請設置正確的GPU ID")
        sys.exit(1)

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

def load_model(model_path, model_name='test'):
    if not osp.exists(model_path):
        logging.error(f'找不到模型文件：{model_path}')
        sys.exit(1)
    
    logging.info(f'從 {model_path} 加載檢查點')
    model = get_model(model_name)
    model = DataParallel(model).cuda()
    try:
        ckpt = torch.load(model_path)
    except Exception as e:
        logging.error(f'加載模型檢查點失敗: {e}')
        sys.exit(1)
    
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()
    return model

def prepare_yolov5():
    try:
        yolov5_model = get_yolo()
    except Exception as e:
        logging.error(f'加載YOLOv5模型失敗: {e}')
        sys.exit(1)
    return yolov5_model

def process_video(video_path, model, yolov5_model, output_path):
    if not os.path.exists(video_path):
        logging.warning(f'影片文件不存在：{video_path}')
        return

    videoInput = cv2.VideoCapture(video_path)
    if not videoInput.isOpened():
        logging.warning(f'無法打開影片：{video_path}')
        return

    cnt = 0
    transform = transforms.ToTensor()
    frame_height, frame_width = None, None
    
    while videoInput.isOpened():
        success, frame = videoInput.read()
        if not success:
            break
        
        if frame_height is None or frame_width is None:
            frame_height, frame_width = frame.shape[:2]
        
        results = yolov5_model(frame)
        if len(results.pandas().xyxy[0]['name']) == 0:
            continue
        
        for i in range(len(results.pandas().xyxy[0]['name'])):
            if results.pandas().xyxy[0]['name'][i] == 'person':
                xmin = results.pandas().xyxy[0]['xmin'][i]
                xmax = results.pandas().xyxy[0]['xmax'][i]
                ymin = results.pandas().xyxy[0]['ymin'][i]
                ymax = results.pandas().xyxy[0]['ymax'][i]
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                break
        else:
            continue

        bbox = process_bbox(bbox, frame_width, frame_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(frame, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transform(img.astype(np.float32)) / 255
        img = img.cuda()[None, :, :, :]

        inputs = {'img': img}
        targets = {}
        meta_info = {}

        try:
            with torch.no_grad():
                out = model(inputs, targets, meta_info, 'test')
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
        except Exception as e:
            logging.error(f'3D模型生成失敗: {e}')
            continue

        focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        rendered_img = render_mesh(white_background, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}).astype('uint8')
        
        cv2.imwrite(f'{output_path}/render_original_img{cnt}.jpg', rendered_img)
        cnt += 1
        gc.collect()
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

    videoInput.release()

def create_output_video(render_img_path, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_out = cv2.VideoWriter(output_video_path, fourcc, fps, (1280, 720))

    for i in range(1000):  # 假設最大幀數為1000，可根據需要調整
        img_path = f'{render_img_path}/render_original_img{i}.jpg'
        if not os.path.isfile(img_path):
            break
        rendered_img = load_img(img_path).astype('uint8')
        video_out.write(rendered_img)

    video_out.release()

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    
    if not torch.cuda.is_available():
        logging.error("CUDA 不可用，請檢查 GPU 配置")
        sys.exit(1)
    else:
        logging.info(f"使用 GPU: {args.gpu_ids}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"可用的 GPU: {torch.cuda.get_device_name(i)}")

    yolov5_model = prepare_yolov5()
    model = load_model('./snapshot_6.pth.tar')

    white_background = load_img('white_background.jpg')
           
    for num in range(0, 100):
        output_video_path = f'C:/Users/USER/python_proj/thesis/demo/demo_video/Mesh_{num:03d}.avi'
        os.makedirs(output_video_path, exist_ok=True)
        
        if os.path.isfile(output_video_path):
            logging.info('影片已存在')
            continue
        
        render_img_path = f"C:/Users/User/python_proj/thesis/demo/render_img/Mesh_{num:03d}"
        os.makedirs(render_img_path, exist_ok=True)
       
        video_path = f"C:/Users/USER/python_proj/thesis/demo/ori_video/{num:03d}.avi"
        process_video(video_path, model, yolov5_model, render_img_path)
        
        fps = 30  # 預設值，可根據需要進行調整
        create_output_video(render_img_path, output_video_path, fps)
        logging.info(f'Mesh{num} done')

if __name__ == "__main__":
    main()
