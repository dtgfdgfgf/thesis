'''
Code for wearing registered clothing items on SMPL models.
Set the "path" variable in the code to the path to the downloaded multi-clothing dataset.

If you use this code, please cite the following paper:
Bharat, Lang, et al. "Multi-Garment Net: Learning to Dress 3D People from Images." Proceedings of the 2019 IEEE International Conference on Computer Vision (ICCV).

Code author:Bharat
Special thanks to Chaitanya for providing the intersection removal code (used to process and eliminate mutual penetration between clothing and body models in 3D models).

This code comes from: https://github.com/bharat-b7/MultiGarmentNetwork
'''

from psbody.mesh import Mesh, MeshViewers
import numpy as np
import pickle as pkl  # Python 3 change
from mgn_utils.smpl_paths import SmplPaths
from lib.ch_smpl import Smpl
from mgn_utils.interpenetration_ind import remove_interpenetration_fast
from os.path import join, split
from glob import glob

from scipy.spatial.transform import Rotation as R
from psbody.mesh.meshviewer import MeshViewer
import os
import cv2

def load_smpl_from_file(file):
    dat = pkl.load(open(file,"rb"),encoding='iso-8859-1')
    dp = SmplPaths(gender=dat['gender'])
    smpl_h = Smpl(dp.get_hres_smpl_model_data())

    smpl_h.pose[:] = dat['pose']
    smpl_h.betas[:] = dat['betas']
    smpl_h.trans[:] = dat['trans']

    return smpl_h


def pose_garment(garment, vert_indices, smpl_params):

    #param smpl_params: dict with pose, betas, v_template, trans, gender
    dp = SmplPaths(gender=smpl_params['gender'])
    smpl = Smpl(dp.get_hres_smpl_model_data())
    smpl.pose[:] = 0 #all
    smpl.betas[:] = smpl_params['betas']
    # smpl.v_template[:] = smpl_params['v_template']

    offsets = np.zeros_like(smpl.r)
    offsets[vert_indices] = garment.v - smpl.r[vert_indices]
    smpl.v_personal[:] = offsets
    smpl.pose[:] = smpl_params['pose']
    smpl.trans[:] = smpl_params['trans']

    mesh = Mesh(smpl.r, smpl.f).keep_vertices(vert_indices)
    return mesh


def retarget(garment_mesh, src, tgt):

    #For each vertex finds the closest point and return    
    verts, _ = src.closest_vertices(garment_mesh.v)
    verts = np.array(verts)
    tgt_garment = garment_mesh.v - src.v[verts] + tgt.v[verts]
    return Mesh(tgt_garment, garment_mesh.f)


def dress(smpl_tgt, body_src, garment, vert_inds, garment_tex=None):
    '''
    :param smpl: SMPL in the output pose
    :param garment: garment mesh in t-pose
    :param body_src: garment body in t-pose
    :param garment_tex: texture file
    :param vert_inds: vertex association b/w smpl and garment
    :return:
    To use texture files, garments must have vt, ft
    '''
    tgt_params = {'pose': np.array(smpl_tgt.pose.r), 
                  'trans': np.array(smpl_tgt.trans.r), 
                  'betas': np.array(smpl_tgt.betas.r), 
                  'gender': 'neutral'
                 }
    smpl_tgt.pose[:] = 0
    body_tgt = Mesh(smpl_tgt.r, smpl_tgt.f)

    # Re-target
    ret = retarget(garment, body_src, body_tgt)

    # Re-pose
    ret_posed = pose_garment(ret, vert_inds, tgt_params)
    body_tgt_posed = pose_garment(body_tgt, range(len(body_tgt.v)), tgt_params)

    # Remove intersections
    ret_posed_interp = remove_interpenetration_fast(ret_posed, body_tgt_posed)
    ret_posed_interp.vt = garment.vt
    ret_posed_interp.ft = garment.ft
    ret_posed_interp.set_texture_image(garment_tex)

    return ret_posed_interp

def process_garment(garment_type, frame_num, smpl_load, path):
    """
    Process a specific garment type and return the dressed mesh.
    
    :param garment_type: Type of garment to process
    :param frame_num: Frame number for processing
    :param smpl_load: Loaded SMPL model parameters
    :param path: Path to the garment dataset
    """
    dp = SmplPaths()
    smpl = Smpl(dp.get_hres_smpl_model_data())
    rot_matrix = smpl_load['full_pose'].detach().cpu().numpy()[0]
    pose_output = R.from_matrix(rot_matrix).as_rotvec().flatten()
    
    smpl.pose = pose_output
    smpl.betas = smpl_load['betas'].detach().cpu().numpy()[0]
    smpl.trans[:] = 0

    body_unposed = load_smpl_from_file(join(path, 'registration.pkl'))
    body_unposed.pose[:] = 0
    body_unposed.trans[:] = 0
    body_unposed_mesh = Mesh(body_unposed.v, body_unposed.f)

    garment_unposed = Mesh(filename=join(path, garment_type + '.obj'))
    vert_indices, _ = pkl.load(open(VERT_INDICES_FILE, "rb"), encoding='iso-8859-1')
    garment_vert_inds = vert_indices[garment_type]
    garment_tex = join(path, 'multi_tex.jpg')
    garment_unposed.set_texture_image(garment_tex)

    garment_dressed = dress(smpl, body_unposed_mesh, garment_unposed, garment_vert_inds, garment_tex)
    return garment_dressed

def create_video_from_snapshots(folder, frame_num):
    #Create a video from snapshot images.
    video_path = TRANSFER_PATH + "video/%d.avi" % folder
    fps = 30
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_out = cv2.VideoWriter('%s' % video_path, fourcc, fps, (1280, 720))

    for i in range(1, frame_num):
        img_path = TRANSFER_PATH + '%02d/snapshot/test%d.jpg' % (folder, i)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            video_out.write(img)

    video_out.release()
    
TRANSFER_PATH = 'C:/Users/USER/python_proj/thesis/demo/transfer/'
DATASET_PATH = 'C:/Users/User/dress_hd4wh/demo/Multi-Garment_dataset/'
GARMENT_CLASSES = ['Pants', 'ShortPants', 'ShirtNoCoat', 'TShirtNoCoat', 'LongCoat']
VERT_INDICES_FILE = 'assets-win/garment_fts_unix.pkl'
all_scans = glob(DATASET_PATH + '*')

gar_dict = {}
for gar in GARMENT_CLASSES:
    gar_dict[gar] = glob(join(DATASET_PATH, '*', gar + '.obj'))

def main():
    for folder in range(0, 1):
        frame_num = 1
        pkl_path = TRANSFER_PATH + "%02d/output" % folder

        while True:
            if os.path.isfile(TRANSFER_PATH + '%02d/snapshot/test%d.jpg' % (folder, frame_num)):
                frame_num += 1
                continue

            try:
                smpl_load = pkl.load(open("%s/obj%d.pkl" % (pkl_path, frame_num), "rb"), encoding='iso-8859-1')

                garment_pants = process_garment('Pants', frame_num, smpl_load,
                                                DATASET_PATH +'125611498005893/')

                garment_tshirt = process_garment('TShirtNoCoat', frame_num, smpl_load,
                                                 DATASET_PATH +'125611494287978/')

                smpl = Smpl(SmplPaths().get_hres_smpl_model_data())
                smpl.pose = R.from_matrix(smpl_load['full_pose'].detach().cpu().numpy()[0]).as_rotvec().flatten()
                smpl.betas = smpl_load['betas'].detach().cpu().numpy()[0]
                smpl.trans[:] = 0

                tgt_body = Mesh(smpl.r, smpl.f)
                save_path = TRANSFER_PATH + "%02d/snapshot/test%d.jpg" % (folder, frame_num)
                mvs = MeshViewers(shape=[1, 1])
                mvs[0][0].set_static_meshes([tgt_body, garment_tshirt, garment_pants])
                mvs[0][0].save_snapshot(save_path)

                print('Frame_%d Done' % frame_num)
                frame_num += 1

                if not os.path.isfile("%s/obj%d.pkl" % (pkl_path, frame_num)):
                    print("All Done")
                    break
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                break
            
        create_video_from_snapshots(folder, frame_num)
        
if __name__ == '__main__':
    main()














