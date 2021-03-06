#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:15:13 2020

@author: abel
"""


import time

import cv2
import numpy as np

import fusion


if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 239
  base_dir='/home/ashfaquekp/output_table'
  mask_dir='/home/ashfaquekp/output_table/mask/'
  cam_intr = np.loadtxt("data_table/camera-intrinsics.txt", delimiter=' ')
  print "Camera Intrinsics",cam_intr
  cam_poses=np.loadtxt("data_table/gt_poses_new.txt")
  #print "Cam poses shape",cam_poses.shape
  vol_bnds = np.zeros((3,2))
  
  
  file=open('data_table/associate.txt')
  lines = file.read().split("\n")
  print "Number of lines in associate",len(lines)
  for i in range(len(lines)-1):
    # Read depth image and camera pose
    depth_file=base_dir+'/'+lines[i].split(" ")[2]
    depth_im = cv2.imread(depth_file,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth_im=depth_im.astype(np.float)
    depth_im /= 5000.  # depth is saved in 16-bit PNG in millimeters
    cam_pose=cam_poses[4*i:4*(i+1),:]


    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
    break
  print "Volume Bounds:",vol_bnds
  file.close()
  # ======================================================================================================== #
  	
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.005)


  t0_elapse = time.time()
  class_id=29 
  file=open('data_table/associate.txt')
  lines = file.read().split("\n")
  for i in range(len(lines)-1):
    rgb_file=base_dir+'/'+lines[i].split(" ")[1]
    depth_file=base_dir+'/'+lines[i].split(" ")[2]
    print("Fusing frame %d/%d"%(i+1, n_imgs))
    img_no=lines[i].split(" ")[1].split("_")[1][:4]
    flag=False
    final_mask=np.zeros((480,640,3),dtype=np.bool)
    mask_index_file=open(mask_dir+img_no+'.txt')
    mask_index_file_lines = mask_index_file.read().split("\n")
    for j in range(len(mask_index_file_lines)-1):
        contents=mask_index_file_lines[j].split(" ")
        if int(contents[0])!=class_id:
            continue
        flag=True
        mfile=contents[1]
        mask=cv2.imread(mfile)
        mask=(mask-mask.min())/(mask.max()-mask.min())
        mask=mask.astype(np.bool)
        final_mask=np.logical_or(mask,final_mask)
        # Read RGB-D image and camera pose
    mask_index_file.close()
    if flag:
        final_mask=final_mask.astype(np.float)
        color_image = cv2.cvtColor(cv2.imread(rgb_file),cv2.COLOR_BGR2RGB)
        color_image=(color_image*final_mask).astype(np.uint8)
        depth_im = cv2.imread(depth_file,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_im=(depth_im*final_mask[:,:,0]).astype(np.float)
        depth_im /= 5000.
          #depth_im[depth_im == 65.535] = 0
        img_no_int=int(img_no)
        cam_pose=cam_poses[4*img_no_int:4*(img_no_int+1),:]
    
    
          # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
     
  file.close()
  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

# Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("suitcase-mesh.ply", verts, faces, norms, colors)

# Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("suitcase-pcd.ply", point_cloud)