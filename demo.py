"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
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
  n_imgs = 199
  base_dir='/home/ashfaquekp/output'
  
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  print "Camera Intrinsics",cam_intr
  cam_poses=np.loadtxt("data/gt_poses.txt")
  #print "Cam poses shape",cam_poses.shape
  vol_bnds = np.zeros((3,2))
  
  
  file=open('data/associate.txt')
  lines = file.read().split("/n")
  for i in range(len(lines)-1):
    # Read depth image and camera pose
    depth_file=base_dir+'/'+lines[i].split(" ")[2]
    print "Depth File",depth_file
    depth_im = cv2.imread(depth_file,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth_im=depth_im.astype(np.float)
    depth_im /= 5000.  # depth is saved in 16-bit PNG in millimeters
    
    print "Depth Shape",depth_im.shape
    print "Depth Max",np.max(depth_im)
    print "Depth Min",np.min(depth_im)
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
#   print("Initializing voxel volume...")
#   tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)


#   t0_elapse = time.time()
 
#   file=open('data/associate.txt')
#   lines = file.read().split("/n")
#   for i in range(len(lines)-1):
#     rgb_file=base_dir+'/'+lines[i].split(" ")[1]
#     depth_file=base_dir+'/'+lines[i].split(" ")[2]
#     print("Fusing frame %d/%d"%(i+1, n_imgs))

#      # Read RGB-D image and camera pose
#     color_image = cv2.cvtColor(cv2.imread(rgb_file, cv2.COLOR_BGR2RGB)
#     depth_im = cv2.imread(depth_file,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#     depth_im=depth_im.astype(np.float)
#     depth_im /= 5000.
#      #depth_im[depth_im == 65.535] = 0
#     cam_pose=cam_poses[4*i:4*(i+1),:]

#      # Integrate observation into voxel volume (assume color aligned with depth)
#     tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
     
#   file.close()
#   fps = n_imgs / (time.time() - t0_elapse)
#   print("Average FPS: {:.2f}".format(fps))

# # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
#   print("Saving mesh to mesh.ply...")
#   verts, faces, norms, colors = tsdf_vol.get_mesh()
#   fusion.meshwrite("mesh-blender.ply", verts, faces, norms, colors)

# # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
#   print("Saving point cloud to pc.ply...")
#   point_cloud = tsdf_vol.get_point_cloud()
#   fusion.pcwrite("pc-blender.ply", point_cloud)