
import os
import sys

import numpy as np
sys.path.append('../third_party/contact_graspnet_pytorch')
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils

from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image
from contact_graspnet_pytorch.checkpoints import CheckpointIO 
from contact_graspnet_pytorch.data import load_available_input_data

CGN_PATH = f'../third_party/contact_graspnet_pytorch'


class ContactGraspNet(object):
    def __init__(self, root_dir, ckpt_dir, forward_passes):
        global_config = config_utils.load_config(ckpt_dir, batch_size=forward_passes, arg_configs=[])
        grasp_estimator = GraspEstimator(global_config)

        model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
        checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
        try:
            load_dict = checkpoint_io.load('model.pt')
        except FileExistsError:
            print('No model checkpoint found')
            load_dict = {}

        self.grasp_estimator = grasp_estimator
        self.root_path = root_dir
        self.forward_passes = forward_passes
        self.init_inference()

    def init_inference(self):
        # initialization with an example
        segmap, rgb, depth, cam_K, _, _ = load_available_input_data(f'{self.root_path}/test_data/0.npy', K=None)
        self.single_depth_inference(rgb, depth, segmap, cam_K, local_regions=True,
                                    skip_border_objects=True, filter_grasps=True,
                                    z_range=[0.5, 2.5], forward_passes=self.forward_passes,
                                    visualization=False)

    def single_depth_inference(self, rgb, depth, segmap, cam_K, local_regions=False,
                               skip_border_objects=False, filter_grasps=False, z_range=[0.2, 2.5],
                               forward_passes=1, visualization=False):
        if segmap is None:
            segmap = np.zeros_like(depth)

        print('Converting depth to point cloud(s)...')
        pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects,
                                                                                    z_range=z_range)

        print('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _, _ = self.grasp_estimator.predict_scene_grasps(pc_full,
                                                                                            pc_segments=pc_segments,
                                                                                            local_regions=local_regions,
                                                                                            filter_grasps=filter_grasps,
                                                                                            forward_passes=forward_passes,
                                                                                            convert_cam_coords=True)

        # Visualize results
        if visualization:
            # show_image(rgb, segmap)
            # scene = trimesh.Scene()
            # axis = trimesh.creation.axis()
            # scene.add_geometry(axis)
            # for k, v in input_pts.items():
            #     pc = trimesh.points.PointCloud(v, colors=np.array([[0, 0, 200, 100]]).repeat(v.shape[0], 0))
            #     scene.add_geometry(pc)
            # scene.show()
            visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

        return pred_grasps_cam, scores, contact_pts, pc_full, pc_colors


    def single_ptd_inference(self, point_clouds, local_regions=False, filter_grasps=False, forward_passes=1):
        res, logs = [], []
        for ptc in point_clouds:
            pc_full = ptc['pc_full']
            pc_segments = {1: ptc['pc_obj']}

            if len(ptc['pc_obj']) < 1:
                res.append({
                    'grasp_poses': {1: []},
                    'scores': {1: []},
                    'contact': {1: []},
                    'pc_full': {1: pc_full},
                    'pc_input': {1: []},
                    'cam_idx': {1: ptc['cam_idx']},
                    'cam_pose': {1: ptc['cam_pose']}
                })
                continue

            pred_grasps_cam, scores, contact_pts, _, input_pts = (
                self.grasp_estimator.predict_scene_grasps(pc_full, pc_segments=pc_segments,
                                                          local_regions=local_regions,
                                                          filter_grasps=filter_grasps,
                                                          forward_passes=forward_passes,
                                                          convert_cam_coords=False))

            res.append({
                'grasp_poses': pred_grasps_cam,
                'scores': scores,
                'contact': contact_pts,
                'pc_full': pc_full,
                'pc_input': input_pts,
                'cam_idx': ptc['cam_idx'],
                'cam_pose': ptc['cam_pose']
            })

        return res


if __name__ == '__main__':
    net = ContactGraspNet('./checkpoints/contact_graspnet', 1)