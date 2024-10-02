
import numpy as np
import os
import torch
import trimesh
import time
import trimesh.transformations as tr

from isaacgymenvs.tasks.fetch.fetch_mesh_pyompl import FetchMeshPyompl
from isaacgymenvs.tasks.fetch.fetch_mesh_curobo import image_to_video


class FetchMeshPyomplRep(FetchMeshPyompl):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        assert self.num_envs == 1  # Buggy Pybullet, only one env at a time
        # Multiple Pybullet Server may cause error

        self.num_max_trials = self.cfg["solution"]["num_max_trials"]

    def motion_plan_to_retract_pose(self):
        target_pos = [[0.13, 0., 0.7]]
        target_quat = [[0., 1., 0., 0.]]

        end_poses = []
        for t in range(len(target_quat)):
            pos = target_pos[t]
            quat = target_quat[t]
            translation = tr.translation_matrix(pos)
            rotation = tr.quaternion_matrix(quat)
            end_poses.append(translation @ rotation)
        end_poses = np.stack(end_poses, axis=0)

        target_poses = []
        for i in range(self.num_envs):
            target_poses.append(end_poses)

        return self.motion_gen_to_pose_goalset(target_poses)

    def solve(self):
        # set goal obj color
        log = {}

        self.set_target_color()
        self._solution_video = []
        self._video_frame = 0
        computing_time = 0.

        for _ in range(self._init_steps):
            self.env_physics_step()
            self.post_phy_step()

        for k in range(self.num_max_trials):

            # Sample Good Grasp Pose
            #start_time = time.time()
            self.update_pyompl_world_collider_pose(attach_goal_obj=False)
            grasp_result = self.sample_goal_obj_collision_free_grasp_pose()
            #computing_time += time.time() - start_time

            if self.cfg["solution"]["direct_grasp"]:

                start_time = time.time()
                traj, success, poses = (
                    self.motion_gen_to_grasp_pose(grasp_result['grasp_poses'], mask=grasp_result['grasp_success'])
                )
                print("Grasp Plan", success)
                log['grasp_plan_success'] = success
                computing_time += time.time() - start_time

                self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement
                print("Grasp Phase End")
                log['grasp_execute_error'] = self.get_end_effect_error(poses)
            else:
                start_time = time.time()
                traj, success, poses = (
                    self.motion_gen_to_grasp_pose(grasp_result['pre_grasp_poses'], mask=grasp_result['grasp_success']))
                print("Pre Grasp Plan", success)
                log['pre_grasp_plan_success'] = success
                computing_time += time.time() - start_time

                self.follow_motion_trajs(traj, gripper_state=0)  # 0 means no movement
                print("Pre Grasp Phase End")
                log['pre_grasp_execute_error'] = self.get_end_effect_error(poses)

                offset = np.array([0, 0, self.cfg["solution"]["pre_grasp_offset"] *
                                   self.cfg["solution"]["grasp_overshoot_ratio"]])
                self.follow_cartesian_linear_motion(offset, gripper_state=0)

            print("Grasp Phase End")

            self.close_gripper()
            log['grasp_finger_obj_contact'] = self.finger_goal_obj_contact()
            print("Gripper Close End")

            if self.cfg["solution"]["retract_offset"] > 0:
                offset = np.array([0, 0, self.cfg["solution"]["retract_offset"]])
                self.follow_cartesian_linear_motion(offset, gripper_state=-1, eef_frame=False)
                log['retract_finger_obj_contact'] = self.finger_goal_obj_contact()

            # Fetch Phase

            self.update_pyompl_world_collider_pose(attach_goal_obj=True)

            start_time = time.time()
            traj, success, poses = self.motion_gen_to_free_space(mask=success)
            print("Fetch Plan", success)
            log['fetch_plan_success'] = success
            computing_time += time.time() - start_time

            self.follow_motion_trajs(traj, gripper_state=-1)  # 0 means no movement
            print("Fetch Phase End")
            log['fetch_execute_error'] = self.get_end_effect_error(poses)

            log['num_repetitive_trials'] = [k]
            if self.eval()['success'][0] or (not self.eval()['task_repeat'][0]):
                break

            self.open_gripper()
            self.update_pyompl_world_collider_pose(attach_goal_obj=False)

            start_time = time.time()
            traj, success, poses = self.motion_plan_to_retract_pose()
            computing_time += time.time() - start_time

            self.follow_motion_trajs(traj, gripper_state=0)
            print("Reset Phase End")

        log['traj_length'] = self._traj_length.cpu().numpy()
        log['computing_time'] = [computing_time / self.num_envs for _ in range(self.num_envs)]

        self.repeat()
        log['end_finger_obj_contact'] = self.finger_goal_obj_contact()
        print("Eval Phase End")
        self.set_default_color()

        return image_to_video(self._solution_video), log