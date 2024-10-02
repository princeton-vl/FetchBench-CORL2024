"""
The MIT License (MIT)

Copyright (c) 2020 NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import trimesh
import trimesh.transformations as tra
import numpy as np
import shapely
from copy import deepcopy
from isaacgymenvs.tasks.fetch.utils.load_utils import sample_random_objects, sample_random_scene


def apply_T(pts, T):
    pts = np.concatenate([pts, [1.]]).reshape(-1, 1)
    pts = (T @ pts).reshape(-1)[:-1]
    return pts


class Surface(object):
    def __init__(self, polygon, translation, z, label):
        self.polygon = polygon
        self.translation = translation
        self.z = z
        self.label = label

    def __eq__(self, other):
        if ((self.polygon.equals_exact(other.polygon, tolerance=0.001) and
            (self.translation == other.translation).all()) and
                (self.z == other.z) and (self.label == other.label)):
            return True
        else:
            return False

    def to_dict(self):
        log = {
            'polygon': shapely.to_geojson(self.polygon),
            'translation': self.translation.tolist(),
            'z': self.z,
            'label': self.label,
        }
        return log


class SupportSurface(object):
    def __init__(self, max_z=10.):
        self.surfaces = []
        self._max_z = max_z

    def __len__(self):
        return len(self.surfaces)

    def polygon_buffer_dist(self, support, dist):
        polygons = support.polygon.buffer(dist)
        if polygons.is_empty:
            return []
        if polygons.geom_type == 'MultiPolygon':
            buffer_supports = []
            for p in polygons.geoms:
                buffer_supports.append((p, support))
            return polygons

        return [(polygons, support)]

    def sample_point3d_uniform(self, label=None, buffer_dist=0.):
        supports = []
        for s in self.surfaces:
            if label is not None and not (s.label == label):
                continue
            buffered = self.polygon_buffer_dist(s, buffer_dist)
            supports.extend(buffered)

        if len(supports) == 0:
            return None, None

        areas = np.array([p[0].area for p in supports])
        selected_idx = np.random.choice(range(len(supports)), p=areas / areas.sum())
        selected = supports[selected_idx]

        pts = trimesh.path.polygons.sample(selected[0], count=1)
        pts3d = np.append(pts, 0) + selected[1].translation

        if selected[1].label == 'on_wall':
            pts3d = (
                apply_T(pts3d,
                        np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32))
            )

        return pts3d, selected[1]

    def load_supports(self, logs):
        for log in logs:
            polygon = shapely.from_geojson(log['polygon'])
            z = np.minimum(log['z'], self._max_z)
            translation = np.array(log['translation'])
            label = log['label']
            self.append(polygon, translation, z, label)

    def get_all_support_types(self):
        labels = []
        for s in self.surfaces:
            if s.label in labels:
                continue
            elif s.label in ['on_table', 'on_shelf', 'in_basket', 'in_drawer']:
                labels.append(s.label)
        return labels

    def get_all_hanging_types(self):
        labels = []
        for s in self.surfaces:
            if s.label in labels:
                continue
            elif s.label in ['on_wall']:
                labels.append(s.label)
        return labels

    def append(self, polygon, trans, z, label):
        self.surfaces.append(Surface(polygon, trans, z, label))


class TrimeshRearrangeScene(object):
    def __init__(self, support_meshes, support_annotations, support_robot_cam_config, spacing=2.5, **kwargs):
        """Create a scene object."""
        self._objects = {}
        self._supports = {}
        self._scene_spacing = spacing
        self._scene_bounds = None
        self._support_surface = SupportSurface()

        self._max_num_spawned_objs = kwargs.get('max_num_objs', 15)
        self._distance_above_support = kwargs.get('dist_above_support', 0.005)  # distance at which objects will be sampled
        self._erosion_ratio = kwargs.get('buffer_ratio', 0.2)  # distance to the edge of the polygon
        self._label_prob = kwargs.get('label_probs', {'in_basket': 0.5, 'on_shelf': 0.3, 'on_table': 0.2,
                                                      'in_drawer': 0.3, 'on_wall': 1.0})

        self.collision_manager = trimesh.collision.CollisionManager()

        self.add_support(support_meshes)
        self._support_surface.load_supports(support_annotations)
        self._support_robot_cam_config = support_robot_cam_config

        self.table_dim = kwargs.get('table_dim', None)
        if self.table_dim is None:
            self.table_dim = self.sample_table_dim()

    def add_support(self, support_meshes):
        scene = trimesh.Scene()
        for i, geo in enumerate(support_meshes):
            self._supports[f'support_{i}'] = {'mesh': geo, 'pose': np.eye(4)}
            scene.add_geometry(geo)
            self.collision_manager.add_object(name=f'support_{i}', mesh=geo, transform=np.eye(4))

        self._scene_bounds = scene.bounds

    def add_object(self, obj_id, obj_name, obj_mesh, pose):
        self._objects[obj_id] = {'name': obj_name, 'mesh': obj_mesh, 'pose': pose}
        self.collision_manager.add_object(name=obj_id, mesh=obj_mesh, transform=pose)

    def find_object_placement(self, i, obj_mesh, obj_poses, max_iter=1):
        iter, valid, placement_T = 0, False, None

        sample_label = None
        all_label_types = self._support_surface.get_all_support_types()

        while iter < max_iter and not valid:
            label_probs = np.array([self._label_prob[l] for l in all_label_types])
            sample_label = np.random.choice(all_label_types, p=label_probs / label_probs.sum())

            dim = np.max(obj_mesh.bounding_box_oriented.extents)
            pts3d, _ = self._support_surface.sample_point3d_uniform(label=sample_label,
                                                                    buffer_dist=-self._erosion_ratio * dim)
            pts3d += np.array([0., 0., self._distance_above_support])

            if pts3d is None:
                iter += 1
                continue

            placement_T = tra.translation_matrix(pts3d)
            pose = self.sample_random_stable_pose(obj_poses)

            bb = obj_mesh.copy().apply_transform(pose).bounding_box
            colliding = self.in_collision_with(bb, placement_T, min_distance=self._distance_above_support)

            valid = not colliding
            iter += 1

            placement_T = np.dot(placement_T, pose)

        return valid, placement_T, sample_label

    def find_combo_placement(self, i, obj_mesh, obj_poses, in_plane_rot=False,
                             placement_type='support', max_iter=1):
        iter, valid, placement_T = 0, False, None

        sample_label = None
        if placement_type == 'support':
            all_label_types = self._support_surface.get_all_support_types()
            if 'in_basket' in all_label_types:
                all_label_types.remove('in_basket')
            if 'in_drawer' in all_label_types:
                all_label_types.remove('in_drawer')
        elif placement_type == 'hanging':
            all_label_types = self._support_surface.get_all_hanging_types()
        else:
            raise NotImplementedError

        if len(all_label_types) == 0:
            return False, None, None

        while iter < max_iter and not valid:
            label_probs = np.array([self._label_prob[l] for l in all_label_types])
            sample_label = np.random.choice(all_label_types, p=label_probs / label_probs.sum())

            dim = np.max(obj_mesh.bounding_box_oriented.extents)
            pts3d, _ = self._support_surface.sample_point3d_uniform(label=sample_label,
                                                                    buffer_dist=-self._erosion_ratio * dim)

            if pts3d is None:
                iter += 1
                continue

            if placement_type == 'support':
                pts3d += np.array([0., 0., self._distance_above_support])
            if placement_type == 'hanging':
                pts3d += np.array([self._distance_above_support, 0., 0.])

            placement_T = tra.translation_matrix(pts3d)
            if in_plane_rot:
                pose = self.sample_random_stable_pose(obj_poses)
            else:
                pose = obj_poses[0]

            bb = obj_mesh.copy().apply_transform(pose).bounding_box
            colliding = self.in_collision_with(bb, placement_T, min_distance=self._distance_above_support)

            valid = not colliding
            iter += 1

            placement_T = np.dot(placement_T, pose)

        return valid, placement_T, sample_label

    def discard_object_placement(self, i, obj_poses):
        # create enough space for remaining objects
        assert self._scene_spacing * 2 >= 5.
        assert self._scene_bounds[0][0] + self._scene_spacing >= 1.

        dy = 2 * self._scene_spacing
        ys = np.arange(-self._scene_spacing + 0.2, self._scene_spacing - 0.2, dy / (self._max_num_spawned_objs+1))
        x = - self._scene_spacing + 0.25
        translation = tra.translation_matrix(np.array([x, ys[i], self._distance_above_support]))
        placement_T = np.dot(translation, obj_poses[0])
        sample_label = 'on_floor'

        return True, placement_T, sample_label

    def sample_random_stable_pose(self, stable_poses):
        pose = stable_poses[np.random.randint(len(stable_poses))]
        inplane_rot = tra.rotation_matrix(angle=np.random.uniform(0, 2.0 * np.pi), direction=[0, 0, 1])
        return inplane_rot.dot(pose)

    def in_collision_with(self, mesh, transform, min_distance=0.0, epsilon=1.0 / 1e3):
        colliding = self.collision_manager.in_collision_single(mesh=mesh, transform=transform)
        if not colliding and min_distance > 0.0:
            distance = self.collision_manager.min_distance_single(mesh=mesh, transform=transform)
            if distance < min_distance - epsilon:
                colliding = True

        return colliding

    def random_arrangement(self, objects, combo_objects, num_obj_discarded=0, num_combo_discarded=0):
        # clear objects
        assert len(objects) + 2 * len(combo_objects) <= self._max_num_spawned_objs
        assert num_obj_discarded <= len(objects) and num_combo_discarded <= len(combo_objects)
        self.remove_objects()

        # deepcopy
        n_objects, n_combos = [], []

        discarded_indices = np.random.choice(range(len(combo_objects)), size=(num_combo_discarded,), replace=False)

        # try to add all combo objects to the scene
        for j, combo in enumerate(combo_objects):
            n_cob = {}
            for k, v in combo.items():
                n_cob[k] = v
            obj_meshes = combo['meshes']
            obj_mesh = trimesh.util.concatenate(obj_meshes)
            obj_name = combo['name']
            obj_poses = combo['stable_poses']
            placement_type = combo['placement_type']
            in_plane_rot = True if placement_type == 'support' else False  # Todo: More case-by-case if else.

            success = False
            if j not in discarded_indices:
                success, placement_T, label = self.find_combo_placement(j, obj_mesh, obj_poses,
                                                                        placement_type=placement_type,
                                                                        in_plane_rot=in_plane_rot,
                                                                        max_iter=100)

            if not success:
                success, placement_T, label = self.discard_object_placement(j, obj_poses)
                print("Couldn't place object", f'obj_combo_{obj_name}_{j}', "!")

            self.add_object(f'obj_combo_{obj_name}_{j}', obj_name, obj_mesh, placement_T)
            n_cob['placement_pose'] = placement_T.copy()
            n_cob['placement_label'] = label
            n_combos.append(n_cob)

        # sample discarded object on_floor
        discarded_indices = np.random.choice(range(len(objects)), size=(num_obj_discarded,), replace=False)

        for i, obj in enumerate(objects):
            n_obj = {}
            for k, v in obj.items():
                n_obj[k] = v
            obj_mesh = obj['mesh']
            obj_name = obj['name']
            obj_poses = obj['stable_poses']

            success = False
            if i not in discarded_indices:
                success, placement_T, label = self.find_object_placement(i, obj_mesh, obj_poses, max_iter=100)

            if not success:
                success, placement_T, label = self.discard_object_placement(i + len(combo_objects), obj_poses)
                print("Couldn't place object", f'obj_{obj_name}_{i}', "!")

            self.add_object(f'obj_{obj_name}_{i}', obj_name, obj_mesh, placement_T)
            n_obj['placement_pose'] = placement_T.copy()
            n_obj['placement_label'] = label
            n_objects.append(n_obj)

        return n_combos, n_objects

    def remove_objects(self):
        raw_objects = self._objects.copy()
        obj_ids = list(self._objects.keys())
        for obj_id in obj_ids:
            self.collision_manager.remove_object(obj_id)
        self._objects = {}
        return raw_objects

    def sample_table_dim(self):
        z = (self._support_robot_cam_config['support_bounds'][0][2] +
             self._support_robot_cam_config['robot_base_offset'][2][0])
        x = np.random.uniform(0.4, 0.7)
        y = np.random.uniform(0.6, 1.0)
        return [x, y, z]

    def sample_robot_base(self):
        # ori of the robot is towards the -x dir
        support_bounds = self._support_robot_cam_config['support_bounds']
        robot_base_offset = self._support_robot_cam_config['robot_base_offset']
        z_mean = support_bounds[0][2]
        robot_z = np.random.uniform(z_mean + robot_base_offset[2][0], z_mean + robot_base_offset[2][1])

        y_mean = (support_bounds[0][1] + support_bounds[1][1]) / 2.
        robot_y = np.random.uniform(y_mean + robot_base_offset[1][0], y_mean + robot_base_offset[1][1])

        x_mean = support_bounds[1][0]
        robot_x = np.random.uniform(x_mean + robot_base_offset[0][0], x_mean + robot_base_offset[0][1])

        # sample base table dim and pos
        base_x_front = np.random.uniform(robot_base_offset[0][0] - 0.08, robot_base_offset[0][0])
        base_y_left = np.random.uniform(0.2, 0.4)
        base_z_top = robot_z - 0.002

        table_x = robot_x + self.table_dim[0] / 2. - base_x_front
        table_y = robot_y + self.table_dim[1] / 2. - base_y_left
        table_z = base_z_top - self.table_dim[2] / 2.

        return [robot_x, robot_y, robot_z], [table_x, table_y, table_z]

    def sample_camera_pose(self, i=0):
        assert i <= 1, "Only support two cams, left & right"

        cam_tar_offset = self._support_robot_cam_config['camera_tar_offset']
        cam_pos_offset = self._support_robot_cam_config['camera_pos_offset']
        support_bounds = self._support_robot_cam_config['support_bounds']

        x_mean = (support_bounds[1][0] + support_bounds[0][0]) / 2.
        y_mean = (support_bounds[1][1] + support_bounds[0][1]) / 2.
        z_mean = (support_bounds[1][2] + support_bounds[0][2]) / 2.

        for _ in range(20):

            x_f = np.random.uniform(x_mean + cam_tar_offset[0][0], x_mean + cam_tar_offset[0][1])
            y_f = np.random.uniform(y_mean + cam_tar_offset[1][0], y_mean + cam_tar_offset[2][1])
            z_f = np.random.uniform(z_mean + cam_tar_offset[2][0], z_mean + cam_tar_offset[2][1])

            x_c = support_bounds[1][0] + np.random.uniform(*cam_pos_offset[0])
            y_c = y_mean + (-1) ** i * (x_c - support_bounds[1][0]) * np.tan(np.random.uniform(*cam_pos_offset[1]))

            z_c = z_f
            z = z_c + np.random.uniform(*cam_pos_offset[2])

            d = np.sqrt((x_c - x_f) ** 2 + (y_c - y_f) ** 2)
            alpha = np.arctan((z - z_f) / d)
            l = (x_c - support_bounds[1][0]) / np.cos(np.arctan((y_c - y_f) / (x_c - x_f)))

            e1 = z - l * np.tan(alpha + (27 / 180) * np.pi)
            e2 = z + l * np.tan(((27 / 180) * np.pi) - alpha)

            if (e1 <= support_bounds[0][2] + 0.03 and e2 >= support_bounds[1][2] - 0.03):
                z_c = z
                break

        cam_pose = {
            'focus': [x_f, y_f, z_f],
            'pos': [x_c, y_c, z_c]
        }

        return cam_pose

    def as_trimesh_scene(self):
        trimesh_scene = trimesh.scene.Scene()
        for obj_id, obj in self._objects.items():
            trimesh_scene.add_geometry(obj['mesh'], node_name=obj_id,
                                       geom_name=obj_id, transform=obj['pose'])

        for obj_id, obj in self._supports.items():
            trimesh_scene.add_geometry(obj['mesh'], node_name=obj_id,
                                       geom_name=obj_id, transform=obj['pose'])

        # Vis bounding region
        bounding_region = trimesh.creation.box(bounds=self._support_robot_cam_config['support_bounds'])
        bounding_region.visual.face_colors = [0, 0, 0, 100]
        trimesh_scene.add_geometry(bounding_region)

        # Add two plane as floor and axis
        plane = trimesh.creation.box((1.5, 1.5, 0.01))
        plane.apply_translation([0, 0, -0.005])
        plane.visual.face_colors = [0, 0, 0, 100]
        axis = trimesh.creation.axis()
        trimesh_scene.add_geometry(plane)
        trimesh_scene.add_geometry(axis)

        return trimesh_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-mode', choices=['bench', 'pose'], default='pose')
    args = parser.parse_args()

    objects = sample_random_objects(5)
    scene = sample_random_scene()

    scene = TrimeshRearrangeScene(scene['meshes'], scene['support'])
    scene.random_arrangement(objects)
    scene.as_trimesh_scene().show()
    scene.random_arrangement(objects)
    scene.as_trimesh_scene().show()

