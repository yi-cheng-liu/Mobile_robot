import numpy as np
import matplotlib.pyplot as plt
import gtsam
import pdb


def readG2OFile(file_name):
    with open(file_name, "r") as file:
        poses = []
        edges = []
        for line in file:
            values = line.strip().split(" ")
            # 1. poses
            if values[0] == 'VERTEX_SE2':
                poses_data = np.array(values[1:]).astype(float)
                poses.append(poses_data)
            # 2. edges
            elif values[0] == 'EDGE_SE2':
                edges_data = np.array(values[1:]).astype(float)
                edges.append(edges_data)
        poses = np.array(poses)
        edges = np.array(edges)
    return poses, edges


def incremental_solution_2d(poses, edges):
    # 1. Initialize iSAM
    parameters = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(parameters)
    print("finish initiliazing")

    # 2. incremental solution
    for i in range(len(poses)):
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        if poses[i, 0] == 0:
            prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
            x = poses[i, 1]
            y = poses[i, 2]
            theta = poses[i, 3]
            graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(x, y, theta),
                                             prior_noise))
            initial_estimate.insert(int(poses[i, 0]), gtsam.Pose2(x, y, theta))
        else:
            prev_pose = result[i-1]
            initial_estimate.insert(poses[i, 0], prev_pose)
            for j in range(len(edges)):
                idx_e1 = edges[j, 0]
                idx_e2 = edges[j, 1]
                dx = edges[j, 2]
                dy = edges[j, 3]
                dtheta = edges[j, 4]
                info = edges[j, 5:]
                if edges[j, 1] == poses[i, 0]:
                    info_matrix = np.array([[info[0], info[1], info[2]],
                                            [info[1], info[3], info[4]],
                                            [info[2], info[4], info[5]]])
                    cov = gtsam.construct_covariance(
                        np.linalg.inv(info_matrix))
                    model = gtsam.noiseModel.Gaussian.Covariance(cov)
                    graph.add(gtsam.PriorFactorPose2(idx_e1, idx_e2,
                              gtsam.Pose2(dx, dy, dtheta), model))
        print('ok')
        print(graph)
        print('ok')
        isam.update(graph, initial_estimate)
        print('ok')
        result = isam.calculateEstimate()
        # prevPose = result.atPose2(i − 1)


if __name__ == "__main__":
    # 1. read the files
    file = "input_INTEL_g2o.g2o"
    poses, edges = readG2OFile(file)

    # 2. batch solution

    # 3. incremental solution
    incremental_solution_2d(poses, edges)
