import numpy as np
import gtsam
import matplotlib.pyplot as plt


def readG2OFile3d(file_name):
    with open(file_name, "r") as file:
        poses = []
        edges = []
        for line in file:
            values = line.strip().split(" ")
            # 1. poses
            if values[0] == 'VERTEX_SE3:QUAT':
                poses_data = np.array(values[1:]).astype(float)
                poses.append(poses_data)
            # 2. edges
            elif values[0] == 'EDGE_SE3:QUAT':
                edges_data = np.array(values[1:]).astype(float)
                edges.append(edges_data)
        poses = np.array(poses)
        edges = np.array(edges)
    return poses, edges


def batch_solution_3d(file):
    # 1. read the data
    is3D = True
    graph, initial = gtsam.readG2o(file, is3D)

    # 2. add a prior factor model to the graph
    priorModel = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4]))
    graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), priorModel))

    # 3. build the optimizer with Gauss-Newton method
    params = gtsam.GaussNewtonParams()
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)

    # 4. optimize the graph
    result = optimizer.optimize()

    return result


def incremental_solution_3d(poses, edges):
    # 1. Initialize iSAM
    params = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(params)

    # 2. incremental solution
    for pose in poses:
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        # extract the x, y, z, qx, qy, qz, qw data from the prestored pose
        idp, x, y, z, qx, qy, qz, qw = pose
        if idp == 0:
            # create the prior_noise and add to the graph
            prior_noise = gtsam.noiseModel.Diagonal.Variances(
                np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-4, 1e-4]))
            # calculate the quaternion rotational matrix, and the transitional matrix
            R = gtsam.Rot3(quaternion_rotation_matrix(qw, qx, qy, qz))
            T = np.array([x, y, z]).reshape((3, 1))
            prior = gtsam.Pose3(R, T)
            # add to the graph
            graph.add(gtsam.PriorFactorPose3(0, prior, prior_noise))
            initial_estimate.insert(int(idp), prior)
        else:
            prev_pose = result.atPose3(int(idp - 1))
            initial_estimate.insert(int(idp), prev_pose)
            for edge in edges:
                # calcualte the information matrix
                ide1, ide2, dx, dy, dz, dqx, dqy, dqz, dqw, info_0, info_1, info_2, info_3, info_4, info_5, info_6, info_7, info_8, info_9, info_10, info_11, info_12, info_13, info_14, info_15, info_16, info_17, info_18, info_19, info_20 = edge
                info_matrix = np.array([[info_0,  info_1,  info_2,  info_3,  info_4,  info_5],
                                        [info_1,  info_6,  info_7,
                                            info_8,  info_9, info_10],
                                        [info_2,  info_7, info_11,
                                            info_12, info_13, info_14],
                                        [info_3,  info_8, info_12,
                                            info_15, info_16, info_17],
                                        [info_4,  info_9, info_13,
                                            info_16, info_18, info_19],
                                        [info_5, info_10, info_14, info_17, info_19, info_20]])
                if ide2 == idp:
                    # build the noise model with the information matrix
                    cov = np.linalg.inv(info_matrix)
                    model = gtsam.noiseModel.Gaussian.Covariance(cov)
                    # calculate the current pose
                    dR = gtsam.Rot3(
                        quaternion_rotation_matrix(dqw, dqx, dqy, dqz))
                    dT = np.array([dx, dy, dz]).reshape((3, 1))
                    pose = gtsam.Pose3(dR, dT)
                    # update the graph
                    graph.add(gtsam.BetweenFactorPose3(
                        int(ide1), int(ide2), pose, model))
        # update the isam and result in every timestep
        isam.update(graph, initial_estimate)
        result = isam.calculateEstimate()
    return result


def quaternion_rotation_matrix(qw, qx, qy, qz):
    r00 = 2 * (qw ** 2 + qx ** 2) - 1
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)

    r10 = 2 * (qx * qy + qw * qz)
    r11 = 2 * (qw ** 2 + qy ** 2) - 1
    r12 = 2 * (qy * qz - qw * qx)

    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 2 * (qw ** 2 + qz ** 2) - 1

    quar_rot = np.array([[r00, r01, r02],
                         [r10, r11, r12],
                         [r20, r21, r22]])

    return quar_rot


def plot_batch(initial, optimized):
    plt.figure()
    plt.title("3D Batch Trajectory Optimization")
    ax = plt.axes(projection='3d')
    ax.plot3D(initial[:, 0], initial[:, 1], initial[:, 2], linestyle="-",
              label="Unoptimized Trajectory", color='blue', linewidth=0.5)
    ax.plot3D(optimized[:, 0], optimized[:, 1], optimized[:, 2], linestyle="-",
              label="Optimized Trajectory", color='red', linewidth=0.5)
    ax.set_xlim3d(-200, 50)
    ax.set_ylim3d(0, 250)
    ax.set_zlim3d(-10, 70)
    ax.view_init(30, -120)
    plt.legend()
    plt.savefig("3D_batch_trajectory_optimization.png")


def plot_incremental(initial, optimized):
    plt.figure()
    plt.title("3D Incremental Trajectory Optimization")
    ax = plt.axes(projection='3d')
    ax.plot3D(initial[:, 0], initial[:, 1], initial[:, 2], linestyle="-",
              label="Unoptimized Trajectory", color='blue', linewidth=0.5)
    ax.plot3D(optimized[:, 0], optimized[:, 1], optimized[:, 2], linestyle="-",
              label="Optimized Trajectory", color='red', linewidth=0.5)
    ax.set_xlim3d(-200, 50)
    ax.set_ylim3d(0, 250)
    ax.set_zlim3d(-10, 70)
    ax.view_init(30, -120)
    plt.legend()
    plt.savefig("3D_incremental_trajectory_optimization.png")


def pose3_to_array(result):
    poses = [result.atPose3(i) for i in range(result.size())]
    return np.array([[pose.x(), pose.y(), pose.z()] for pose in poses])


if __name__ == "__main__":
    # 1. read the files
    file = "parking-garage.g2o"
    poses, edges = readG2OFile3d(file)

    # 2. batch solution
    result = batch_solution_3d(file)
    optimized_poses = pose3_to_array(result)
    initial_xyz = poses[:, 1:4]
    plot_batch(initial_xyz, optimized_poses)

    # 3. incremental solution
    incremental_solution_3d(poses, edges)
    optimized_poses_incremental = pose3_to_array(result)
    plot_incremental(initial_xyz, optimized_poses_incremental)
