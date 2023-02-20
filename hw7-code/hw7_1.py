import numpy as np
import gtsam
import matplotlib.pyplot as plt


def readG2OFile2d(file_name):
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


def batch_solution_2d(file):
    is3D = False
    graph, initial = gtsam.readG2o(file, is3D)
    params = gtsam.GaussNewtonParams()
    priorModel = gtsam.noiseModel.Diagonal.Variances(
        gtsam.Point3(1e-6, 1e-6, 1e-8))

    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(), priorModel))
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
    result = optimizer.optimize()

    return result


def incremental_solution_2d(poses, edges):
    # 1. Initialize iSAM
    params = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(params)
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # 2. incremental solution
    for i in range(len(poses)):
        print('i: ', i)
        if poses[i, 0] == 0:
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.3, 0.3, 0.1]))
            idx = int(poses[i, 0])
            x = poses[i, 1]
            y = poses[i, 2]
            theta = poses[i, 3]
            graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(x, y, theta),
                                             prior_noise))
            initial_estimate.insert(idx, gtsam.Pose2(x, y, theta))
        else:
            prev_pose = result.atPose2(i-1)
            initial_estimate.insert(int(poses[i, 0]), prev_pose)
            for j in range(len(edges)):
                idx_e1 = int(edges[j, 0])
                idx_e2 = int(edges[j, 1])
                dx = edges[j, 2]
                dy = edges[j, 3]
                dtheta = edges[j, 4]
                info = edges[j, 5:]
                if edges[j, 1] == poses[i, 0]:
                    info_matrix = np.array([[info[0], info[1], info[2]],
                                            [info[1], info[3], info[4]],
                                            [info[2], info[4], info[5]]])
                    cov = np.linalg.inv(info_matrix)
                    model = gtsam.noiseModel.Gaussian.Covariance(cov)
                    graph.add(gtsam.BetweenFactorPose2(idx_e1, idx_e2,
                                                       gtsam.Pose2(dx, dy,
                                                                   dtheta),
                                                       model))
        isam.update(graph, initial_estimate)
        result = isam.calculateEstimate()
        initial_estimate.clear()
    return result


def plot_batch(initial, optimized):
    plt.figure()
    plt.title("2D Batch Trajectory Optimization")
    plt.plot(initial[:, 0], initial[:, 1], linestyle="-",
             label="Unoptimized Trajectory", color='blue')
    plt.plot(optimized[:, 0], optimized[:, 1], linestyle="-",
             label="Optimized Trajectory", color='red')
    plt.legend()
    plt.axis("equal")
    plt.savefig("2D_batch_trajectory_optimization.png")


def plot_incremental(initial, optimized):
    plt.figure()
    plt.title("2D Incremental Trajectory Optimization")
    plt.plot(initial[:, 0], initial[:, 1], linestyle="-",
             label="Unoptimized Trajectory", color='blue')
    plt.plot(optimized[:, 0], optimized[:, 1], linestyle="-",
             label="Optimized Trajectory", color='red')
    plt.legend()
    plt.axis("equal")
    plt.savefig("2D_incremental_trajectory_optimization.png")


def pose2array(result):
    poses = [result.atPose2(i) for i in range(result.size())]
    return np.array([[pose.x(), pose.y(), pose.theta()] for pose in poses])


if __name__ == "__main__":
    # 1. read the files
    file = "input_INTEL_g2o.g2o"
    poses, edges = readG2OFile2d(file)

    # 2. batch solution
    result = batch_solution_2d(file)
    optimized_poses_batch = pose2array(result)
    initial_xy = poses[:, 1:3]
    plot_batch(initial_xy, optimized_poses_batch)

    # 3. incremental solution
    result = incremental_solution_2d(poses, edges)
    optimized_poses_incremental = pose2array(result)
    plot_incremental(initial_xy, optimized_poses_incremental)

