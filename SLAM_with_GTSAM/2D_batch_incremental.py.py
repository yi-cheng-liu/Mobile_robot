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
    # 1. read the data
    is3D = False
    graph, initial = gtsam.readG2o(file, is3D)

    # 2. add a prior factor model to the graph
    priorModel = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-6, 1e-6, 1e-4]))
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(), priorModel))

    # 3. build the optimizer with Gauss-Newton method
    params = gtsam.GaussNewtonParams()
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)

    # 4. optimize the graph
    result = optimizer.optimize()

    return result


def incremental_solution_2d(poses, edges):
    # 1. Initialize iSAM
    params = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(params)

    # 2. incremental solution
    for pose in poses:
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        # extract the x, y, theta data from the prestored pose
        idp, x, y, theta = pose
        if idp == 0:
            # create the prior_noise and add to the graph
            prior_noise = gtsam.noiseModel.Diagonal.Variances(
                np.array([1e-6, 1e-6, 1e-4]))
            graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(x, y, theta),
                                             prior_noise))
            initial_estimate.insert(int(idp), gtsam.Pose2(x, y, theta))
        else:
            # add the previously calcualted result add add to the subsequence idx
            prev_pose = result.atPose2(int(idp-1))
            initial_estimate.insert(int(idp), prev_pose)
            for edge in edges:
                # calcualte the information matrix
                ide1, ide2, dx, dy, dtheta, info_0, info_1, info_2, info_3, info_4, info_5 = edge
                info_matrix = np.array([[info_0, info_1, info_2],
                                        [info_1, info_3, info_4],
                                        [info_2, info_4, info_5]])
                if ide2 == idp:
                    # build the noise model with the information matrix
                    cov = np.linalg.inv(info_matrix)
                    model = gtsam.noiseModel.Gaussian.Covariance(cov)
                    # update the graph
                    graph.add(gtsam.BetweenFactorPose2(int(ide1), int(ide2),
                                                       gtsam.Pose2(dx, dy,
                                                                   dtheta),
                                                       model))
        # update the isam and result in every timestep
        isam.update(graph, initial_estimate)
        result = isam.calculateEstimate()
    return result


def plot_batch(initial, optimized):
    plt.figure()
    plt.title("2D Batch Trajectory Optimization")
    plt.plot(initial[:, 0], initial[:, 1], linestyle="-",
             label="Unoptimized Trajectory", color='blue', linewidth=1)
    plt.plot(optimized[:, 0], optimized[:, 1], linestyle="-",
             label="Optimized Trajectory", color='red', linewidth=1)
    plt.legend()
    plt.axis("equal")
    plt.savefig("2D_batch_trajectory_optimization.png")


def plot_incremental(initial, optimized):
    plt.figure()
    plt.title("2D Incremental Trajectory Optimization")
    plt.plot(initial[:, 0], initial[:, 1], linestyle="-",
             label="Unoptimized Trajectory", color='blue', linewidth=1)
    plt.plot(optimized[:, 0], optimized[:, 1], linestyle="-",
             label="Optimized Trajectory", color='red', linewidth=1)
    plt.legend()
    plt.axis("equal")
    plt.savefig("2D_incremental_trajectory_optimization.png")


def pose2_to_array(result):
    poses = [result.atPose2(i) for i in range(result.size())]
    return np.array([[pose.x(), pose.y(), pose.theta()] for pose in poses])


if __name__ == "__main__":
    # 1. read the files
    file = "input_INTEL_g2o.g2o"
    poses, edges = readG2OFile2d(file)

    # 2. batch solution
    result = batch_solution_2d(file)
    optimized_poses_batch = pose2_to_array(result)
    initial_xy = poses[:, 1:3]
    plot_batch(initial_xy, optimized_poses_batch)

    # 3. incremental solution
    result = incremental_solution_2d(poses, edges)
    optimized_poses_incremental = pose2_to_array(result)
    plot_incremental(initial_xy, optimized_poses_incremental)
