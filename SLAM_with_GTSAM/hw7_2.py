import numpy as np
import gtsam
import pdb


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


def batch_solution_3d(poses, edges):
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    params = gtsam.GaussNewtonParams()
    noise = gtsam.noiseModel.Gaussian.Covariance(3, 0.1)
    optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, params)

    
    # actual = optimize_using(gtsam.GaussNewtonOptimizer,
    #                         optimizer, graph, initial_estimate)


def incremental_solution_3d(poses, edges):
    # 1. Initialize iSAM
    params = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(params)
    # print("finish initiliazing")
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # 2. incremental solution
    for i in range(len(poses)):
        print("index: ", i)
        if poses[i, 0] == 0:    
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(0.01*np.ones((6,1)))
            idx = int(poses[i, 0])
            x = poses[i, 1]
            y = poses[i, 2]
            z = poses[i, 3]
            qx = poses[i, 4]
            qy = poses[i, 5]
            qz = poses[i, 6]
            qw = poses[i, 7]
            T = gtsam.Point3(x, y, z)
            R = gtsam.Rot3(qx, qy, qz, qw)
            prior = gtsam.Pose3(R, T)
            graph.add(gtsam.PriorFactorPose3(0, prior, prior_noise))
            initial_estimate.insert(idx, prior)
        else:
            prev_pose = result.atPose3(i-1)
            initial_estimate.insert(int(poses[i, 0]), prev_pose)
            for j in range(len(edges)):
                idx_e1 = int(edges[j, 0])
                idx_e2 = int(edges[j, 1])
                dx = edges[j, 2]
                dy = edges[j, 3]
                dz = edges[j, 4]
                dqx = edges[i, 5]
                dqy = edges[i, 6]
                dqz = edges[i, 7]
                dqw = edges[i, 8]
                info = edges[j, 9:]
                if edges[j, 1] == poses[i, 0]:
                    info_matrix = np.array([[info[0],  info[1],  info[2],  info[3],  info[4],  info[5]],
                                            [info[1],  info[6],  info[7],  info[8],  info[9], info[10]],
                                            [info[2],  info[7], info[11], info[12], info[13], info[14]], 
                                            [info[3],  info[8], info[12], info[15], info[16], info[17]],
                                            [info[4],  info[9], info[13], info[16], info[18], info[19]],
                                            [info[5], info[10], info[14], info[17], info[19], info[20]]])
                    cov = np.linalg.inv(info_matrix)
                    model = gtsam.noiseModel.Gaussian.Covariance(cov)
                    T = gtsam.Point3(dx, dy, dz)
                    R = gtsam.Rot3(dqx, dqy, dqz, dqw)
                    pose = gtsam.Pose3(R, T)
                    graph.add(gtsam.BetweenFactorPose3(idx_e1, idx_e2, pose, model))
        isam.update(graph, initial_estimate)
        result = isam.calculateEstimate()
        print('result: ', result)
        initial_estimate.clear()


if __name__ == "__main__":
    # 1. read the files
    file = "input_INTEL_g2o.g2o"
    poses, edges = readG2OFile3d(file)

    # 2. batch solution

    # 3. incremental solution
    incremental_solution_3d(poses, edges)