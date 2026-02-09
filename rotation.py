import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def rotation_matrix_from_vectors(v1, v2):
    """
    Returns the rotation matrix that aligns v1 to v2
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    if np.allclose(cross, 0):  # Vectors are parallel or anti-parallel
        if dot > 0:
            return np.eye(3)  # Same direction
        else:
            # Opposite direction
            # Rotate 180 degrees about an orthogonal axis
            orthogonal = np.array([1, 0, 0])
            if np.allclose(v1, orthogonal):
                orthogonal = np.array([0, 1, 0])
            axis = np.cross(v1, orthogonal)
            axis = axis / np.linalg.norm(axis)
            return rotation_matrix_from_axis_angle(axis, np.pi)

    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    s = np.linalg.norm(cross)
    c = dot

    R = np.eye(3) + skew + (skew @ skew) * ((1 - c) / (s ** 2))
    # print(R)
    return R

def rotation_matrix_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    
    R=np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ])
    print(R)
    return R

def rot_vec(theta, n):
    """
    Rotate theta (in radian) about unit vector n returns the rotation matrix
    """
    if theta:
        n_norm = n / np.linalg.norm(n)
        r_mat = R.from_rotvec(theta * np.array(n_norm)).as_matrix()
    else:
        r_mat = np.identity(3)
    return np.array(r_mat)

if __name__ == "__main__":
    # Example vectors
    v1 = np.array([0, 0, 1])
    v2 = np.array([0, 0, -1])

    # Compute rotation matrix
    R = rotation_matrix_from_vectors(v1, v2)
    #test for orthogonality
    print(np.transpose(R) @ R)
    v1_rotated = R @ v1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Original vector
    ax.quiver(0, 0, 0, *v1, color='r', label='v1', linewidth=5)
    # Rotated vector
    ax.quiver(0, 0, 0, *v2, color='g', label='v2', linewidth=5)
    # Rotated vector
    ax.quiver(0, 0, 0, *v1_rotated, color='b', label='R * v1', linewidth=5)

    ax.quiver(0, 0, 0, *np.cross(v1,v2), color='black', label='rotation axis', linewidth=5)
    if False:
        test_vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 1, 1])]
        for i,vec in enumerate(test_vectors):
        # Plotting
            color_random = np.random.rand(3,)
            
            # Original vector
            ax.quiver(0, 0, 0, *vec, color=color_random, label=f'v{i}', linewidth=1)
            # Rotated vector
            ax.quiver(0, 0, 0, *(R@vec), color=color_random, label=f'v_rot{i}', linewidth=3)
        

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Rotation of v1 to v2')

    plt.tight_layout()
    plt.show()
