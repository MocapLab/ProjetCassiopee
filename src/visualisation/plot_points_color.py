import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image as im

import sys
import os

src_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..\..'))
sys.path.append(src_folder)

from src.dataset import MocaplabDatasetCNN
from src.setup import setup_python, setup_pytorch
from src.models.mocaplab import TestCNN


def create_images(i, data, label, prediction, nom, heatmap, size):

    if not os.path.exists(f"{src_folder}/train_results/mocaplab/animation/{nom}"):
        os.mkdir(f"{src_folder}/train_results/mocaplab/animation/{nom}")

    print(f"i={i}")
    print(f"data={data}")
    print(f"label={label}")
    print(f"prediction={prediction}")
    print(f"nom={nom}")

    #We say that the list of joints to put in color is the heatmap parameter
    joints_color = heatmap

    model = "CNN"

    data = data.numpy()
    data = im.fromarray(data[0,...])

    data = data.resize((size[1], size[0]))

    data = np.array(data).T
    print(data.shape)
    
    x_data = data[:, 0::3]
    y_data = data[:, 1::3]
    z_data = data[:, 2::3]

    # Initialize lines
    # line_points_indices = [
    #     (0, 1), (0, 2), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 9), (8, 9), (1, 70),     # Chest and head
    #     (3, 10), (2, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),                  # Right arm (without hand)
    #     (2, 40), (3, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45),                  # Left arm (without hand)
    #     (70, 71), (71, 72), (72, 73), (73, 74),                                              # Right leg
    #     (70, 75), (75, 76), (76, 77), (77, 78),                                              # Left leg
    #     (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),                                    # Right hand, pinky
    #     (15, 21), (21, 22), (22, 23), (23, 24), (24, 25),                                    # Right hand, ring
    #     (15, 26), (26, 27), (27, 28), (28, 29), (29, 30),                                    # Right hand, mid
    #     (15, 31), (31, 32), (32, 33), (33, 34), (34, 35),                                    # Right hand, index
    #     (15, 36), (36, 37), (37, 38), (38, 39),                                              # Right hand, thumb
    #     (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),                                    # Left hand, pinky
    #     (45, 51), (51, 52), (52, 53), (53, 54), (54, 55),                                    # Left hand, ring
    #     (45, 56), (56, 57), (57, 58), (58, 59), (59, 60),                                    # Left hand, mid
    #     (45, 61), (61, 62), (62, 63), (63, 64), (64, 65),                                    # Left hand, index
    #     (45, 66), (66, 67), (67, 68), (68, 69)                                               # Left hand, thumb                
    # ]

    for frame in range(len(data)):

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(min([min(x_data[i]) for i in range(len(x_data))]),
                    max([max(x_data[i]) for i in range(len(x_data))]))
        ax.set_ylim(min([min(y_data[i]) for i in range(len(y_data))]),
                    max([max(y_data[i]) for i in range(len(y_data))]))
        ax.set_zlim(min([min(z_data[i]) for i in range(len(z_data))]),
                    max([max(z_data[i]) for i in range(len(z_data))]))

        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")

        # Initialize an empty scatter plot (to be updated in the animation)
        scatter = ax.scatter([], [], [])

        # num_lines = len(line_points_indices)
        # lines = [ax.plot([], [], [], "-", color="red")[0] for _ in range(num_lines)]


        frame_coordinates = (x_data[frame], z_data[frame], y_data[frame])

        # Update the scatter plot with new point positions
        scatter._offsets3d = frame_coordinates

        # Set the title to display the current data, label and frame
        ax.set_title(f"Data {i}, Label : {label}, Prediction : {prediction} with model {model} \nFrame: {frame}")

        # Adding lines between the joints
        # for line, (start, end) in zip(lines, line_points_indices):
        #     line.set_data_3d([x_data[frame][start], x_data[frame][end]],
        #                      [z_data[frame][start], z_data[frame][end]],
        #                      [y_data[frame][start], y_data[frame][end]])

        # Update the colors of the points based on the joints_color list
        colors = ["limegreen" if idx in joints_color[frame] else "blue" for idx in range(len(x_data[frame]))]

        sizes = [200 if idx in joints_color[frame] else 50 for idx in range(len(x_data[frame]))] # Set larger size for points with red color

        scatter.set_edgecolors(colors)
        scatter.set_facecolors(colors)
        # scatter.set_sizes(sizes) # Broken
        
        fig.savefig(f"{src_folder}/train_results/mocaplab/animation/{nom}/{frame}.png")
        plt.close()

def create_animation(i, data, label, prediction, nom, heatmap, size):

    print(f"i={i}")
    print(f"data={data}")
    print(f"label={label}")
    print(f"prediction={prediction}")
    print(f"nom={nom}")

    # List of points that should appear in different color each frame (list of 100 lists of len 10) 
    '''
    points_color_indices = [
        [231, 220, 230,  24, 232, 221,  23, 219, 222,  25],
        [ 24,  23, 220, 231, 221,  25, 222, 230,  22, 223],
        [ 24,  23,  25, 227,  22, 226, 225,  26, 224, 228],
        [ 24,  25,  23,  26,  27,  22,  28, 227, 228, 226],
        [ 24,  25,  26,  23,  27,  28, 227,  22, 228, 226],
        [ 24,  25,  26,  23,  27, 227, 228, 226,  28,  22],
        [ 24,  25,  26, 227,  27,  23, 228, 226,  28, 229],
        [ 24,  25, 227,  26,  27,  23, 228, 226, 229,  28],
        [ 24, 227,  25,  26,  27,  23, 228, 226, 229,  28],
        [ 24, 227,  25,  26,  27,  23, 228, 226, 229, 116],
        [ 24, 227,  25,  26,  23,  27, 228, 226, 109,  94],
        [ 24, 227,  25,  26,  23,  27, 228, 109, 226,  94],
        [ 24, 227,  25,  26,  23,  27, 228, 109, 226,  94],
        [ 24, 227,  25,  26, 109, 228,  27,  23,  61, 226],
        [ 24, 227,  25, 109,  26,  61,  27, 228,  23,  60],
        [ 24, 227,  61,  25, 109,  26,  27, 228,  60,  23],
        [ 61, 227,  24,  25, 109,  26,  27,  60, 228,  23],
        [227, 109,  24,  61,  25,  26,  27, 228, 108,  60],
        [109, 227,  24,  61,  25,  26,  27, 108, 228,  60],
        [109, 227,  24,  61,  25,  26,  27, 108, 228,  60],
        [109, 227,  24,  25,  61,  26,  27, 108, 228,  60],
        [227, 109,  24,  25,  26,  61,  27, 108, 228,  60],
        [227,  24, 109,  25,  26,  27,  61, 228, 108,  23],
        [227,  24,  25,  26, 109,  27,  61, 228, 108,  23],
        [227,  24,  25,  26,  27, 109,  61, 228,  23, 108],
        [227,  24,  25,  26,  27, 109,  61, 228,  23, 108],
        [227,  24,  25,  26,  27, 109,  61, 228,  23, 108],
        [227,  24,  25,  26,  27, 109,  61, 228,  23, 108],
        [227,  24,  25,  26, 109,  27,  61, 228,  23, 108],
        [227,  24,  25,  26, 109,  27,  61, 228,  23, 108],
        [227,  24,  25,  26, 109,  27,  61, 228,  23,  60],
        [227,  24,  25,  26, 109,  61,  27, 228,  23,  60],
        [227,  24,  25,  26,  61, 109,  27, 228,  23,  60],
        [227,  24,  25,  26,  61, 109,  27, 228,  23,  60],
        [227,  24,  25,  26,  61, 109,  27, 228,  23,  60],
        [227,  24,  25,  26,  61,  27, 109, 228,  23,  60],
        [227,  24,  25,  26,  61,  27, 109, 228,  23,  60],
        [227,  24,  25,  26,  61,  27, 109, 228,  23,  60],
        [227,  24,  25,  26,  61,  27, 109, 228,  23,  60],
        [227,  24,  25,  26,  61,  27, 109, 228,  60,  23],
        [227,  24,  25,  61,  26,  27, 109, 228,  60,  23],
        [227,  24,  25,  61,  26,  27, 228, 109,  60,  23],
        [227,  24,  61,  25,  26,  27, 228, 109,  60,  23],
        [227,  61,  24,  25,  26,  27,  60, 109, 228,  23],
        [ 61, 227,  24,  25,  26, 109,  60,  27, 228,  23],
        [ 61, 227, 109,  24,  60,  25,  26,  27, 108, 228],
        [ 61, 109, 227,  60,  24,  25, 108,  26,  27, 110],
        [ 61, 109,  60, 108, 227,  24, 110,  25,  26,  27],
        [ 61, 109,  60, 108, 227, 110,  24,  25,  26,  27],
        [ 61, 109,  60, 108, 227,  24, 110,  25,  26,  27],
        [ 61, 109,  60, 227,  24,  25, 108,  26, 110,  27],
        [ 61, 109, 227,  60,  24,  25,  26,  27, 108, 228],
        [ 61, 227,  60,  24, 109,  25,  26,  27, 228,  23],
        [ 61, 227,  60,  24,  25,  26, 109,  27, 228,  23],
        [ 61, 227,  60,  24,  25,  26,  27, 109, 228,  23],
        [ 61,  60, 227,  24,  25,  26,  27, 109, 228,  23],
        [ 61,  60, 227, 109,  24,  25,  26,  27, 228, 108],
        [ 61,  60, 109, 227,  24, 108,  25, 110,  26,  27],
        [ 61, 109,  60, 108, 227, 110,  24,  25,  26,  62],
        [ 61, 109,  60, 108, 110, 227,  24,  25,  26, 107],
        [109,  61, 108,  60, 110, 227,  24,  25,  26,  27],
        [109,  61, 227, 108,  24,  25,  60,  26, 110,  27],
        [227, 109,  24,  25,  26,  61,  27, 228, 108,  60],
        [227,  24,  25,  26,  27, 228, 109,  23, 226,  61],
        [227,  24,  25,  26,  27, 228,  23, 226, 109, 229],
        [227,  24,  25,  26,  27, 228, 226,  23, 229, 225],
        [227,  24,  25,  26,  27, 228, 226,  23, 229, 225],
        [227,  24,  25,  26,  27, 228,  23, 226, 229, 225],
        [227,  24,  25,  26, 228,  27,  23, 226, 229, 225],
        [227,  24,  25,  26, 228,  27,  23, 226, 229,  28],
        [227,  24,  25,  26, 228,  27,  23, 226, 229,  28],
        [227,  24,  25,  26, 228,  27,  23, 226, 229,  28],
        [227,  24,  25,  26, 228,  27,  23, 226, 229,  28],
        [227,  24,  25,  26, 228,  27,  23, 226, 229,  28],
        [ 24, 227,  25,  26, 228,  27,  23, 226, 229,  28],
        [ 24, 227,  25,  26, 228,  27,  23, 226, 229,  28],
        [ 24, 227,  25,  26,  27, 228,  23, 226,  28, 229],
        [ 24,  25,  26, 227,  27,  23, 228, 226,  28, 229],
        [227,  24,  25,  26,  27, 228,  23, 226,  28, 229],
        [227,  24,  25, 228,  26,  27,  23, 226, 229,  28],
        [227,  24,  25, 228,  23,  26, 226,  27, 229, 225],
        [ 24,  25,  23,  26,  27,  22, 227, 228, 116,  28],
        [ 24,  23,  25,  26,  22,  27, 116,  76,  16,  15],
        [ 24,  23,  25,  22,  26,  21, 198,  27,  76, 197],
        [236, 235,  24, 234,  23,  25, 233,  26,  22,  27],
        [236, 235, 234, 233,  24,  23,  25,  35,  26,  75],
        [235, 236, 234,  20,  21, 183,  75,  74,  73,  19],
        [ 20,  21, 236, 235,  22,  19,  72,  23,  73,  74],
        [231, 230, 229, 228, 227, 232,  23,  24,  22,  21],
        [231, 232,  23,  24, 230, 233,  22, 229, 234, 236],
        [236, 235, 234, 233, 232, 231,  23,  24, 230,  22],
        [236, 235, 234, 233, 232, 231, 213, 212, 230, 214],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227],
        [235, 236, 234, 233, 232, 231, 230, 229, 228, 227],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227],
        [235, 236, 234, 233, 232, 231, 230, 229, 228, 227],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227]
    ]

    # Transform the list into a list of lists of joints that should be in different color in each frame
    joints_color = []
    for i in range(100):
        joints_color_frame = []
        for j in range(10):
            joints_color_frame.append(points_color_indices[i][j] // 3)
        joints_color.append(joints_color_frame)
    '''
    
    #We say that the list of joints to put in color is the heatmap parameter
    joints_color = heatmap

    model = "CNN"
    data = data.numpy()
    data = im.fromarray(data[0,...])
    data = data.resize((size[1], size[0]))

    data = np.array(data).T
    print(data.shape)

    x_data = data[:, 0::3]
    y_data = data[:, 1::3]
    z_data = data[:, 2::3]

    print(x_data.shape)
    print(y_data.shape)
    print(z_data.shape)
    fig = plt.figure()
    

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(min([min(x_data[i]) for i in range(len(x_data))]),
                max([max(x_data[i]) for i in range(len(x_data))]))
    ax.set_ylim(min([min(y_data[i]) for i in range(len(y_data))]),
                max([max(y_data[i]) for i in range(len(y_data))]))
    ax.set_zlim(min([min(z_data[i]) for i in range(len(z_data))]),
                max([max(z_data[i]) for i in range(len(z_data))]))

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    # Initialize an empty scatter plot (to be updated in the animation)
    scatter = ax.scatter([], [], [])

    # Initialize lines
    # line_points_indices = [
    #     (0, 1), (0, 2), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 9), (8, 9), (1, 70),     # Chest and head
    #     (3, 10), (2, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),                  # Right arm (without hand)
    #     (2, 40), (3, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45),                  # Left arm (without hand)
    #     (70, 71), (71, 72), (72, 73), (73, 74),                                              # Right leg
    #     (70, 75), (75, 76), (76, 77), (77, 78),                                              # Left leg
    #     (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),                                    # Right hand, pinky
    #     (15, 21), (21, 22), (22, 23), (23, 24), (24, 25),                                    # Right hand, ring
    #     (15, 26), (26, 27), (27, 28), (28, 29), (29, 30),                                    # Right hand, mid
    #     (15, 31), (31, 32), (32, 33), (33, 34), (34, 35),                                    # Right hand, index
    #     (15, 36), (36, 37), (37, 38), (38, 39),                                              # Right hand, thumb
    #     (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),                                    # Left hand, pinky
    #     (45, 51), (51, 52), (52, 53), (53, 54), (54, 55),                                    # Left hand, ring
    #     (45, 56), (56, 57), (57, 58), (58, 59), (59, 60),                                    # Left hand, mid
    #     (45, 61), (61, 62), (62, 63), (63, 64), (64, 65),                                    # Left hand, index
    #     (45, 66), (66, 67), (67, 68), (68, 69)                                               # Left hand, thumb                
    # ]

    # num_lines = len(line_points_indices)
    # lines = [ax.plot([], [], [], "-", color="red")[0] for _ in range(num_lines)]

    # Function to update the scatter plot
    def update(frame):    

        # Get the coordinates for the current frame
        frame_coordinates = (x_data[frame], z_data[frame], y_data[frame])

        # Update the scatter plot with new point positions
        scatter._offsets3d = frame_coordinates

        # Set the title to display the current data, label and frame
        ax.set_title(f"Data {nom[3:-6]}, Number {i}, Label : {label}, Prediction : {prediction} with model {model} \nFrame: {frame}")

        # Adding lines between the joints
        # for line, (start, end) in zip(lines, line_points_indices):
        #     line.set_data_3d([x_data[frame][start], x_data[frame][end]],
        #                      [z_data[frame][start], z_data[frame][end]],
        #                      [y_data[frame][start], y_data[frame][end]])

        # Update the colors of the points based on the joints_color list
        colors = ["limegreen" if idx in joints_color[frame] else "blue" for idx in range(len(x_data[frame]))]
        sizes = [200 if idx in joints_color[frame] else 50 for idx in range(len(x_data[frame]))] # Set larger size for points with red color
        scatter.set_edgecolors(colors)
        scatter.set_facecolors(colors)
        # scatter.set_sizes(sizes)

        return scatter

    # Create the animation
    animation = FuncAnimation(fig, update, frames=len(data), blit=True)
    
    # Save the animation as a GIF
    animation.save(f"{src_folder}/src/visualisation/mocaplab_points_color/{nom}.gif",
                   writer='pillow')
    plt.close(fig)


def create_all_animations(results_dir="train_results/mocaplab/supervised"):
    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch(gpu=False)

    print("#### Dataset ####")
    # dataset_fc = MocaplabDatasetFC(path=(f"{src_folder}/data/mocaplab/Cassiopée_Allbones"),
    #                             padding=True, 
    #                             train_test_ratio=8,
    #                             validation_percentage=0.01)
    # bones_to_keep = "abdomenUpper_T_glob;abdomenUpper_T_glob;abdomenUpper_T_glob;chestLower_T_glob;chestLower_T_glob;chestLower_T_glob;chestUpper_T_glob;chestUpper_T_glob;chestUpper_T_glob;neckLower_T_glob;neckLower_T_glob;neckLower_T_glob;rCollar_T_glob;rCollar_T_glob;rCollar_T_glob;rShldrBend_T_glob;rShldrBend_T_glob;rShldrBend_T_glob;rShldrTwist_T_glob;rShldrTwist_T_glob;rShldrTwist_T_glob;rForearmBend_T_glob;rForearmBend_T_glob;rForearmBend_T_glob;rForearmTwist_T_glob;rForearmTwist_T_glob;rForearmTwist_T_glob;rHand_T_glob;rHand_T_glob;rHand_T_glob;rCarpal4_T_glob;rCarpal4_T_glob;rCarpal4_T_glob;rPinky1_T_glob;rPinky1_T_glob;rPinky1_T_glob;rPinky2_T_glob;rPinky2_T_glob;rPinky2_T_glob;rPinky3_T_glob;rPinky3_T_glob;rPinky3_T_glob;rPinky3_end_T_glob;rPinky3_end_T_glob;rPinky3_end_T_glob;rCarpal3_T_glob;rCarpal3_T_glob;rCarpal3_T_glob;rRing1_T_glob;rRing1_T_glob;rRing1_T_glob;rRing2_T_glob;rRing2_T_glob;rRing2_T_glob;rRing3_T_glob;rRing3_T_glob;rRing3_T_glob;rRing3_end_T_glob;rRing3_end_T_glob;rRing3_end_T_glob;rCarpal2_T_glob;rCarpal2_T_glob;rCarpal2_T_glob;rMid1_T_glob;rMid1_T_glob;rMid1_T_glob;rMid2_T_glob;rMid2_T_glob;rMid2_T_glob;rMid3_T_glob;rMid3_T_glob;rMid3_T_glob;rMid3_end_T_glob;rMid3_end_T_glob;rMid3_end_T_glob;rCarpal1_T_glob;rCarpal1_T_glob;rCarpal1_T_glob;rIndex1_T_glob;rIndex1_T_glob;rIndex1_T_glob;rIndex2_T_glob;rIndex2_T_glob;rIndex2_T_glob;rIndex3_T_glob;rIndex3_T_glob;rIndex3_T_glob;rIndex3_end_T_glob;rIndex3_end_T_glob;rIndex3_end_T_glob;rThumb1_T_glob;rThumb1_T_glob;rThumb1_T_glob;rThumb2_T_glob;rThumb2_T_glob;rThumb2_T_glob;rThumb3_T_glob;rThumb3_T_glob;rThumb3_T_glob;rThumb3_end_T_glob;rThumb3_end_T_glob;rThumb3_end_T_glob;lCollar_T_glob;lCollar_T_glob;lCollar_T_glob;lShldrBend_T_glob;lShldrBend_T_glob;lShldrBend_T_glob;lShldrTwist_T_glob;lShldrTwist_T_glob;lShldrTwist_T_glob;lForearmBend_T_glob;lForearmBend_T_glob;lForearmBend_T_glob;lForearmTwist_T_glob;lForearmTwist_T_glob;lForearmTwist_T_glob;lHand_T_glob;lHand_T_glob;lHand_T_glob;lCarpal4_T_glob;lCarpal4_T_glob;lCarpal4_T_glob;lPinky1_T_glob;lPinky1_T_glob;lPinky1_T_glob;lPinky2_T_glob;lPinky2_T_glob;lPinky2_T_glob;lPinky3_T_glob;lPinky3_T_glob;lPinky3_T_glob;lPinky3_end_T_glob;lPinky3_end_T_glob;lPinky3_end_T_glob;lCarpal3_T_glob;lCarpal3_T_glob;lCarpal3_T_glob;lRing1_T_glob;lRing1_T_glob;lRing1_T_glob;lRing2_T_glob;lRing2_T_glob;lRing2_T_glob;lRing3_T_glob;lRing3_T_glob;lRing3_T_glob;lRing3_end_T_glob;lRing3_end_T_glob;lRing3_end_T_glob;lCarpal2_T_glob;lCarpal2_T_glob;lCarpal2_T_glob;lMid1_T_glob;lMid1_T_glob;lMid1_T_glob;lMid2_T_glob;lMid2_T_glob;lMid2_T_glob;lMid3_T_glob;lMid3_T_glob;lMid3_T_glob;lMid3_end_T_glob;lMid3_end_T_glob;lMid3_end_T_glob;lCarpal1_T_glob;lCarpal1_T_glob;lCarpal1_T_glob;lIndex1_T_glob;lIndex1_T_glob;lIndex1_T_glob;lIndex2_T_glob;lIndex2_T_glob;lIndex2_T_glob;lIndex3_T_glob;lIndex3_T_glob;lIndex3_T_glob;lIndex3_end_T_glob;lIndex3_end_T_glob;lIndex3_end_T_glob;lThumb1_T_glob;lThumb1_T_glob;lThumb1_T_glob;lThumb2_T_glob;lThumb2_T_glob;lThumb2_T_glob;lThumb3_T_glob;lThumb3_T_glob;lThumb3_T_glob;lThumb3_end_T_glob;lThumb3_end_T_glob;lThumb3_end_T_glob".split(';')
    bones_to_keep = list(set("UPHD;UPHD;UPHD;LFHD;LFHD;LFHD;RFHD;RFHD;RFHD;LBHD;LBHD;LBHD;RBHD;RBHD;RBHD;C7;C7;C7;T10;T10;T10;LBAC;LBAC;LBAC;RBAC;RBAC;RBAC;CLAV;CLAV;CLAV;STRN;STRN;STRN;LCLAV;LCLAV;LCLAV;RCLAV;RCLAV;RCLAV;LFSHO;LFSHO;LFSHO;LSHOULD;LSHOULD;LSHOULD;LBSHO;LBSHO;LBSHO;LUPA;LUPA;LUPA;LELB;LELB;LELB;LELBEXT;LELBEXT;LELBEXT;LFRM;LFRM;LFRM;LWRA;LWRA;LWRA;LWRB;LWRB;LWRB;RFSHO;RFSHO;RFSHO;RSHOULD;RSHOULD;RSHOULD;RBSHO;RBSHO;RBSHO;RUPA;RUPA;RUPA;RELB;RELB;RELB;RELBEXT;RELBEXT;RELBEXT;RFRM;RFRM;RFRM;RWRA;RWRA;RWRA;RWRB;RWRB;RWRB;LFWT;LFWT;LFWT;RFWT;RFWT;RFWT;LBWT;LBWT;LBWT;RBWT;RBWT;RBWT;LHIP;LHIP;LHIP;LUPLEG;LUPLEG;LUPLEG;LKNE;LKNE;LKNE;LPER;LPER;LPER;LTIB;LTIB;LTIB;LANK;LANK;LANK;LHEE;LHEE;LHEE;LMT5;LMT5;LMT5;LTOE;LTOE;LTOE;LMT1;LMT1;LMT1;RHIP;RHIP;RHIP;RUPLEG;RUPLEG;RUPLEG;RKNE;RKNE;RKNE;RPER;RPER;RPER;RTIB;RTIB;RTIB;RANK;RANK;RANK;RHEE;RHEE;RHEE;RMT5;RMT5;RMT5;RTOE;RTOE;RTOE;RMT1;RMT1;RMT1".split(';')))
    
    dataset_cnn = MocaplabDatasetCNN(path=(f"{src_folder}/data/mocaplab/LSDICOS"),
                                padding=True, 
                                train_test_ratio=8,
                                validation_percentage=0.01, bones_to_keep=bones_to_keep)
    print("#### Data Loader ####")
    # data_loader_fc = DataLoader(dataset_fc,
    #                          batch_size=1,
    #                          shuffle=False)

    
    print(dataset_cnn.max_length)
    data_loader_cnn = DataLoader(dataset_cnn,
                             batch_size=1,
                             shuffle=False)

    
    print("#### Model ####")
    # model = MocaplabFC(dataset_fc.max_length*237).to(DEVICE)
    # model.load_state_dict(torch.load((f"{src_folder}/src/models/mocaplab/all/saved_models/FC/model_20240325_141951.ckpt"),
    #                                  map_location=torch.device("cpu")))
    # model = model.to(DEVICE)
    # model = model.double()
  

    #intialize the CNN model
    cnn = TestCNN()

    # Load the trained weights cnn old model
    cnn.load_state_dict(torch.load((f"{src_folder}/src/models/mocaplab/all/saved_models/CNN/CNN_20240614_193233.ckpt"),
                                    map_location=torch.device("cpu")))

    # Load the trained weights cnn new model
    #cnn.load_state_dict(torch.load(("/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/models/mocaplab/cnn/saved_models/CNN_20240514_211739.ckpt"),
                                    # map_location=torch.device("cpu")))

    # set the evaluation mode
    cnn.eval()

    heatmap_data = []   # a table that contains 112 lists of 100 lists of size 10 (10 max joints for each frame for each data)

    for k, img in enumerate(data_loader_cnn):
        ###TO GET THE HEATMAP LIST OF 10 MOST IMPORTANT JOINTS###
        img, label, name = img
        print(f"img {os.path.splitext(name[0])[0]}: {k:4} / {len(data_loader_cnn)} ")
        # get the most likely prediction of the model
        pred = cnn(img)

        if int(pred[0,1].detach().numpy().round(decimals=0)) != label[0].numpy().round(decimals=0):
            print("error")
            print(pred[0][1].detach().numpy().round(decimals=0))
            print(label[0].numpy().round(decimals=0))
        # get the gradient of the output with respect to the parameters of the model
        pred[:,0].backward()

        # pull the gradients out of the model
        gradients = cnn.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = cnn.get_activations(img).detach()

        # weight the channels by corresponding gradients
        for i in range(256):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        #For each time frame, get the coordinates of the ten maximum activation values to find the most significant joints
        #First, reshape the heatmap (64x64) to original size (100x237)
        heatmap_resized = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap_resized = F.interpolate(heatmap_resized,size=(dataset_cnn.max_length,dataset_cnn[0][0].shape[1]), mode='bilinear')
        heatmap_resized = torch.squeeze(heatmap_resized)
        ten_max_joints_all_frames = []

        for i in range(0, dataset_cnn.max_length):
            max_for_one_joint = []
            for j in range(0, int(dataset_cnn[0][0].shape[1]/3)): #237/3
                max = torch.max(heatmap_resized[i][j*3:j*3+3])
                max_for_one_joint.append(max)
            _, max_activations_indices = torch.topk(torch.as_tensor(max_for_one_joint), k=10)
            ten_max_joints_all_frames.append(max_activations_indices)
        
        heatmap_data.append(ten_max_joints_all_frames)

        ### CREATING THE HEATMAPS PLOTS ###
        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        nom = f'{k}_{os.path.splitext(name[0])[0]}'

        # draw the heatmap
        plt.matshow(heatmap.squeeze())
        plt.savefig(f"{src_folder}/src/visualisation/LSDICOS_heatmap_cnn_{nom}.png")

    print("#### Plot ####")
    for i, batch in enumerate(data_loader_cnn):
        
        print(f"## Batch {i:4} / {len(data_loader_cnn)} ##")
        data, label, name = batch
        data = data.to(torch.float32).to(DEVICE)
        label = label.to(torch.float32).to(DEVICE)
    
        # Make predictions for batch
        #data_flattened = data.view(data.size(0), -1)
        output = cnn(data)

        # Update accuracy variables
        _, predicted = torch.max(output.data, dim=1)


        label = int(label[0])
        predicted = int(predicted[0])
        
        data = data.squeeze(0)
        
        nom = f"cnn_{i}_{name[0]}_{label}_{predicted}"
        
        create_images(i, data, label, predicted, nom, heatmap_data[i], size=[dataset_cnn.max_length, dataset_cnn[0][0].shape[1]])
        create_animation(i, data, label, predicted, nom, heatmap_data[i], size=[dataset_cnn.max_length, dataset_cnn[0][0].shape[1]])

    print("#### DONE ####")


'''
def create_animation(data, label, prediction, name, model):

    print(f"data={data}")
    print(f"type(data)={type(data)}")
    print(f"label={label}")
    print(f"prediction={prediction}")
    print(f"name={name}")
    print(f"model={model}")

    # List of points that should appear in different color each frame (list of 100 lists of len 10) 
    points_color_indices = [
        [231, 220, 230,  24, 232, 221,  23, 219, 222,  25],
        [ 24,  23, 220, 231, 221,  25, 222, 230,  22, 223],
        [ 24,  23,  25, 227,  22, 226, 225,  26, 224, 228],
        [ 24,  25,  23,  26,  27,  22,  28, 227, 228, 226],
        [ 24,  25,  26,  23,  27,  28, 227,  22, 228, 226],
        [ 24,  25,  26,  23,  27, 227, 228, 226,  28,  22],
        [ 24,  25,  26, 227,  27,  23, 228, 226,  28, 229],
        [ 24,  25, 227,  26,  27,  23, 228, 226, 229,  28],
        [ 24, 227,  25,  26,  27,  23, 228, 226, 229,  28],
        [ 24, 227,  25,  26,  27,  23, 228, 226, 229, 116],
        [ 24, 227,  25,  26,  23,  27, 228, 226, 109,  94],
        [ 24, 227,  25,  26,  23,  27, 228, 109, 226,  94],
        [ 24, 227,  25,  26,  23,  27, 228, 109, 226,  94],
        [ 24, 227,  25,  26, 109, 228,  27,  23,  61, 226],
        [ 24, 227,  25, 109,  26,  61,  27, 228,  23,  60],
        [ 24, 227,  61,  25, 109,  26,  27, 228,  60,  23],
        [ 61, 227,  24,  25, 109,  26,  27,  60, 228,  23],
        [227, 109,  24,  61,  25,  26,  27, 228, 108,  60],
        [109, 227,  24,  61,  25,  26,  27, 108, 228,  60],
        [109, 227,  24,  61,  25,  26,  27, 108, 228,  60],
        [109, 227,  24,  25,  61,  26,  27, 108, 228,  60],
        [227, 109,  24,  25,  26,  61,  27, 108, 228,  60],
        [227,  24, 109,  25,  26,  27,  61, 228, 108,  23],
        [227,  24,  25,  26, 109,  27,  61, 228, 108,  23],
        [227,  24,  25,  26,  27, 109,  61, 228,  23, 108],
        [227,  24,  25,  26,  27, 109,  61, 228,  23, 108],
        [227,  24,  25,  26,  27, 109,  61, 228,  23, 108],
        [227,  24,  25,  26,  27, 109,  61, 228,  23, 108],
        [227,  24,  25,  26, 109,  27,  61, 228,  23, 108],
        [227,  24,  25,  26, 109,  27,  61, 228,  23, 108],
        [227,  24,  25,  26, 109,  27,  61, 228,  23,  60],
        [227,  24,  25,  26, 109,  61,  27, 228,  23,  60],
        [227,  24,  25,  26,  61, 109,  27, 228,  23,  60],
        [227,  24,  25,  26,  61, 109,  27, 228,  23,  60],
        [227,  24,  25,  26,  61, 109,  27, 228,  23,  60],
        [227,  24,  25,  26,  61,  27, 109, 228,  23,  60],
        [227,  24,  25,  26,  61,  27, 109, 228,  23,  60],
        [227,  24,  25,  26,  61,  27, 109, 228,  23,  60],
        [227,  24,  25,  26,  61,  27, 109, 228,  23,  60],
        [227,  24,  25,  26,  61,  27, 109, 228,  60,  23],
        [227,  24,  25,  61,  26,  27, 109, 228,  60,  23],
        [227,  24,  25,  61,  26,  27, 228, 109,  60,  23],
        [227,  24,  61,  25,  26,  27, 228, 109,  60,  23],
        [227,  61,  24,  25,  26,  27,  60, 109, 228,  23],
        [ 61, 227,  24,  25,  26, 109,  60,  27, 228,  23],
        [ 61, 227, 109,  24,  60,  25,  26,  27, 108, 228],
        [ 61, 109, 227,  60,  24,  25, 108,  26,  27, 110],
        [ 61, 109,  60, 108, 227,  24, 110,  25,  26,  27],
        [ 61, 109,  60, 108, 227, 110,  24,  25,  26,  27],
        [ 61, 109,  60, 108, 227,  24, 110,  25,  26,  27],
        [ 61, 109,  60, 227,  24,  25, 108,  26, 110,  27],
        [ 61, 109, 227,  60,  24,  25,  26,  27, 108, 228],
        [ 61, 227,  60,  24, 109,  25,  26,  27, 228,  23],
        [ 61, 227,  60,  24,  25,  26, 109,  27, 228,  23],
        [ 61, 227,  60,  24,  25,  26,  27, 109, 228,  23],
        [ 61,  60, 227,  24,  25,  26,  27, 109, 228,  23],
        [ 61,  60, 227, 109,  24,  25,  26,  27, 228, 108],
        [ 61,  60, 109, 227,  24, 108,  25, 110,  26,  27],
        [ 61, 109,  60, 108, 227, 110,  24,  25,  26,  62],
        [ 61, 109,  60, 108, 110, 227,  24,  25,  26, 107],
        [109,  61, 108,  60, 110, 227,  24,  25,  26,  27],
        [109,  61, 227, 108,  24,  25,  60,  26, 110,  27],
        [227, 109,  24,  25,  26,  61,  27, 228, 108,  60],
        [227,  24,  25,  26,  27, 228, 109,  23, 226,  61],
        [227,  24,  25,  26,  27, 228,  23, 226, 109, 229],
        [227,  24,  25,  26,  27, 228, 226,  23, 229, 225],
        [227,  24,  25,  26,  27, 228, 226,  23, 229, 225],
        [227,  24,  25,  26,  27, 228,  23, 226, 229, 225],
        [227,  24,  25,  26, 228,  27,  23, 226, 229, 225],
        [227,  24,  25,  26, 228,  27,  23, 226, 229,  28],
        [227,  24,  25,  26, 228,  27,  23, 226, 229,  28],
        [227,  24,  25,  26, 228,  27,  23, 226, 229,  28],
        [227,  24,  25,  26, 228,  27,  23, 226, 229,  28],
        [227,  24,  25,  26, 228,  27,  23, 226, 229,  28],
        [ 24, 227,  25,  26, 228,  27,  23, 226, 229,  28],
        [ 24, 227,  25,  26, 228,  27,  23, 226, 229,  28],
        [ 24, 227,  25,  26,  27, 228,  23, 226,  28, 229],
        [ 24,  25,  26, 227,  27,  23, 228, 226,  28, 229],
        [227,  24,  25,  26,  27, 228,  23, 226,  28, 229],
        [227,  24,  25, 228,  26,  27,  23, 226, 229,  28],
        [227,  24,  25, 228,  23,  26, 226,  27, 229, 225],
        [ 24,  25,  23,  26,  27,  22, 227, 228, 116,  28],
        [ 24,  23,  25,  26,  22,  27, 116,  76,  16,  15],
        [ 24,  23,  25,  22,  26,  21, 198,  27,  76, 197],
        [236, 235,  24, 234,  23,  25, 233,  26,  22,  27],
        [236, 235, 234, 233,  24,  23,  25,  35,  26,  75],
        [235, 236, 234,  20,  21, 183,  75,  74,  73,  19],
        [ 20,  21, 236, 235,  22,  19,  72,  23,  73,  74],
        [231, 230, 229, 228, 227, 232,  23,  24,  22,  21],
        [231, 232,  23,  24, 230, 233,  22, 229, 234, 236],
        [236, 235, 234, 233, 232, 231,  23,  24, 230,  22],
        [236, 235, 234, 233, 232, 231, 213, 212, 230, 214],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227],
        [235, 236, 234, 233, 232, 231, 230, 229, 228, 227],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227],
        [235, 236, 234, 233, 232, 231, 230, 229, 228, 227],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227],
        [236, 235, 234, 233, 232, 231, 230, 229, 228, 227]
    ]

    # Transform the list into a list of lists of joints that should be in different color in each frame
    joints_color = []
    for i in range(100):
        joints_color_frame = []
        for j in range(10):
            joints_color_frame.append(points_color_indices[i][j] // 3)
        joints_color.append(joints_color_frame)

    data = data.numpy()
    
    x_data = data[:, 0::3]
    y_data = data[:, 1::3]
    z_data = data[:, 2::3]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(min([min(x_data[i]) for i in range(len(x_data))]),
                max([max(x_data[i]) for i in range(len(x_data))]))
    ax.set_ylim(min([min(y_data[i]) for i in range(len(y_data))]),
                max([max(y_data[i]) for i in range(len(y_data))]))
    ax.set_zlim(min([min(z_data[i]) for i in range(len(z_data))]),
                max([max(z_data[i]) for i in range(len(z_data))]))

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    # Initialize an empty scatter plot (to be updated in the animation)
    scatter = ax.scatter([], [], [])

    # Initialize lines
    line_points_indices = [
        (0, 1), (0, 2), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 9), (8, 9), (1, 70),     # Chest and head
        (3, 10), (2, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),                  # Right arm (without hand)
        (2, 40), (3, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45),                  # Left arm (without hand)
        (70, 71), (71, 72), (72, 73), (73, 74),                                              # Right leg
        (70, 75), (75, 76), (76, 77), (77, 78),                                              # Left leg
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),                                    # Right hand, pinky
        (15, 21), (21, 22), (22, 23), (23, 24), (24, 25),                                    # Right hand, ring
        (15, 26), (26, 27), (27, 28), (28, 29), (29, 30),                                    # Right hand, mid
        (15, 31), (31, 32), (32, 33), (33, 34), (34, 35),                                    # Right hand, index
        (15, 36), (36, 37), (37, 38), (38, 39),                                              # Right hand, thumb
        (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),                                    # Left hand, pinky
        (45, 51), (51, 52), (52, 53), (53, 54), (54, 55),                                    # Left hand, ring
        (45, 56), (56, 57), (57, 58), (58, 59), (59, 60),                                    # Left hand, mid
        (45, 61), (61, 62), (62, 63), (63, 64), (64, 65),                                    # Left hand, index
        (45, 66), (66, 67), (67, 68), (68, 69)                                               # Left hand, thumb                
    ]

    # Plus qu'à faire les mains 

    num_lines = len(line_points_indices)
    lines = [ax.plot([], [], [], "-", color="red")[0] for _ in range(num_lines)]

    # Function to update the scatter plot
    def update(frame):    

        # Get the coordinates for the current frame
        frame_coordinates = (x_data[frame], z_data[frame], y_data[frame])

        # Update the scatter plot with new point positions
        scatter._offsets3d = frame_coordinates

        # Set the title to display the current data, label and frame
        ax.set_title(f"Data {i}, Label : {label}, Prediction : {prediction} with model {model} \nFrame: {frame}")

        # Adding lines between the joints
        for line, (start, end) in zip(lines, line_points_indices):
            line.set_data_3d([x_data[frame][start], x_data[frame][end]],
                             [z_data[frame][start], z_data[frame][end]],
                             [y_data[frame][start], y_data[frame][end]])

        # Update the colors of the points based on the joints_color list
        colors = ["red" if idx in joints_color[frame] else "blue" for idx in range(len(x_data[frame]))]
        sizes = [200 if idx in joints_color[frame] else 50 for idx in range(len(x_data[frame]))] # Set larger size for points with red color
        scatter.set_edgecolors(colors)
        scatter.set_facecolors(colors)
        scatter.set_sizes(sizes)

        print(f"scatter={scatter}")
        print(f"lines={lines}")

        return scatter, *lines

    # Create the animation
    print(f"data={data}")
    animation = FuncAnimation(fig, update, frames=len(data), blit=True)
    
    # Save the animation as a GIF
    animation.save(f"/home/self_supervised_learning_gr/self_supervised_learning/dev/ProjetCassiopee/src/visualisation/mocaplab_points_color/{name}.gif",
                   writer='pillow')
    plt.close()


def create_all_animations(results_dir="visualisation_results/mocaplab/supervised"):
    # Begin set-up
    print("#### Set-Up ####")

    # Set-up Python
    setup_python()

    # Set-up PyTorch
    DEVICE = setup_pytorch(gpu=False)

    print("#### Dataset ####")
    dataset = MocaplabDatasetFC(path="data/mocaplab/Cassiopée_Allbones",
                                padding=True, 
                                train_test_ratio=8,
                                validation_percentage=0.01)
    
    print("#### Data Loader ####")
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False)
    
    print("#### Model ####")
    model = MocaplabFC(dataset.max_length * 237)
    model.load_state_dict(torch.load("src/models/mocaplab/fc/saved_models/model_20240325_141951.ckpt", map_location=torch.device('cpu')))
    model = model.to(DEVICE)
    
    print("#### Plot ####")
    for i, sample in enumerate(data_loader) :
        
        print(f"## Sample {i:4} / {len(data_loader)} ##")
        data, label = sample
        data = data.to(DEVICE)
        label = label.to(DEVICE)
    
        data_flattened = data.view(data.size(0), -1)
        output = model(data_flattened.float())
    
        _, predicted = torch.max(output.data, dim=1)

        label = int(label[0])
        predicted = int(predicted[0])
        
        data = data.squeeze(0)
        
        name = f"{i}_{label}_{predicted}"
        create_animation(data, label, predicted, name, type(model))

    print("#### End ####")
    '''

if __name__ == "__main__":
    create_all_animations()