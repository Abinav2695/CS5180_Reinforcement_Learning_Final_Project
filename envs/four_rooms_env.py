import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from utils.four_rooms_drawing_utils import FourRoomDrawingUtils


class LargeFourRooms(gym.Env):
    def __init__(self, grid_size=20, max_steps_per_episode=1000, mode="None"):
        super(LargeFourRooms, self).__init__()
        # Define the grid size
        self.grid_size = grid_size
        self.max_steps_per_episode = max_steps_per_episode
        self.mode = mode

        self.grid = np.ones((self.grid_size, self.grid_size))
        # Define freespace (value 0)
        half_value = grid_size // 2
        box_start_loc_row = (grid_size // 4) * 3
        box_start_loc_col = grid_size // 4

        ## Lower Half
        self.grid[:half_value, : half_value - 1] = 0
        self.grid[: half_value - 1, half_value:] = 0

        ## Upper Half
        self.grid[half_value + 1 :, :half_value] = 0
        self.grid[half_value:, half_value + 1 :] = 0

        ## box
        self.grid[
            box_start_loc_row : box_start_loc_row + 2,
            box_start_loc_col - 2 : box_start_loc_col + 2,
        ] = 1

        ## Doors
        self.grid[half_value, half_value//2] = 0
        self.grid[half_value-1, 3*half_value//2] = 0
        self.grid[half_value//2, half_value-1] = 0
        self.grid[(3*half_value//2) + 2, half_value] = 0

        self.walls = np.argwhere(self.grid == 1.0).tolist()  # find all wall cells
        self.walls = self.arr_coords_to_four_room_coords(self.grid_size, self.walls)
        # print(f"Grid :: {self.grid}")
        # print(f"Wall locations :: {self.walls}")
        # Define the observation space
        self.observation_space = np.argwhere(self.grid == 0).tolist()
        self.observation_space = self.arr_coords_to_four_room_coords(
            self.grid_size, self.observation_space
        )
        # print(f"Size :: {self.observation_space.shape}")

        # Define the action space
        self.action_space = spaces.Discrete(4)  # Up, down, left, right

        # Start and goal locations
        self.start_location = np.array([0, 0])
        self.goal_location = np.array([self.grid_size - 1, self.grid_size - 1])

        # Initialize state
        self.agent_pos = None
        self.action = None
        self.t = 0
        self.drawing_utils = None
        if self.mode == "human":
            self.drawing_utils = FourRoomDrawingUtils(self.grid_size)

    @staticmethod
    def arr_coords_to_four_room_coords(grid_size, arr_coords_list):
        """
        Function converts the array coordinates ((row, col), origin is top left)
        to the Four Rooms coordinates ((x, y), origin is bottom left)
        E.g., The coordinates (0, 0) in the numpy array is mapped to (0, 10) in the Four Rooms coordinates.
        Args:
            arr_coords_list (list): List variable consisting of tuples of locations in the numpy array

        Return:
            four_room_coords_list (list): List variable consisting of tuples of converted locations in the
                                          Four Rooms environment.
        """
        # Flip the coordinates from (row_idx, column_idx) -> (y, x),
        # where x = column_idx, y = row_idx
        four_room_coords_list = [
            (column_idx, grid_size - row_idx - 1)
            for (row_idx, column_idx) in arr_coords_list
        ]
        return np.array(four_room_coords_list)

    def reset(self):
        # Reset the agent's location to the start location
        self.agent_pos = self.start_location

        # Reset the timeout tracker to be 0
        self.t = 0
        if isinstance(self.drawing_utils, FourRoomDrawingUtils):
            self.drawing_utils.reset()
        # Reset the information
        info = {"message": "Environment reset."}
        return self.agent_pos, info

    def step(self, action):
        if np.random.uniform() < 0.2:
            if action == 2 or action == 3:
                action = np.random.choice([0, 1], 1)[0]
            else:
                action = np.random.choice([2, 3], 1)[0]

        movements = {
            0: np.array([0, 1]),  # up
            1: np.array([0, -1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([1, 0]),  # right
        }
        reward = 0

        proposed_location = np.array(self.agent_pos) + movements[action]
        # proposed_location = np.array([0,8])
        if (0 <= proposed_location[0] < self.grid_size) and (
            0 <= proposed_location[1] < self.grid_size
        ):
            if not np.any(np.all(self.walls == proposed_location, axis=1)):
                self.agent_pos = proposed_location
        else:
            reward = -1

        reward += -0.1

        terminated = False
        truncated = False
        info = {}

        if (np.all(proposed_location == self.goal_location, axis=0)):
            reward+=100
            terminated = True
            info["reason"] = "Goal reached"
            # print("Goal Reached")
        elif self.t == self.max_steps_per_episode:
            truncated = True
            info["reason"] = "Max steps reached"

        self.t += 1
        self.action = action
        # print(f"Action :: {self.action} ... New Pos :: {self.agent_pos}")
        return self.agent_pos, reward, terminated, truncated, info

    def render(self):
        if self.mode == "human":
            self.drawing_utils.draw(
                self.walls.tolist(), self.agent_pos, self.goal_location
            )

    def close(self):
        self.drawing_utils.close()
