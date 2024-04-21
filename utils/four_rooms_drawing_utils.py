import pygame


class FourRoomDrawingUtils:
    def __init__(self, grid_size, cell_size=40):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode(
            (grid_size * cell_size, grid_size * cell_size)
        )
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Four Rooms Environment")

    def reset(self):
        # This method can reset any necessary visual components or settings
        self.screen.fill((255, 255, 255))  # Fills the screen with white

    # def draw(self, walls, observation_space, agent_pos, goal_pos):
    # # Clear the screen first
    #     self.screen.fill((255, 255, 255))  # Fills the screen with white

    #     # Draw all cells in the observation space
    #     for pos in observation_space:
    #         rect = pygame.Rect(pos[0] * self.cell_size, pos[1] * self.cell_size, self.cell_size, self.cell_size)
    #         pygame.draw.rect(self.screen, (255, 255, 255), rect)  # Draw observation space in white

    #     # Draw walls
    #     for wall in walls:
    #         rect = pygame.Rect(wall[0] * self.cell_size, wall[1] * self.cell_size, self.cell_size, self.cell_size)
    #         pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Draw walls in black

    #     # Draw the goal
    #     goal_rect = pygame.Rect(goal_pos[0] * self.cell_size, goal_pos[1] * self.cell_size, self.cell_size, self.cell_size)
    #     pygame.draw.rect(self.screen, (0, 255, 0), goal_rect)  # Draw goal in green

    #     # Draw the agent as a triangle
    #     agent_center = (int(agent_pos[0] * self.cell_size + self.cell_size / 2), int(agent_pos[1] * self.cell_size + self.cell_size / 2))
    #     agent_triangle = [
    #         (agent_center[0], agent_center[1] - self.cell_size // 3),
    #         (agent_center[0] - self.cell_size // 3, agent_center[1] + self.cell_size // 3),
    #         (agent_center[0] + self.cell_size // 3, agent_center[1] + self.cell_size // 3)
    #     ]
    #     pygame.draw.polygon(self.screen, (255, 0, 0), agent_triangle)  # Draw agent in red

    #     pygame.display.update()
    #     self.clock.tick(30)

    def draw(self, walls, agent_pos, goal_pos):
        # Draw all cells and gridlines
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                if [x, y] in walls:
                    pygame.draw.rect(
                        self.screen, (0, 0, 0), rect
                    )  # Draw walls in black
                elif (x, y) == tuple(goal_pos):
                    pygame.draw.rect(
                        self.screen, (0, 255, 0), rect
                    )  # Draw goal in green
                else:
                    pygame.draw.rect(
                        self.screen, (255, 255, 255), rect
                    )  # Draw open space in white
                pygame.draw.rect(
                    self.screen, (200, 200, 200), rect, 1
                )  # Draw grid lines in light grey

        # Draw the agent as a triangle
        agent_center = (
            int(agent_pos[0] * self.cell_size + self.cell_size / 2),
            int(agent_pos[1] * self.cell_size + self.cell_size / 2),
        )
        agent_triangle = [
            (agent_center[0], agent_center[1] - self.cell_size // 3),
            (
                agent_center[0] - self.cell_size // 3,
                agent_center[1] + self.cell_size // 3,
            ),
            (
                agent_center[0] + self.cell_size // 3,
                agent_center[1] + self.cell_size // 3,
            ),
        ]
        pygame.draw.polygon(
            self.screen, (255, 0, 0), agent_triangle
        )  # Draw agent in red

        pygame.display.update()
        self.clock.tick(10)  # Maintain 60 FPS

    def close(self):
        pygame.quit()
