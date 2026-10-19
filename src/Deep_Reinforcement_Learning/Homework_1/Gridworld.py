class Gridworld:
    def __init__(self):
        self.grid = [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", "S", ".", "D", "#", ".", "#"],
            ["#", ".", ".", ".", "M", ".", "#"],
            ["#", ".", "#", "#", "#", ".", "#"],
            ["#", ".", "#", "D", "#", ".", "#"],
            ["#", ".", ".", ".", "#", "G", "#"],
            ["#", "#", "#", "#", "#", "#", "#"]
        ]
        # #:=Wall, S:=Start, D:=Death, M:=Mud, G:=Goal
        self.start_position =(1,1)
        self.position = self.start_position
        self.terminated = False

    def reset(self):
        self.position = self.start_position
        self.terminated = False
        return self.position

    def step(self,action):
        move_dic = {
            0: (-1, 0),   # hoch
            1: (0, -1),   # links
            2: (1, 0),    # runter
            3: (0, 1)     # rechts
        }
        if action not in move_dic:
            raise ValueError("action needs to be 0,1,2 or 3")

        #cal new pos
        row,col= self.position
        delta_row,delta_col=move_dic[action]

        new_row = row+delta_row
        new_col= col+ delta_col

        #check if in grid and then move if no wall
        if (0 <= new_row < len(self.grid) and 0 <= new_col < len(self.grid[0])):
            if self.grid[new_row][new_col] != "#":
                self.position = (new_row, new_col)

        #check whats in the position of the grid
        cell = self.grid[self.position[0]][self.position[1]]
        cell_dic = {
            ".": (0, False),
            "S": (0, False),
            "G": (1, True),
            "D": (-1, True),
            "M": (-0.1, False)
        }
        if cell not in cell_dic:
            raise ValueError("Cell kein gültiges symbol")
        reward, self.terminated = cell_dic[cell]



        return self.position, reward, self.terminated

    def render(self):
        for row in range(len(self.grid)):
            for col in range(len(self.grid[row])):
                if(row, col)==self.position:
                    print("A", end=" ")
                else:
                    print(self.grid[row][col], end=" ")

            print()

if __name__ == "__main__":
    import os
    import subprocess

    env = Gridworld()
    S = env.reset()

    G = 0
    gamma = 1
    t = 0

    directions = {"w": 0, "a": 1, "s": 2, "d": 3}

    while True:
        subprocess.run(["cmd", "/c", "cls"] if os.name == "nt" else ["clear"])

        env.render()
        print("Return:", G)

        d = input("Enter Action (w/a/s/d): ").strip()

        if d not in directions:
            continue

        A = directions[d]
        S, R, termination = env.step(A)

        G += gamma ** t * R
        t += 1

        if termination:
            break

    env.render()
    print("Final Return:", G)