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
        self.start_position =(0,0)
        self.positiom = self.start_position
        self.terminated = False