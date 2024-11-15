from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

class RackAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class OutputAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class ChargeStationAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class InputAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class EmptyAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class WarehouseModel(Model):
    def __init__(self, layout=None):
        if layout is None:
            layout = self.DEFAULT_LAYOUT
            
        height = len(layout)
        width = len(layout[0])
        
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        # Create agents based on layout
        for y, row in enumerate(layout):
            for x, cell in enumerate(row):
                agent = None
                if cell == 'R':
                    agent = RackAgent(f"rack_{x}_{y}", self)
                elif cell == 'O':
                    agent = OutputAgent(f"output_{x}_{y}", self)
                elif cell == 'C':
                    agent = ChargeStationAgent(f"charge_{x}_{y}", self)
                elif cell == 'I':
                    agent = InputAgent(f"input_{x}_{y}", self)
                elif cell == ' ' or cell == 'E':
                    agent = EmptyAgent(f"empty_{x}_{y}", self)
                
                if agent:
                    self.grid.place_agent(agent, (x, height-y-1))
                    self.schedule.add(agent)

    DEFAULT_LAYOUT = [
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' , # 1
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' , # 1   
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' , # 1
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' , # 2
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' , # 3
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' , # 4
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' , # 5
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' , # 6
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' , # 7
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' , # 8
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' , # 9
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' , # 10
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 11
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 12
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 13
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 14
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 15
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 16
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 17
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 18
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 19
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 20
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 21
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 22
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 23
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 24
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 25
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 26
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 27
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 28
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 29
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 30
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 31
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERR' ,  # 32
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERR' ,  # 33
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERR' ,  # 34
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERR' ,  # 35
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 36
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 37
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 38
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 39
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 40
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 41
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 42
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 43
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 44
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 45
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 46
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 47
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 48
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 49
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 50
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 51
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 52
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 53
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 54
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 55
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 56
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 57
    'EEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 58
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERR' ,  # 59
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERR' ,  # 60
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERR' ,  # 61
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERR' ,  # 62
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 63
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 64
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 65
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 66
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 67
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 68
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 69
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 70
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 71
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 72
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 73
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 74
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 75
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 76
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 77
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 78
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 79
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 80
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 81
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR' ,  # 82
    'EEEEEEEEEEEEEEEEEEEEEEEEEEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERRRREEERR',# 83
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE', # 84
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE', # 85
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE', # 86
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE', #87
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' #88
    ]

def agent_portrayal(agent):
    if isinstance(agent, RackAgent):
        return {
            "Shape": "rect",
            "Color": "gray",
            "Filled": True,
            "Layer": 0,
            "w": 1,
            "h": 1
        }
    elif isinstance(agent, OutputAgent):
        return {
            "Shape": "rect",
            "Color": "black",
            "Filled": True,
            "Layer": 0,
            "w": 0.8,
            "h": 0.8
        }
    elif isinstance(agent, ChargeStationAgent):
        return {
            "Shape": "rect",
            "Color": "green",
            "Filled": True,
            "Layer": 0,
            "w": 0.8,
            "h": 0.8
        }
    elif isinstance(agent, InputAgent):
        return {
            "Shape": "rect",
            "Color": "yellow",
            "Filled": True,
            "Layer": 0,
            "w": 0.8,
            "h": 0.8
        }
    elif isinstance(agent, EmptyAgent):
        return {
            "Shape": "rect",
            "Color": "white",
            "Filled": True,
            "Layer": 0,
            "w": 1,
            "h": 1
        }
    return {}

# Create visualization
grid = CanvasGrid(agent_portrayal, 113, 90, 904, 720)  # Adjust size based on your layout
server = ModularServer(WarehouseModel,
                      [grid],
                      "Warehouse Layout",
                      {"layout": WarehouseModel.DEFAULT_LAYOUT})

if __name__ == '__main__':
    server.port = 8521
    server.launch()