from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from .model import WarehouseModel

def agent_portrayal(agent):
    if agent is None:
        return {}
        
    portrayal = {
        "Shape": "rect",
        "Filled": True,
        "Layer": 0,
        "w": 1,
        "h": 1
    }
    
    if hasattr(agent, "__class__"):
        if agent.__class__.__name__ == "RackAgent":
            portrayal["Color"] = "gray"
        elif agent.__class__.__name__ == "OutputAgent":
            portrayal.update({"Color": "black", "w": 0.8, "h": 0.8})
        elif agent.__class__.__name__ == "ChargeStationAgent":
            portrayal.update({"Color": "green", "w": 0.8, "h": 0.8})
        elif agent.__class__.__name__ == "InputAgent":
            portrayal.update({"Color": "yellow", "w": 0.8, "h": 0.8})
        elif agent.__class__.__name__ == "MovableAgent":
            portrayal.update({"Color": "red", "Layer": 1, "w": 0.8, "h": 0.8})
        elif agent.__class__.__name__ == "EmptyAgent":
            portrayal["Color"] = "white"
            
    return portrayal

grid = CanvasGrid(agent_portrayal, 113, 90, 904, 720)
server = ModularServer(WarehouseModel,
                      [grid],
                      "Warehouse Layout",
                      {"layout": None})

if __name__ == '__main__':
    server.port = 8521
    server.launch()