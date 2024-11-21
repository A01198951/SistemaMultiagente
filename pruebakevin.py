from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from heapq import heappush, heappop
from typing import List, Tuple, Dict, Set
import math
import random
from enum import Enum
from typing import Tuple, Optional

class TaskType(Enum):
    STORE = "STORE"         # Input -> Rack
    RETRIEVE = "RETRIEVE"   # Rack -> Output

class Task:
    def __init__(self, task_type: TaskType, source_pos: Tuple[int, int], 
                 dest_pos: Tuple[int, int], pallet_id: Optional[int] = None):
        self.task_type = task_type
        self.source_pos = source_pos
        self.dest_pos = dest_pos
        self.pallet_id = pallet_id
        self.assigned_forklift = None
        self.completed = False
        
    def __str__(self):
        return f"Task({self.task_type}, from={self.source_pos}, to={self.dest_pos})"
    
class TaskManager:
    def __init__(self, model):
        self.model = model
        self.pending_tasks = []
        self.active_tasks = []
        self.generate_initial_tasks()  # Añadir generación inicial de tareas

    def debug_print_status(self):
        print("\n=== Task Manager Status ===")
        print(f"Pending tasks: {len(self.pending_tasks)}")
        for task in self.pending_tasks:
            print(f"- {task}")
        print(f"Active tasks: {len(self.active_tasks)}")
        for task in self.active_tasks:
            print(f"- {task}")
    
    def assign_charge_task(self, forklift):
            # Encontrar la estación de carga más cercana
            nearest_station = min(
                self.model.CHARGE_STATIONS,
                key=lambda pos: abs(forklift.pos[0] - pos[0]) + abs(forklift.pos[1] - pos[1])
            )
            charge_task = Task(TaskType.STORE, forklift.pos, nearest_station)
            self.active_tasks.append(charge_task)
            return charge_task

    def generate_initial_tasks(self):
        # Usar posiciones fijas del modelo
        input_positions = self.model.INPUT_POSITIONS
        output_positions = self.model.OUTPUT_POSITIONS
        rack_positions = self.model.RACK_POSITIONS

        total_tasks = 10  # Total de tareas a generar

        for i in range(total_tasks):
            if i % 2 == 0:  # Alternar entre tareas STORE y RETRIEVE
                # Generar tarea STORE
                source = random.choice(input_positions)
                # Buscar racks vacíos
                dest_candidates = [pos for pos in rack_positions 
                                    if not any(rack.has_pallet for rack in self.model.rack_agents 
                                            if rack.pos == pos)]
                if dest_candidates:
                    dest = random.choice(dest_candidates)
                    store_task = Task(TaskType.STORE, source, dest)
                    self.pending_tasks.append(store_task)
                else:
                    print("No available racks for STORE task. Skipping.")
            else:
                # Generar tarea RETRIEVE
                source_candidates = [pos for pos in rack_positions 
                                    if any(rack.has_pallet for rack in self.model.rack_agents 
                                            if rack.pos == pos)]
                if source_candidates:
                    source = random.choice(source_candidates)
                    dest = random.choice(output_positions)
                    retrieve_task = Task(TaskType.RETRIEVE, source, dest)
                    self.pending_tasks.append(retrieve_task)
                else:
                    print("No available racks with pallets for RETRIEVE task. Skipping.")


    def assign_task(self, forklift):
        if not self.pending_tasks:
            return None
            
        # Only assign task if forklift is at a valid starting position
        if forklift.pos in self.model.FORKLIFT_POSITIONS:
            task = self.pending_tasks.pop(0)
            task.assigned_forklift = forklift
            self.active_tasks.append(task)
            return task
        return None

    def complete_task(self, task):
        if task in self.active_tasks:
            self.active_tasks.remove(task)
            task.completed = True
            print(f"Task {task} marked as completed.")


class WarehouseModel(Model):
    def step(self):
        print("\n=== Warehouse Step ===")
        self.task_manager.debug_print_status()
        self.schedule.step()

class RackAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.has_pallet = random.choice([True, False])
        self.pallet_id = self.unique_id if self.has_pallet else None

    def store_pallet(self, pallet_id=None):
        if not self.has_pallet:
            self.has_pallet = True
            self.pallet_id = pallet_id
            return True
        return False

    def remove_pallet(self):
        if self.has_pallet:
            pallet_id = self.pallet_id
            self.has_pallet = False  # Set to False when pallet is taken
            self.pallet_id = None
            return pallet_id
        return None

    def is_available(self):
        return not self.has_pallet

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

class PathFinder:
    @staticmethod
    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(b[0] - a[0]) + abs(b[1] - a[1])

   
    @staticmethod
    def is_valid_position(pos: Tuple[int, int], grid, height, width) -> bool:
        # Check boundaries
        if not (0 <= pos[0] < width and 0 <= pos[1] < height):
            return False

        # Check for other forklifts
        cell_contents = grid.get_cell_list_contents(pos)
        for agent in cell_contents:
            # Evitar celdas ocupadas por otros montacargas
            if isinstance(agent, ForkliftAgent):
                return False

        # Permitir entrada en celdas con InputAgent o OutputAgent
        for agent in cell_contents:
            if isinstance(agent, (InputAgent, OutputAgent)):
                return True

        # Default: allow movement to empty cells
        return True



    @staticmethod
    def is_adjacent_to_target(pos: Tuple[int, int], target: Tuple[int, int], grid) -> bool:
        """Verifica si la posición actual es adyacente o igual al objetivo."""
        dx = abs(pos[0] - target[0])
        dy = abs(pos[1] - target[1])
        return dx == 0 and dy == 0  # Solo marca como alcanzado si es la misma posición



    @staticmethod
    def get_neighbors(pos: Tuple[int, int], grid, height, width) -> List[Tuple[int, int]]:
        # Only use orthogonal movements (no diagonals)
        neighbors = []
        movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    
        for dx, dy in movements:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if PathFinder.is_valid_position(new_pos, grid, height, width):
                neighbors.append(new_pos)
        return neighbors

    @staticmethod
    def find_alternative_path(start: Tuple[int, int], goal: Tuple[int, int], 
                            grid, height, width) -> List[Tuple[int, int]]:
        """Find path to adjacent position if goal is blocked"""
        adjacent_positions = PathFinder.get_adjacent_positions(goal, grid, height, width)
        
        best_path = []
        min_dist = float('inf')
        
        for adj_pos in adjacent_positions:
            path = PathFinder.a_star(start, adj_pos, grid, height, width)
            if path and len(path) < min_dist:
                best_path = path
                min_dist = len(path)
                
        return best_path

    @staticmethod
    def a_star(start: Tuple[int, int], goal: Tuple[int, int], grid, height, width) -> List[Tuple[int, int]]:
        
        frontier = []
        heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heappop(frontier)[1]
            
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for next_pos in PathFinder.get_neighbors(current, grid, height, width):
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + PathFinder.heuristic(next_pos, goal)
                    heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        return []  # No path found
    
    
class MovableAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.path = []
        self.goal = None
        self.pathfinder = PathFinder()

    def find_random_empty_cell(self) -> Tuple[int, int]:
        empty_cells = []
        for cell in self.model.grid.coord_iter():
            cell_content, pos = cell
            if all(isinstance(agent, EmptyAgent) for agent in cell_content):
                empty_cells.append(pos)
        return self.random.choice(empty_cells) if empty_cells else None

    def step(self):
        # Si no hay objetivo o se alcanzó, busca uno nuevo
        if not self.goal or self.pos == self.goal:
            self.goal = self.find_random_empty_cell()
            if self.goal:
                self.path = self.pathfinder.a_star(
                    self.pos,
                    self.goal,
                    self.model.grid,
                    self.model.grid.height,
                    self.model.grid.width
                )

        # Si hay un camino, sigue avanzando
        if self.path and len(self.path) > 1:
            next_pos = self.path[1]
            cell_contents = self.model.grid.get_cell_list_contents(next_pos)

            # Permitir movimiento si la celda contiene InputAgent o OutputAgent
            if not cell_contents or all(isinstance(agent, (EmptyAgent, RackAgent, InputAgent, OutputAgent, ChargeStationAgent)) for agent in cell_contents):
                self.model.grid.move_agent(self, next_pos)
                self.path = self.path[1:]
            else:
                # Recalcular ruta si está bloqueado
                self.path = self.pathfinder.a_star(
                    self.pos,
                    self.goal,
                    self.model.grid,
                    self.model.grid.height,
                    self.model.grid.width
                )


class ForkliftAgent(MovableAgent):
    def __init__(self, unique_id, model, starting_position):
        super().__init__(unique_id, model)
        self.carrying_pallet = False
        self.current_task = None
        self.current_pallet_id = None
        self.task_state = "IDLE"  # IDLE, MOVING_TO_SOURCE, LOADING, MOVING_TO_DEST, UNLOADING, CHARGING
        self.starting_position = starting_position  # Posición inicial del agente
        self.battery_level = 100  # Nivel inicial de batería
        self.reward = 0  # Sistema de recompensas inicializado en 0

    def step(self):
        print(f"\n=== Forklift {self.unique_id} Status ===")
        print(f"Position: {self.pos}")
        print(f"Carrying Pallet: {self.carrying_pallet}")
        print(f"Battery Level: {self.battery_level}%")
        print(f"Reward: {self.reward}")
        print(f"Current Task: {self.current_task.task_type if self.current_task else 'None'}")
        print(f"Task State: {self.task_state}")

        # Simulación del consumo de batería
        self.battery_level -= 2

        # Penalizar si el agente está atascado o no tiene camino
        if not self.path:
            print(f"Forklift {self.unique_id} is stuck or has no path. Penalizing.")
            self.reward -= 10

        # Si la batería está baja, asignar una tarea de carga
        if self.battery_level < 50 and self.task_state != "CHARGING":
            self.current_task = self.model.task_manager.assign_charge_task(self)
            if self.current_task:
                self.task_state = "CHARGING"
                self.goal = self.current_task.dest_pos
                self.path = self.pathfinder.a_star(
                    self.pos,
                    self.goal,
                    self.model.grid,
                    self.model.grid.height,
                    self.model.grid.width
                )

        # Manejo de la tarea de cargar batería
        if self.task_state == "CHARGING":
            if self.pos == self.current_task.dest_pos:
                print(f"Forklift {self.unique_id} reached charging station at {self.pos}")
                self.battery_level = 100  # Recargar batería
                self.reward += 20  # Recompensar por cargar batería
                self.task_state = "RETURNING_HOME"
                self.current_task = None

        # Asignar nuevas tareas cuando está en estado IDLE
        if self.task_state == "IDLE":
            self.current_task = self.model.task_manager.assign_task(self)
            if self.current_task:
                print(f"Got new task: {self.current_task}")
                self.task_state = "MOVING_TO_SOURCE"
                self.goal = self.current_task.source_pos
                self.path = self.pathfinder.a_star(
                    self.pos,
                    self.goal,
                    self.model.grid,
                    self.model.grid.height,
                    self.model.grid.width
                )
            else:
                print("No tasks available - staying idle")
                return

        # Regresar a la posición inicial después de cargar la batería o completar una tarea
        if self.task_state == "RETURNING_HOME":
            if self.pos == self.starting_position:
                print(f"Forklift {self.unique_id} returned to starting position {self.starting_position}")
                self.task_state = "IDLE"
                self.goal = None
                self.reward += 10  # Recompensar por regresar a la posición inicial
            else:
                self.goal = self.starting_position
                self.path = self.pathfinder.a_star(
                    self.pos,
                    self.goal,
                    self.model.grid,
                    self.model.grid.height,
                    self.model.grid.width
                )

        # Realizar la tarea y actualizar recompensas
        if self.current_task or self.task_state == "RETURNING_HOME":
            self._execute_current_task()
            if self.path:  # Solo moverse si hay un camino
                super().step()

    def _execute_current_task(self):
        # Verificar si la tarea actual es válida
        if not self.current_task:
            print(f"Forklift {self.unique_id} has no current task to execute.")
            return

        if not self.path:
            # Recalcular el camino si no hay uno activo
            if self.task_state == "MOVING_TO_SOURCE":
                self.goal = self.current_task.source_pos
            elif self.task_state == "MOVING_TO_DEST":
                self.goal = self.current_task.dest_pos

            self.path = self.pathfinder.a_star(
                self.pos,
                self.goal,
                self.model.grid,
                self.model.grid.height,
                self.model.grid.width
            )
            print(f"Forklift {self.unique_id} recalculated path: {self.path}")

        if self.path:
            # Verificar si llegó al origen de la tarea
            if self.pathfinder.is_adjacent_to_target(self.pos, self.current_task.source_pos, self.model.grid) and self.task_state == "MOVING_TO_SOURCE":
                print(f"Forklift {self.unique_id} reached source position {self.current_task.source_pos}")
                success = self._handle_loading()
                if success:
                    self.task_state = "MOVING_TO_DEST"
                    self.goal = self.current_task.dest_pos
                    self.path = None  # Forzar recálculo del camino

            # Verificar si llegó al destino de la tarea
            elif self.pathfinder.is_adjacent_to_target(self.pos, self.current_task.dest_pos, self.model.grid) and self.task_state == "MOVING_TO_DEST":
                print(f"Forklift {self.unique_id} reached destination position {self.current_task.dest_pos}")
                success = self._handle_unloading()
                if success:
                    print(f"Forklift {self.unique_id} completed task and is returning home.")
                    self.task_state = "RETURNING_HOME"
                    self.current_task = None
                    self.path = None  # Forzar recálculo para regresar

    def _handle_loading(self):
        print(f"Forklift {self.unique_id} attempting to load at {self.pos}")
        if self.pos != self.current_task.source_pos:
            print(f"Forklift {self.unique_id} is not at the correct source position: {self.current_task.source_pos}")
            return False

        # Interactuar con los agentes en la celda actual
        cell_contents = self.model.grid.get_cell_list_contents(self.pos)
        for agent in cell_contents:
            if self.current_task.task_type == TaskType.STORE and isinstance(agent, InputAgent):
                self.carrying_pallet = True
                print(f"Forklift {self.unique_id} picked up a pallet from InputAgent at {self.pos}")
                self.reward += 15  # Recompensar por cargar un pallet
                return True
            elif self.current_task.task_type == TaskType.RETRIEVE and isinstance(agent, RackAgent):
                if agent.has_pallet:
                    pallet_id = agent.remove_pallet()
                    if pallet_id:
                        self.current_pallet_id = pallet_id
                        self.carrying_pallet = True
                        print(f"Forklift {self.unique_id} picked up pallet {pallet_id} from RackAgent at {self.pos}")
                        self.reward += 15  # Recompensar por cargar un pallet
                        return True

        print(f"Forklift {self.unique_id} failed to load at {self.pos}")
        self.reward -= 5  # Penalizar por fallar al cargar
        return False

    def _handle_unloading(self):
        print(f"Forklift {self.unique_id} attempting to unload at {self.pos}")
        cell_contents = self.model.grid.get_cell_list_contents(self.pos)
        for agent in cell_contents:
            if self.current_task.task_type == TaskType.STORE and isinstance(agent, RackAgent):
                if not agent.has_pallet:
                    if agent.store_pallet(self.current_pallet_id):
                        print(f"Forklift {self.unique_id} stored pallet {self.current_pallet_id} in RackAgent at {self.pos}")
                        self.current_pallet_id = None
                        self.carrying_pallet = False
                        self.reward += 20  # Recompensar por descargar un pallet
                        return True
            elif self.current_task.task_type == TaskType.RETRIEVE and isinstance(agent, OutputAgent):
                print(f"Forklift {self.unique_id} delivered pallet {self.current_pallet_id} to OutputAgent at {self.pos}")
                self.current_pallet_id = None
                self.carrying_pallet = False
                self.reward += 20  # Recompensar por entregar un pallet
                return True

        print(f"Forklift {self.unique_id} failed to unload at {self.pos}")
        self.reward -= 10  # Penalizar por fallar al descargar
        return False



class WarehouseModel(Model):
    # Definir posiciones fijas como constantes de clase
    INPUT_POSITIONS = [(0, 1)]  # Posiciones I en el layout
    OUTPUT_POSITIONS = [(7, 9), (8, 9)]  # Posiciones O en el layout
    RACK_POSITIONS = [
        (1, 7), (1, 6), (1, 5),  
        (2, 7), (2, 6), (2, 5),  
        (4, 7), (4, 6), (4, 5),
        (5, 7), (5, 6), (5, 5),
        (7, 7), (7, 6), (7, 5),
        (8, 7), (8, 6), (8, 5)  
    ]
    FORKLIFT_POSITIONS = [(4, 1), (6, 1), (4, 2), (6, 2)]  # New forklift starting positions
    CHARGE_STATIONS = [(8, 0), (9, 0)]

    def __init__(self, layout=None):
        super().__init__()  # Importante: inicializar Model
        if layout is None:
            layout = self.DEFAULT_LAYOUT
            
        height = len(layout)
        width = len(layout[0])
        
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        # Create empty spaces first
        for x in range(width):
            for y in range(height):
                empty = EmptyAgent(f"empty_{x}_{y}", self)
                self.grid.place_agent(empty, (x, y))
                self.schedule.add(empty)

        self.input_agents = []
        self.output_agents = []
        self.rack_agents = []
        self.forklift_agents = []
        self.charge_stations = []

        self.charge_stations = []
        for pos in self.CHARGE_STATIONS:
            agent = ChargeStationAgent(f"charge_{pos[0]}_{pos[1]}", self)
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
            self.charge_stations.append(agent)

        for pos in self.INPUT_POSITIONS:
            agent = InputAgent(f"input_{pos[0]}_{pos[1]}", self)
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
            self.input_agents.append(agent)
            
        for pos in self.OUTPUT_POSITIONS:
            agent = OutputAgent(f"output_{pos[0]}_{pos[1]}", self)
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
            self.output_agents.append(agent)
            
        for pos in self.RACK_POSITIONS:
            agent = RackAgent(f"rack_{pos[0]}_{pos[1]}", self)
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
            self.rack_agents.append(agent)
        
        for pos in self.FORKLIFT_POSITIONS:
            agent = ForkliftAgent(f"forklift_{pos[0]}_{pos[1]}", self, starting_position=pos)
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
            self.forklift_agents.append(agent)

        self.task_manager = TaskManager(self)

    def step(self):
        print("\n=== Warehouse Step ===")
        self.task_manager.debug_print_status()
        self.schedule.step()

    DEFAULT_LAYOUT = [
    'EEEEEEEOOE',  # 1
    'EEEEEEEEEE',  # 2
    'ERRERRERRE',  # 3
    'ERRERRERRE',  # 4
    'ERRERRERRE',  # 5
    'EEEEEEEEEE',  # 6
    'EEEEEEEEEE',  # 7
    'EEEEEAEAEE',  # 8
    'IEEEEAEAEE',  # 9
    'EEEECCEEEE'   # 10
]

def agent_portrayal(agent):
    if isinstance(agent, RackAgent):
        color = "brown" if agent.has_pallet else "gray"
        return {
            "Shape": "rect",
            "Color": color,
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
            "Shape": "circle",
            "Color": "green",
            "Filled": True,
            "Layer": 0,
            "r": 0.5
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
    elif isinstance(agent, MovableAgent):
        return {
            "Shape": "rect",
            "Color": "red",
            "Filled": True,
            "Layer": 1,
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
grid = CanvasGrid(agent_portrayal, 10, 10, 400, 400)  
server = ModularServer(WarehouseModel,
                      [grid],
                      "Warehouse Layout",
                      {"layout": None})

if __name__ == '__main__':
    server.port = 8521
    server.launch()