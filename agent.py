# Simulacion Python
# Equipo 6

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
import json
import matplotlib.pyplot as plt

class TaskType(Enum):
    STORE = "STORE"         
    RETRIEVE = "RETRIEVE"   

class Task:
    def __init__(self, task_type: TaskType, source_pos: Tuple[int, int], 
                 dest_pos: Tuple[int, int], pallet_id: Optional[int] = None):
        self.task_type = task_type
        self.source_pos = source_pos
        self.dest_pos = dest_pos
        self.pallet_id = pallet_id
        self.assigned_forklift = None
        self.completed = False
        self.assigned_time = None  
        self.completed_time = None  
        self.created_time = None

    def __eq__(self, other):
        if not isinstance(other, Task):
            return False
        return (self.task_type == other.task_type and 
                self.source_pos == other.source_pos and 
                self.dest_pos == other.dest_pos)

    def __hash__(self):
        return hash((self.task_type, self.source_pos, self.dest_pos))    

    def __str__(self):
        return f"Task({self.task_type}, from={self.source_pos}, to={self.dest_pos})"
    
class TaskManager:
    def __init__(self, model):
        self.model = model
        self.pending_tasks = []
        self.active_tasks = []
        self.generate_initial_tasks()  

    def dRBug_print_status(self):
        print("\n=== Task Manager Status ===")
        print(f"Pending tasks: {len(self.pending_tasks)}")
        for task in self.pending_tasks:
            print(f"- {task}")
        print(f"Active tasks: {len(self.active_tasks)}")
        for task in self.active_tasks:
            print(f"- {task}")
    
    def assign_charge_task(self, forklift):
        charge_station_positions = [agent.pos for agent in self.model.charge_stations]
        nearest_station = min(
            charge_station_positions,
            key=lambda pos: abs(forklift.pos[0] - pos[0]) + abs(forklift.pos[1] - pos[1])
        )
        charge_task = Task(TaskType.STORE, forklift.pos, nearest_station)
        return charge_task


    def generate_initial_tasks(self):
        input_positions = [agent.pos for agent in self.model.input_agents]
        output_positions = [agent.pos for agent in self.model.output_agents]

        racks_with_pallets = [agent.pos for agent in self.model.rack_agents if agent.has_pallet]
        empty_racks = [agent.pos for agent in self.model.rack_agents if not agent.has_pallet]

        total_tasks = 50
        tasks_set = set()

        for i in range(total_tasks):
            if i % 2 == 0:  
                if not empty_racks:
                    print("No hay racks vacíos disponibles para tareas de almacenamiento.")
                    continue
                source_pos = random.choice(input_positions)
                dest_pos = random.choice(empty_racks)
                task = Task(TaskType.STORE, source_pos, dest_pos)
                empty_racks.remove(dest_pos)  
            else:  
                if not racks_with_pallets:
                    print("No hay racks con pallets disponibles para tareas de recuperación.")
                    continue
                source_pos = random.choice(racks_with_pallets)
                dest_pos = random.choice(output_positions)
                task = Task(TaskType.RETRIEVE, source_pos, dest_pos)
                racks_with_pallets.remove(source_pos)

            if task not in tasks_set:
                task.created_time = self.model.total_simulation_time
                tasks_set.add(task)
                self.pending_tasks.append(task)


    def assign_task(self, forklift):
        if not self.pending_tasks and not self.active_tasks:
            return None

        if forklift.pos == forklift.starting_position:
            viable_task = None
            for task in self.pending_tasks:
                if task.task_type == TaskType.RETRIEVE:
                    rack_agent = self.model.get_agent_at_position(task.source_pos)
                    if isinstance(rack_agent, RackAgent) and rack_agent.has_pallet:
                        viable_task = task
                        break
                elif task.task_type == TaskType.STORE:
                    rack_agent = self.model.get_agent_at_position(task.dest_pos)
                    if isinstance(rack_agent, RackAgent) and not rack_agent.has_pallet:
                        viable_task = task
                        break

            if viable_task:
                self.pending_tasks.remove(viable_task)
                viable_task.assigned_forklift = forklift
                viable_task.assigned_time = self.model.schedule.time
                self.active_tasks.append(viable_task)
                return viable_task
            else:
                print("No hay tareas viables disponibles para asignar.")
                return None
        return None

    def complete_task(self, task):
        if task in self.active_tasks:
            self.active_tasks.remove(task)
            task.completed = True
            task.completed_time = self.model.schedule.time  
            self.model.completed_tasks.append(task)  
            print(f"Task {task} marked as completed.")

class BlockAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

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
            self.has_pallet = False  
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
        if not (0 <= pos[0] < width and 0 <= pos[1] < height):
            return False

        cell_contents = grid.get_cell_list_contents(pos)
        for agent in cell_contents:
            if isinstance(agent, (ForkliftAgent, BlockAgent)):
                return False  

        return True  

    @staticmethod
    def get_neighbors(pos: Tuple[int, int], grid, height, width) -> List[Tuple[int, int]]:
        neighbors = []
        movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    
        for dx, dy in movements:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if PathFinder.is_valid_position(new_pos, grid, height, width):
                neighbors.append(new_pos)
        return neighbors

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

        return []  
    
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

        if self.path and len(self.path) > 1:
            next_pos = self.path[1]
            cell_contents = self.model.grid.get_cell_list_contents(next_pos)

            if not cell_contents or all(isinstance(agent, (EmptyAgent, RackAgent, InputAgent, OutputAgent, ChargeStationAgent)) for agent in cell_contents):
                self.model.grid.move_agent(self, next_pos)
                self.path = self.path[1:]
            else:
                self.path = self.pathfinder.a_star(
                    self.pos,
                    self.goal,
                    self.model.grid,
                    self.model.grid.height,
                    self.model.grid.width
                )

class ForkliftAgent(MovableAgent):
    all_movement_sequences = {}

    def __init__(self, unique_id, model, starting_position):
        super().__init__(unique_id, model)
        self.carrying_pallet = False
        self.current_task = None
        self.current_pallet_id = None
        self.task_state = "IDLE" 
        self.starting_position = starting_position  
        self.battery_level = 100 
        self.reward = 0  
        self.movement_sequence = []
        self.time_on_task = 0  

    def step(self):
        print(f"\n=== Forklift {self.unique_id} Status ===")
        print(f"Position: {self.pos}")
        print(f"Carrying Pallet: {self.carrying_pallet}")
        print(f"Battery Level: {self.battery_level}%")
        print(f"Reward: {self.reward}")
        print(f"Current Task: {self.current_task.task_type if self.current_task else 'None'}")
        print(f"Task State: {self.task_state}")

        if self.pos:
            self.movement_sequence.append({
                "x": self.pos[0],
                "y": self.pos[1]
            })
            self.save_path_to_json()

        self.battery_level -= 0.1

        if not self.path and self.pos != self.goal:
            print(f"Forklift {self.unique_id} is stuck or has no path. Penalizing.")
            self.reward -= 10

        needs_urgent_charging = self.battery_level < 20
        if needs_urgent_charging:
            print(f"¡Batería crítica ({self.battery_level}%)! Forzando carga...")
        
        if (self.battery_level < 50 and self.task_state == "IDLE") or needs_urgent_charging:
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

        if self.task_state == "CHARGING":
            if self.pos == self.current_task.dest_pos:
                print(f"Forklift {self.unique_id} reached charging station at {self.pos}")
                self.battery_level = 100 
                self.reward += 20  
                self.task_state = "RETURNING_HOME"
                self.current_task = None

        if self.task_state == "IDLE":
            self.time_on_task += self.model.step_duration_seconds
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

        if self.task_state == "RETURNING_HOME":
            if self.pos == self.starting_position:
                print(f"Forklift {self.unique_id} returned to starting position {self.starting_position}")
                self.task_state = "IDLE"
                self.goal = None
                self.reward += 10  
            else:
                self.goal = self.starting_position
                self.path = self.pathfinder.a_star(
                    self.pos,
                    self.goal,
                    self.model.grid,
                    self.model.grid.height,
                    self.model.grid.width
                )

        if self.current_task or self.task_state == "RETURNING_HOME" or self.task_state == "CHARGING":
            self._execute_current_task()
            if self.path:  
                super().step()

    def _execute_current_task(self):
        if not self.current_task and self.task_state not in ["RETURNING_HOME", "CHARGING"]:
            print(f"Forklift {self.unique_id} has no current task to execute.")
            return

        if not self.path:
            if self.task_state == "MOVING_TO_SOURCE":
                self.goal = self.current_task.source_pos
            elif self.task_state == "MOVING_TO_DEST":
                self.goal = self.current_task.dest_pos
            elif self.task_state == "RETURNING_HOME":
                self.goal = self.starting_position
            elif self.task_state == "CHARGING":
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
            if self.pos == self.goal:
                if self.task_state == "MOVING_TO_SOURCE":
                    print(f"Forklift {self.unique_id} reached source position {self.current_task.source_pos}")
                    success = self._handle_loading()
                    if success:
                        self.task_state = "MOVING_TO_DEST"
                        self.goal = self.current_task.dest_pos
                        self.path = None  

                elif self.task_state == "MOVING_TO_DEST":
                    print(f"Forklift {self.unique_id} reached destination position {self.current_task.dest_pos}")
                    success = self._handle_unloading()
                    if success:
                        print(f"Forklift {self.unique_id} completed task and is returning home.")
                        self.task_state = "RETURNING_HOME"
                        self.model.task_manager.complete_task(self.current_task)
                        self.current_task = None
                        self.path = None  

                elif self.task_state == "RETURNING_HOME":
                    print(f"Forklift {self.unique_id} returned to starting position {self.starting_position}")
                    self.task_state = "IDLE"
                    self.goal = None
                    self.reward += 10  

                elif self.task_state == "CHARGING":
                    print(f"Forklift {self.unique_id} reached charging station at {self.pos}")
                    self.battery_level = 100 
                    self.reward += 20  
                    self.task_state = "RETURNING_HOME"
                    self.current_task = None
                    self.path = None

    def save_path_to_json(self):
        """Save movement sequence to JSON"""
        filename = f'forklift_paths.json'
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        positions = [[pos['x'], pos['y']] for pos in self.movement_sequence]
        
        agent_found = False
        for agent in data:
            if agent['id'] == self.unique_id:
                agent['positions'] = positions
                agent_found = True
                break
        
        if not agent_found:
            data.append({
                'id': self.unique_id,
                'positions': positions
            })

        data.sort(key=lambda x: x['id'])

        formatted_json = '[\n' + ',\n'.join(json.dumps(agent) for agent in data) + '\n]'
    
        with open(filename, 'w') as f:
            f.write(formatted_json)

    def _handle_loading(self):
        print(f"Forklift {self.unique_id} attempting to load at {self.pos}")
        cell_contents = self.model.grid.get_cell_list_contents(self.pos)
        for agent in cell_contents:
            if self.current_task.task_type == TaskType.STORE and isinstance(agent, InputAgent):
                self.carrying_pallet = True
                print(f"Forklift {self.unique_id} picked up a pallet from InputAgent at {self.pos}")
                self.reward += 15  
                at_time = self.model.total_simulation_time - self.current_task.created_time
                self.model.total_accumulation_time += at_time  
                return True
            elif self.current_task.task_type == TaskType.RETRIEVE and isinstance(agent, RackAgent):
                if agent.has_pallet:
                    pallet_id = agent.remove_pallet()
                    if pallet_id:
                        self.current_pallet_id = pallet_id
                        self.carrying_pallet = True
                        print(f"Forklift {self.unique_id} picked up pallet {pallet_id} from RackAgent at {self.pos}")
                        self.reward += 15 
                        return True

        print(f"Forklift {self.unique_id} failed to load at {self.pos}")
        self.reward -= 5  
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
                        self.reward += 20  
                        return True
            elif self.current_task.task_type == TaskType.RETRIEVE and isinstance(agent, OutputAgent):
                print(f"Forklift {self.unique_id} delivered pallet {self.current_pallet_id} to OutputAgent at {self.pos}")
                self.current_pallet_id = None
                self.carrying_pallet = False
                self.reward += 20  
                return True

        print(f"Forklift {self.unique_id} failed to unload at {self.pos}")
        self.reward -= 10  
        return False
    
class WarehouseModel(Model):
    SYMBOL_AGENT_MAPPING = {
        'E': EmptyAgent,
        'B': BlockAgent,
        'R': RackAgent,
        'I': InputAgent,
        'O': OutputAgent,
        'C': ChargeStationAgent,
        'A': ForkliftAgent
    }
    
    def __init__(self, layout=None):
        super().__init__()
        if layout is None:
            layout = self.DEFAULT_LAYOUT

        self.completed_tasks = []
        height = len(layout)
        width = len(layout[0])

        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivation(self)

        self.input_agents = []
        self.output_agents = []
        self.rack_agents = []
        self.forklift_agents = []
        self.charge_stations = []
        self.step_duration_seconds = 2
        self.total_steps = 0
        self.lgv_mission_times = []
        self.pallets_moved = 0
        self.pallet_stationary_time = {}
        self.total_accumulation_time = 0  
        self.total_simulation_time = 0
        self.time_data_upf = []
        self.upf_data = []
        self.next_upf_collection_time = 200

        self.time_data_pallets = []
        self.pallets_delivered_over_time = []

        self.time_data_accumulation = []
        self.accumulation_time_over_time = []

        forklift_id_counter = 1  

        for y in range(height):
            row = layout[y]
            for x in range(width):
                char = row[x]
                pos = (x, height - y - 1)
                agent_class = self.SYMBOL_AGENT_MAPPING.get(char, EmptyAgent)

                if agent_class == ForkliftAgent:
                    agent_id = forklift_id_counter 
                    forklift_id_counter += 1       
                    agent = agent_class(agent_id, self, starting_position=pos)
                    self.forklift_agents.append(agent)
                else:
                    agent_id = f"{agent_class.__name__}_{x}_{y}"
                    agent = agent_class(agent_id, self)
                    if agent_class == RackAgent:
                        self.rack_agents.append(agent)
                    elif agent_class == InputAgent:
                        self.input_agents.append(agent)
                    elif agent_class == OutputAgent:
                        self.output_agents.append(agent)
                    elif agent_class == ChargeStationAgent:
                        self.charge_stations.append(agent)

                self.grid.place_agent(agent, pos)
                self.schedule.add(agent)

        self.task_manager = TaskManager(self)

    def get_agent_at_position(self, pos):
        cell_contents = self.grid.get_cell_list_contents(pos)
        for agent in cell_contents:
            if isinstance(agent, RackAgent):
                return agent
        return None

    def write_data_to_file(self):
        with open('Informacion.txt', 'w') as f:
            f.write("Datos de la Simulacion\n")
            f.write("======================\n\n")
            total_tasks = len(self.completed_tasks)
            f.write(f"Total de tareas completadas: {total_tasks}\n")
            f.write(f"Total de pallets movidos: {total_tasks}\n\n")

            f.write("Detalles de las tareas:\n")
            for task in self.completed_tasks:
                f.write(f"- Tipo de tarea: {task.task_type.value}\n")
                f.write(f"  Posicion de origen: {task.source_pos}\n")
                f.write(f"  Posicion de destino: {task.dest_pos}\n")
                f.write(f"  LGV asignado: {task.assigned_forklift.unique_id}\n")
                f.write(f"  Tiempo completado: {task.completed_time}\n\n")
            
            total_time_on_tasks = sum(forklift.time_on_task for forklift in self.forklift_agents)
            average_time_percentage = (total_time_on_tasks / (self.total_simulation_time * len(self.forklift_agents))) * 100

            pallets_retrieved = sum(1 for task in self.completed_tasks if task.task_type == TaskType.RETRIEVE)

            n = len(self.forklift_agents)
            
            T = self.total_simulation_time
            if T > 0 and n > 0:
                upf = (100 / (n * T)) * total_time_on_tasks
            else:
                upf = 0

            f.write(f"Tiempo total de simulacion: {self.total_simulation_time} segundos\n")
            f.write(f"Tiempo total en misiones: {total_time_on_tasks} segundos\n")
            f.write(f"Utilizacion del Porcentaje de la Flota (UPF): {upf:.2f}%\n")
            f.write(f"Cantidad de pallets entregados: {pallets_retrieved}\n")
            f.write(f"Tiempo de acumulacion de pallets: {self.total_accumulation_time} segundos\n")

    def generate_plots(self):
        plt.figure()
        plt.plot(self.time_data_upf, self.upf_data, marker='o')
        plt.title('Utilización del Porcentaje de la Flota (UPF)')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('UPF (%)')
        plt.grid(True)
        plt.savefig('upf_plot.png')
        plt.close()

        plt.figure()
        plt.plot(self.time_data_pallets, self.pallets_delivered_over_time)
        plt.title('Cantidad de pallets entregados')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Cantidad de pallets entregados')
        plt.grid(True)
        plt.savefig('pallets_delivered_plot.png')
        plt.close()

        plt.figure()
        plt.plot(self.time_data_accumulation, self.accumulation_time_over_time)
        plt.title('Tiempo de acumulación de pallets')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Tiempo de acumulación (segundos)')
        plt.grid(True)
        plt.savefig('accumulation_time_plot.png')
        plt.close()


    def step(self):
        print("\n=== Warehouse Step ===")
        self.task_manager.dRBug_print_status()
        self.total_simulation_time += self.step_duration_seconds

        if self.total_simulation_time >= self.next_upf_collection_time:
            total_time_on_tasks = sum(forklift.time_on_task for forklift in self.forklift_agents)
            n = len(self.forklift_agents)
            T = self.total_simulation_time
            upf = (100 / (n * T)) * total_time_on_tasks if T > 0 and n > 0 else 0

            self.time_data_upf.append(self.total_simulation_time)
            self.upf_data.append(upf)

            self.next_upf_collection_time += 200

        pallets_retrieved = sum(1 for task in self.completed_tasks if task.task_type == TaskType.RETRIEVE)
        self.time_data_pallets.append(self.total_simulation_time)
        self.pallets_delivered_over_time.append(pallets_retrieved)

        self.time_data_accumulation.append(self.total_simulation_time)
        self.accumulation_time_over_time.append(self.total_accumulation_time)

        all_tasks_complete = (
            len(self.task_manager.pending_tasks) == 0 and 
            len(self.task_manager.active_tasks) == 0
        )
        all_forklifts_idle = all(
            forklift.task_state == "IDLE" and 
            not forklift.carrying_pallet
            for forklift in self.forklift_agents
        )

        if not self.running:
            for forklift in self.forklift_agents:
                forklift.save_path_to_json()

        if all_tasks_complete and all_forklifts_idle:
            print("\n=== All tasks completed. Simulation finished ===")
            for forklift in self.forklift_agents:
                forklift.save_path_to_json()   
            with open('forklift_paths.json', 'r') as f:
                final_paths = json.load(f)
                for i, path in enumerate(final_paths):
                    if i > 0:
                        print()
                    print(json.dumps([path], separators=(',', ':')), end='')
            self.write_data_to_file()
            self.generate_plots()
            self.running = False
            return
    
        self.schedule.step()

def agent_portrayal(agent):
    if isinstance(agent, BlockAgent):
        portrayal = {
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "Color": "black",
            "w": 1,
            "h": 1
        }
        return portrayal
    elif isinstance(agent, RackAgent):
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
    elif isinstance(agent, ForkliftAgent):
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

layout = [
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' , # 1
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' , # 1   
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEOEEEEEEEEEEEE' , # 1
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' , # 2
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' , # 3
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' , # 4
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' , # 5
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' , # 6
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' , # 7
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' , # 8
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' , # 9
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' , # 10
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 11
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 12
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 13
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 14
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 15
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 16
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBREEEEEEEEEEEEEEEEEEEEEEEEE' ,  # 17
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 18
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 19
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 20
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 21
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 22
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 23
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 24
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 25
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 26
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 27
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 28
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 29
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 30
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 31
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERBB' ,  # 32
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERBB' ,  # 33
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERBB' ,  # 34
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERBB' ,  # 35
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 36
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 37
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 38
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 39
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 40
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 41
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 42
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 43
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 44
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 45
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 46
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 47
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 48
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 49
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 50
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 51
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 52
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 53
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 54
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 55
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 56
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 57
    'EEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 58
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERBB' ,  # 59
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERBB' ,  # 60
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERBB' ,  # 61
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERBB' ,  # 62
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 63
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 64
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 65
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 66
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 67
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 68
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 69
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 70
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 71
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 72
    'EEEEEEEEAEEEEEEEEEAEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 73
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 74
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 75
    'EEEEEEEEAEEEEEEEEEAEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 76
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 77
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 78
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 79
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 80
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 81
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB' ,  # 82
    'EEEEEEEEEEEEEEEEEEEEEEEEEERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBBBBRERBB',# 83
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE', # 84
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE', # 85
    'EEIEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE', # 86
    'EEEEEEEEEEEEEEEEEECCEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE', #87
    'EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE' #88
    ]

grid = CanvasGrid(agent_portrayal, len(layout[0]), len(layout), 1130, 900)
server = ModularServer(
    WarehouseModel,
    [grid],
    "Warehouse Layout",
    {"layout": layout}
)

if __name__ == '__main__':
    server.port = 8521
    server.launch()