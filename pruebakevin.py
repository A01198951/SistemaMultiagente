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
        if not cell_contents or all(isinstance(agent, (EmptyAgent, InputAgent, OutputAgent)) for agent in cell_contents):
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
