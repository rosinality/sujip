class Updater:
    def __init__(self, scheduler, optimizer):
        self.optimizer = optimizer

        if scheduler is not None:
            self.scheduler = scheduler(optimizer)

        else:
            self.scheduler = None

    def step(self):
        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.step()
        self.optimizer.zero_grad()

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
